# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=unnecessary-comprehension
"""
Simulate the imported model in x86.
"""
import logging
import os
import time

import numpy as np

import tvm
from tvm.contrib import graph_executor

from .core.frontend_manage import import_model
from .core.common import hhb_register_parse, HHBException, ensure_dir, AttributeDict
from .core.common import generate_config_file, ALL_ARGUMENTS_DESC, collect_arguments_info
from .core.arguments_manage import (
    add_preprocess_argument,
    add_common_argument,
    add_simulate_argument,
    add_postprocess_argument,
    add_import_argument,
    add_optimize_argument,
    add_quantize_argument,
    add_hardware_argument,
    add_codegen_argument,
    ArgumentFilter,
)
from .core.hhbir_manage import (
    HHBRelayIR,
    HHBQNNIR,
    HHBFloatCodegenIR,
    HHBX86QnnCodegenIR,
    get_input_info_from_relay,
)
from .core.quantization_manage import (
    collect_quantization_config,
    set_quantize_params_by_board,
    get_config_dict,
)
from .core.preprocess_manage import (
    collect_preprocess_config,
    set_preprocess_params,
    DatasetLoader,
)
from .core.codegen_manage import collect_codegen_config


# pylint: disable=invalid-name
logger = logging.getLogger("HHB")


@hhb_register_parse
def add_benchmark_parser(subparsers):
    """Include parser for 'benchmark' subcommand"""

    parser = subparsers.add_parser("benchmark")
    parser.set_defaults(func=driver_benchmark)

    parser.add_argument(
        "--reference-label", metavar="", type=str, help="The true labels of test dataset."
    )
    parser.add_argument(
        "--print-interval",
        metavar="",
        type=int,
        default=100,
        help="Print log every time how many images are inferred",
    )
    parser.add_argument("--save-temps", action="store_true", help="Save temp files.")
    parser.add_argument(
        "--no-quantize", action="store_true", help="If set, don't quantize the model."
    )

    add_import_argument(parser)
    add_quantize_argument(parser)
    add_hardware_argument(parser)
    add_simulate_argument(parser)
    add_preprocess_argument(parser)
    add_postprocess_argument(parser)
    add_common_argument(parser)
    add_optimize_argument(parser)
    add_codegen_argument(parser)

    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity")
    parser.add_argument(
        "FILE", nargs="+", help="Path to the input model file, can pass multi files"
    )

    ALL_ARGUMENTS_DESC["benchmark"] = collect_arguments_info(parser._actions)


def driver_benchmark(args_filter: ArgumentFilter):
    """Driver main command"""
    args = args_filter.filtered_args
    args.output = ensure_dir(args.output)

    if args.generate_config:
        generate_config_file(os.path.join(args.output, "cmd_benchmark_params.yml"))

    mod, params = import_model(
        args.FILE, args.model_format, args.input_name, args.input_shape, args.output_name
    )
    relay_ir = HHBRelayIR()
    relay_ir.set_model(mod, params)

    input_name_list, input_shape_list, _ = get_input_info_from_relay(mod, params)
    # filter arguments and prepare all needed args
    all_filters = [
        collect_preprocess_config,
        set_preprocess_params,
        collect_quantization_config,
        set_quantize_params_by_board,
        collect_codegen_config,
    ]
    extra_args = AttributeDict()
    extra_args.input_shape = input_shape_list
    args_filter.filter_argument(all_filters, extra=extra_args)
    args = args_filter.filtered_args

    config_dict = get_config_dict(args)
    if not args.no_quantize:
        dataset_list = []
        if args.calibrate_dataset:
            logger.info("get calibrate dataset from %s", args.calibrate_dataset)
            dl = DatasetLoader(
                args.calibrate_dataset, args.preprocess_config, input_shape_list, input_name_list
            )
            dataset = dl.get_data()
            for d in dataset:
                dataset_list.append(d)

        qnn_ir = HHBQNNIR()
        qnn_ir.convert((mod, params), config_dict, dataset_list)

    if args.board == "x86_ref":
        if args.no_quantize:
            x86_codegen_ir = HHBFloatCodegenIR()
            x86_codegen_ir.convert((mod, params), args.board, args.opt_level)
        else:
            x86_codegen_ir = HHBX86QnnCodegenIR()
            x86_codegen_ir.convert(
                qnn_ir.get_model(), args.board, args.opt_level, args.output, config_dict
            )
    else:
        raise HHBException("can not simulate th1520.")

    ctx = tvm.cpu(0)
    if args.no_quantize:
        m = graph_executor.GraphModule(x86_codegen_ir.get_model()["default"](ctx))
    else:
        factory = x86_codegen_ir.get_factory()
        lib = x86_codegen_ir.get_lib(args.output)
        m = tvm.contrib.graph_executor.create(factory.get_graph_json(), lib, tvm.cpu(0))
        m.load_params(tvm.runtime.save_param_dict(factory.get_params()))
    dl = DatasetLoader(
        args.simulate_data,
        args.preprocess_config,
        input_shape_list,
        input_name_list,
    )
    dataset = dl.get_data()

    # prepare inference labels
    # if args.simulate_data is None:
    #     raise HHBException("Please specify validate dataset directory.")
    imgname2label = {}
    if not args.reference_label:
        raise HHBException("Please specify validate label.")
    with open(args.reference_label, "r") as f:
        for line in f:
            tmp = line.strip().split(" ")
            imgname = tmp[0]
            label = int(tmp[-1])
            imgname2label[imgname] = label

    inter_log_list = ["Filename \t\t\t\t\t\t\t Top1 \t\t\t Top5\n"]
    top1 = 0
    top5 = 0

    index = 0
    t_total_start = time.time()
    t_mid_start = t_total_start
    t_total = 0
    for data in dataset:
        inter_log = ""
        m.run(**data)
        output = m.get_output(0).asnumpy()

        output = np.squeeze(output)
        idx = np.argsort(output)
        idx = idx[::-1]
        label = imgname2label[dl.all_file_path[index].strip().split("/")[-1]]
        if output.size == 1001:
            label += 1
        inter_log += os.path.basename(dl.all_file_path[index]) + " \t\t "
        if idx[0] == label:
            top1 += 1
            inter_log += "true" + " \t\t\t "
        else:
            inter_log += "false" + " \t\t\t "
        if label in idx[:5]:
            top5 += 1
            inter_log += "true\n"
        else:
            inter_log += "false\n"

        inter_log_list.append(inter_log)
        index += 1
        if index % args.print_interval == 0:
            t_mid_end = time.time()
            print(
                "num-{} top1:{}, top5:{}, time cost: {}s".format(
                    index, (top1 / index), (top5 / index), (t_mid_end - t_mid_start)
                )
            )
            t_total += t_mid_end - t_mid_start
            t_mid_start = t_mid_end

            if args.save_temps:
                args.output = ensure_dir(args.output)
            with open(os.path.join(args.output, "inter_results.log"), "w") as f:
                f.writelines(inter_log_list)
    print("Total time cost: {}s".format(t_total))
