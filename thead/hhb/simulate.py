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

import tvm
from tvm import runtime
from tvm.contrib import graph_executor

from .core.arguments_manage import (
    add_simulate_argument,
    add_common_argument,
    add_postprocess_argument,
    add_preprocess_argument,
    add_codegen_argument,
    add_optimize_argument,
    ArgumentFilter,
)
from .core.common import (
    hhb_register_parse,
    HHBException,
    ensure_dir,
    AttributeDict,
    generate_config_file,
    ALL_ARGUMENTS_DESC,
    collect_arguments_info,
)
from .core.hhbir_manage import (
    guess_ir_type,
    HHBIRType,
    HHBRelayIR,
    HHBQNNIR,
    HHBFloatCodegenIR,
    HHBX86QnnCodegenIR,
)
from .core.preprocess_manage import collect_preprocess_config, set_preprocess_params, DatasetLoader
from .core.simulate_manage import inference_model
from .core.codegen_manage import collect_codegen_config
from .core.frontend_manage import insert_preprocess_node

# pylint: disable=invalid-name
logger = logging.getLogger("HHB")


def hhb_runner(codegen_ir, config):
    """Wrapper for hhb runner.

    Parameters
    ----------
    codegen_ir : HHBFloatCodegenIR or HHBX86QnnCodegenIR
        The codegened model.
    config : Config
        All config for HHB

    Returns
    -------
    ret : GraphModule
        The object that can be executed for x86_ref

    """
    hhb_config = config._cmd_config
    ctx = tvm.cpu(0)
    if isinstance(codegen_ir, HHBFloatCodegenIR):
        m = graph_executor.GraphModule(codegen_ir.get_model()["default"](ctx))
    elif isinstance(codegen_ir, HHBX86QnnCodegenIR):
        factory = codegen_ir.get_factory()
        lib = codegen_ir.get_lib(hhb_config.output)
        m = tvm.contrib.graph_executor.create(factory.get_graph_json(), lib, tvm.cpu(0))
        m.load_params(tvm.runtime.save_param_dict(factory.get_params()))
    else:
        raise HHBException("Can not create runner for {}".format(type(codegen_ir)))

    return m


def hhb_inference(graph_module, dataset):
    """Inference for hhb model.

    Parameters
    ----------
    graph_module : GraphModule
        The object that can be executed for x86_ref
    dataset : Dict[str, numpy.ndarray]
        The input data

    Returns
    -------
    output : List[numpy.ndarray]
        The output data.
    """
    graph_module.run(**dataset)
    output = []
    for i in range(graph_module.get_num_outputs()):
        out = graph_module.get_output(i).numpy()
        output.append(out)
    return output


@hhb_register_parse
def add_simulate_parser(subparsers):
    """Include parser for 'simulate' subcommand"""

    parser = subparsers.add_parser("simulate", help="Simulate the imported model")
    parser.set_defaults(func=driver_simulate)

    add_simulate_argument(parser)
    add_preprocess_argument(parser)
    add_postprocess_argument(parser)
    add_optimize_argument(parser)
    add_codegen_argument(parser)
    add_common_argument(parser)

    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity")
    parser.add_argument("FILE", help="Directory to the model file")

    ALL_ARGUMENTS_DESC["simulate"] = collect_arguments_info(parser._actions)


def driver_simulate(args_filter: ArgumentFilter):
    """Driver simulate command"""
    args = args_filter.filtered_args
    if not os.path.exists(args.FILE) or not os.path.isdir(args.FILE):
        raise HHBException("The directory is not exists: {}\n".format(args.FILE))
    model_type = guess_ir_type(args.FILE)
    logger.debug("Infer the ir type: %s in %s" % (HHBIRType.TYPE2NAME[model_type], args.FILE))

    args.output = ensure_dir(args.output)

    x86_codegen_ir = None
    quantize_config = None
    if args.board == "x86_ref":
        if model_type == HHBIRType.RELAY:
            # get relay ir
            relay_ir = HHBRelayIR()
            relay_ir.load_model(args.FILE)
            input_module = relay_ir.get_model()

            # add preprocess node into mod
            all_filters = [
                collect_preprocess_config,
            ]
            args_filter.filter_argument(all_filters)
            args = args_filter.filtered_args
            if args.preprocess_config.add_preprocess_node:
                input_mod, input_params = insert_preprocess_node(
                    input_module[0],
                    input_module[1],
                    args.preprocess_config.data_mean,
                    args.preprocess_config.data_scale,
                )
                input_module = (input_mod, input_params)
                logger.debug("Insert preprocess node into model successfully!")

            # convert to float codegen ir
            float_codegen_ir = HHBFloatCodegenIR()
            float_codegen_ir.convert(input_module, args.board, args.opt_level)
            float_codegen_ir.save_model(args.output)
            x86_codegen_ir = float_codegen_ir
        elif model_type == HHBIRType.QNN:
            # get qnn ir
            qnn_ir = HHBQNNIR()
            qnn_ir.load_model(args.FILE)
            input_module = qnn_ir.get_model()
            # convert to x86 qnn codegen ir
            x86_qnn_codegen_ir = HHBX86QnnCodegenIR()
            quantize_config = x86_qnn_codegen_ir.get_quant_env(
                os.path.join(args.FILE, qnn_ir.info_file)
            )
            x86_qnn_codegen_ir.convert(
                input_module, args.board, args.opt_level, args.output, quantize_config
            )
            x86_qnn_codegen_ir.save_model(args.output)
            x86_codegen_ir = x86_qnn_codegen_ir
        else:
            raise HHBException("unsupport for IR type: {}".format(HHBIRType.TYPE2NAME[model_type]))
    else:
        raise HHBException("Only x86_ref support for simulation!")

    if not args.simulate_data:
        raise HHBException("Please set simulate data by --simulate-data.\n")

    logger.info("get simulate data from %s", args.simulate_data)

    ctx = tvm.cpu(0)
    if model_type == HHBIRType.RELAY:
        m = graph_executor.GraphModule(x86_codegen_ir.get_model()["default"](ctx))
    else:
        factory = x86_codegen_ir.get_factory()
        lib = x86_codegen_ir.get_lib(args.output)
        m = tvm.contrib.graph_executor.create(factory.get_graph_json(), lib, tvm.cpu(0))
        m.load_params(tvm.runtime.save_param_dict(factory.get_params()))

    # filter arguments and prepare all needed args
    all_filters = [
        collect_preprocess_config,
        set_preprocess_params,
        collect_codegen_config,
    ]
    extra_args = AttributeDict()
    extra_args.input_shape = x86_codegen_ir.info_dict["input_shape_list"]
    args_filter.filter_argument(all_filters, extra=extra_args)
    args = args_filter.filtered_args

    target_layout = "NCHW"
    if quantize_config:
        target_layout = quantize_config["layout"]
    dl = DatasetLoader(
        args.simulate_data,
        args.preprocess_config,
        x86_codegen_ir.info_dict["input_shape_list"],
        x86_codegen_ir.info_dict["input_name_list"],
        target_layout=target_layout,
    )
    inference_model(m, dl, args.postprocess, args.output)

    if args.generate_config:
        args.output = ensure_dir(args.output)
        generate_config_file(os.path.join(args.output, "cmd_simulate_params.yml"))
