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
"""Manage main command."""
import logging
import os
import numpy as np
import tvm
from tvm.contrib import graph_executor
from tvm.relay.quantize.quantize_hhb import detect_quantized_model

from .arguments_manage import ArgumentFilter
from .frontend_manage import import_model, insert_preprocess_node
from .common import ensure_dir, AttributeDict, HHBException, generate_config_file
from .hhbir_manage import (
    HHBRelayIR,
    HHBQNNIR,
    HHBFloatCodegenIR,
    HHBX86QnnCodegenIR,
    HHBBoardQnnCodegenIR,
    HHBBoardBuildRuntime,
    get_input_info_from_relay,
    get_output_info_from_relay,
    reorder_pixel_format,
)
from .quantization_manage import (
    collect_quantization_config,
    set_quantize_params_by_board,
    get_config_dict,
    update_hybrid_layer,
    ignore_layers_from_auto_quant,
)
from .preprocess_manage import (
    collect_preprocess_config,
    set_preprocess_params,
    DatasetLoader,
    parse_mean,
)
from .codegen_manage import collect_codegen_config, set_codegen_config
from .simulate_manage import inference_model, inference_elf


LOG = 25
logger = logging.getLogger("HHB")


def generate_dataset(args_filter: ArgumentFilter):
    """Generate preprocessed dataset"""
    input_shape = args_filter.filtered_args.input_shape
    assert len(input_shape) == 1, "Unsupport for multi-inputs"

    # filter arguments and prepare all needed args
    all_filters = [
        collect_preprocess_config,
        set_preprocess_params,
    ]
    extra_args = AttributeDict()
    extra_args.input_shape = input_shape
    args_filter.filter_argument(all_filters, extra=extra_args)
    args = args_filter.filtered_args

    if not args.simulate_data:
        raise HHBException("Please set simulate data by --simulate-data\n")

    logger.info("get simulate data from %s", args.simulate_data)
    dl = DatasetLoader(
        args.simulate_data,
        args.preprocess_config,
        input_shape,
        [""],
    )

    index = 0
    dataset = dl.get_data()
    for d in dataset:
        data = list(d.values())[0]
        filename = os.path.basename(dl.all_file_path[index])
        filename, _ = os.path.splitext(filename)
        data = data.astype("float32")
        data.tofile(os.path.join(args.output, filename + ".tensor"), "\n")
        data.tofile(os.path.join(args.output, filename + ".bin"))

        index += 1


def driver_main_command(args_filter: ArgumentFilter):
    """Driver main command"""
    args = args_filter.filtered_args
    args.output = ensure_dir(args.output)

    if args.generate_config:
        generate_config_file(os.path.join(args.output, "cmd_params.yml"))
        if not (args.E or args.Q or args.C or args.simulate):
            return 0

    if args.generate_dataset:
        generate_dataset(args_filter)
        return 0

    if not (args.E or args.Q or args.C or args.D or args.S or args.simulate):
        raise HHBException("No subcommand select.\n")

    #######################################################################
    #
    # Execute '-E' command
    #
    logger.log(LOG, "Start import model.")
    mod, params = import_model(
        args.model_file, args.model_format, args.input_name, args.input_shape, args.output_name
    )

    if args.reorder_pixel_format:
        mod, params = reorder_pixel_format(mod, params)

        if args.pixel_format == "RGB":
            args.pixel_format = "BGR"
        else:
            args.pixel_format = "RGB"

        if args.data_mean:
            args.data_mean = parse_mean(args.data_mean)
            args.data_mean = args.data_mean[::-1]

    logger.debug("Relay model:")
    logger.debug(mod["main"])
    logger.log(LOG, "Model import completed! ")
    relay_ir = HHBRelayIR()
    relay_ir.set_model(mod, params)

    if args.E or args.save_temps:
        relay_ir.save_model(args.output)

    if args.E:
        return 0

    if args.board == "th1520" and (args.hybrid_computing or args.auto_hybrid_quantization):
        args.board = "hth1520"

    #######################################################################
    #
    # Execute '-Q' command
    #
    input_name_list, input_shape_list, _ = get_input_info_from_relay(mod, params)
    output_shape_list, _ = get_output_info_from_relay(mod, params)

    if not args.no_quantize:
        detected_quant_type = detect_quantized_model(mod)
        if detected_quant_type:
            if len(detected_quant_type) == 1:
                detected_quant_type = detected_quant_type.pop()
                if detected_quant_type == "uint8":
                    args.quantization_scheme = "uint8_asym"
                elif detected_quant_type == "int8":
                    args.quantization_scheme = "int8_asym"
                else:
                    raise HHBException(
                        "Unsupport quantization type:{}.\n".format(detected_quant_type)
                    )
                logger.log(
                    LOG,
                    "Detect that current model has been quantized with {}, "
                    "--quantization-scheme will be overwritten to {}".format(
                        detected_quant_type, args.quantization_scheme
                    ),
                )
            else:
                logger.warning("Detect that there are multi quantization types in model.")
    # filter arguments and prepare all needed args
    all_filters = [
        collect_preprocess_config,
        set_preprocess_params,
        collect_quantization_config,
        set_quantize_params_by_board,
        collect_codegen_config,
        set_codegen_config,
    ]
    extra_args = AttributeDict()
    extra_args.input_shape = input_shape_list
    extra_args.input_num = len(input_shape_list)
    extra_args.output_num = len(output_shape_list)
    extra_args.model_save = args.model_save
    args_filter.filter_argument(all_filters, extra=extra_args)
    args = args_filter.filtered_args

    # add preprocess node into mod
    if args.preprocess_config.add_preprocess_node:
        mod, params = insert_preprocess_node(
            mod, params, args.preprocess_config.data_mean, args.preprocess_config.data_scale
        )
        logger.debug("Insert preprocess node into model successfully!")

    config_dict = get_config_dict(args)

    if not args.no_quantize:
        logger.log(LOG, "Start quantization.")
        dataset_list = []
        if args.calibrate_dataset:
            logger.log(LOG, "get calibrate dataset from %s", args.calibrate_dataset)
            dl = DatasetLoader(
                args.calibrate_dataset, args.preprocess_config, input_shape_list, input_name_list
            )
            dataset = dl.get_data()
            for d in dataset:
                dataset_list.append(d)

        qnn_ir = HHBQNNIR()
        qnn_ir.convert((mod, params), config_dict, dataset_list, args.board)
        logger.log(LOG, "Quantization completed!")
        if args.Q or args.save_temps:
            qnn_ir.save_model(args.output, args.preprocess_config, config_dict)
        if args.Q:
            return 0
    else:
        if args.Q:
            raise HHBException("can not set '-Q' and '--no_quantize' at the same time.\n")

    #######################################################################
    #
    # Execute '-C' command
    #
    target_board_list = (
        "anole",
        "th1520",
        "hth1520",
        "e907",
        "c906",
        "rvm",
        "c908",
        "c920",
        "x86_ref",
    )
    config_dict = get_config_dict(args)
    if config_dict["auto_hybrid_quantization"]:
        update_hybrid_layer(config_dict, args.output)

        limited_layer = ignore_layers_from_auto_quant(qnn_ir.get_model()[0], config_dict["target"])
        logger.info(
            "These layers will be removed from hybrid quant list: {}".format(
                set(config_dict["hybrid_layer_name"]) & set(limited_layer)
            )
        )
        config_dict["hybrid_layer_name"] = list(
            set(config_dict["hybrid_layer_name"]) - set(limited_layer)
        )

        if args.quantize_config.ignore_hybrid_layer:
            config_dict["hybrid_layer_name"] = list(
                set(config_dict["hybrid_layer_name"])
                - set(args.quantize_config.ignore_hybrid_layer)
            )
    th1520_input_fix_size = args.codegen_config.th1520_input_fix_size

    if args.board == "x86_ref":
        if args.no_quantize:
            x86_codegen_ir = HHBFloatCodegenIR()
            x86_codegen_ir.convert((mod, params), args.board, args.opt_level)
            x86_codegen_ir.save_model(args.output)
        else:
            x86_codegen_ir = HHBX86QnnCodegenIR()
            x86_codegen_ir.convert(
                qnn_ir.get_model(), args.board, args.opt_level, args.output, config_dict
            )

    if args.no_quantize:
        if args.board == "x86_ref":
            pass
        else:
            raise HHBException(
                "can not set '--no-quantize' with '--board {}'.\n".format(args.board)
            )
    else:
        if args.board in target_board_list:
            board_codegen_ir = HHBBoardQnnCodegenIR()

            board_codegen_ir.convert(
                qnn_ir.get_model(),
                args.board,
                args.opt_level,
                args.output,
                config_dict,
            )
        else:
            raise HHBException("unsupport for board: {}.\n".format(args.board))

    if args.C or args.D or args.S or args.save_temps:
        if args.board in target_board_list and not args.no_quantize:
            input_name_list, input_shape_list, _ = get_input_info_from_relay(
                qnn_ir.get_model()[0], None
            )
            hhb_gen = False
            if args.ahead_of_time == "intrinsic":
                hhb_gen = True
            board_codegen_ir.save_model(
                input_shape_list,
                output_shape_list,
                args.board,
                args.output,
                args.postprocess,
                args.codegen_config.model_save,
                args.codegen_config.without_preprocess,
                args.preprocess_config,
                args.codegen_config.multithread,
                args.codegen_config.input_memory_type,
                args.quantize_config.quantization_scheme,
                args.codegen_config,
                hhb_gen,
            )

            # save part data in calibrate dataset into tensor file
            data_count = 0
            for k in input_name_list:
                if not dataset_list:
                    break
                safe_k = k.replace("/", "_")
                v = dataset_list[0][k]
                v = v.astype("float32")
                if args.target_layout == "NHWC":
                    v = v.transpose([0, 2, 3, 1])
                v.tofile(os.path.join(args.output, safe_k + ".{}.tensor".format(data_count)), "\n")
                v.tofile(os.path.join(args.output, safe_k + ".{}.bin".format(data_count)))
                if len(th1520_input_fix_size) == 2:
                    v = np.pad(
                        v,
                        (
                            (0, 0),
                            (0, 0),
                            (0, int(th1520_input_fix_size[0]) - v.shape[2]),
                            (0, int(th1520_input_fix_size[1]) - v.shape[3]),
                        ),
                        "constant",
                    )
                    v.tofile(
                        os.path.join(args.output, safe_k + ".{}.pad.tensor".format(data_count)),
                        "\n",
                    )
                    v.tofile(os.path.join(args.output, safe_k + ".{}.pad.bin".format(data_count)))
                data_count += 1

    if args.C:
        return 0

    #######################################################################
    #
    # Execute '-D' command, build all source files into one elf
    #

    if args.D or args.S:
        intrinsic = False
        if args.ahead_of_time == "intrinsic":
            intrinsic = True
        platform_deploy = HHBBoardBuildRuntime(args.board, args.output, intrinsic, args.link_lib)

        # build all c source files to .o
        platform_deploy.build_c()
        # link_elf for linux platform
        platform_deploy.link_elf()
        # generate makefile
        if args.with_makefile:
            platform_deploy.generate_makefile()

        if args.board in ("th1520", "hth1520"):
            # for x86 simulate
            platform_deploy = HHBBoardBuildRuntime("th1520_x86", args.output)

            # build all c source files to .o
            platform_deploy.build_c()
            # link_elf for linux platform
            platform_deploy.link_elf("hhb_th1520_x86_runtime", "hhb_th1520_x86_jit")
            # generate makefile
            if args.with_makefile:
                platform_deploy.generate_makefile()

    if args.D:
        return 0

    #######################################################################
    #
    # Execute '-S' command
    #
    dl = DatasetLoader(
        args.simulate_data,
        args.preprocess_config,
        input_shape_list,
        input_name_list,
        target_layout=args.target_layout,
    )
    if args.S:
        dataset = dl.get_data()
        all_file_path = dl.all_file_path
        if args.board == "x86_ref":
            inference_elf("./hhb_runtime", dataset, input_name_list, all_file_path, args.output)
        elif args.board == "c906":
            inference_elf(
                "qemu-riscv64 -cpu c906fdv hhb_runtime",
                dataset,
                input_name_list,
                all_file_path,
                args.output,
            )
        elif args.board == "c908":
            inference_elf(
                "qemu-riscv64 -cpu c908v hhb_runtime",
                dataset,
                input_name_list,
                all_file_path,
                args.output,
            )
        elif args.board == "c920":
            inference_elf(
                "qemu-riscv64 -cpu c920 hhb_runtime",
                dataset,
                input_name_list,
                all_file_path,
                args.output,
            )
        else:
            raise HHBException("Unsupport to simulate for %s.\n", args.board)
        return 0

    #######################################################################
    #
    # Execute '--simulate' command
    #
    if not args.simulate_data:
        raise HHBException("Please set simulate data by --simulate-data.\n")

    logger.info("get simulate data from %s", args.simulate_data)

    ctx = tvm.cpu(0)
    if args.no_quantize:
        m = graph_executor.GraphModule(x86_codegen_ir.get_model()["default"](ctx))
    else:
        x86_codegen_ir.save_model(args.output)
        factory = x86_codegen_ir.get_factory()
        lib = x86_codegen_ir.get_lib(args.output)
        m = tvm.contrib.graph_executor.create(factory.get_graph_json(), lib, tvm.cpu(0))
        m.load_params(tvm.runtime.save_param_dict(factory.get_params()))

    inference_model(m, dl, args.postprocess, args.output)
