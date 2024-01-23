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
from typing import List, Union
import copy

import numpy as np
import onnx
from onnxsim import simplify

import tvm
from tvm import relay
from tvm.contrib import graph_executor
from tvm.relay.quantize.quantize_hhb import _bind_params, check_bn_variance
from tvm.relay.quantize.quantize_hhb import optimization_phase0, InsertNOp, InsertRelu
from tvm.relay.quantize.quantize_hhb import save_const_output, _check_unsupported_ops
from tvm.relay.quantize._convert_to_csi import convert_to_csi_qnn, fuse_layer
from tvm.relay.quantize.qnn2onnx import qnn_to_onnx
from tvm.relay.quantize.qnn_transform import QNNConvertReshapeToFlatten

from .llm_manage import llm_import, llm_quantize
from .arguments_manage import ArgumentFilter, Config
from .frontend_manage import import_model, insert_preprocess_node, get_io_info_from_onnx
from .common import ensure_dir, AttributeDict, HHBException, generate_config_file, to_json
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
    get_quant_scheme_from_qnn,
    convert_per_channel_scheme,
)
from .preprocess_manage import (
    collect_preprocess_config,
    set_preprocess_params,
    DatasetLoader,
    parse_mean,
)
from .codegen_manage import collect_codegen_config, set_codegen_config
from .simulate_manage import inference_model, inference_elf
from .profiler_manage import aitrace_options, convert_tvm_trace2python, dump_profile_result

from hhb.analysis.trace import HHBIRTrace


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


def optimize_model(
    model_file: List[str],
    opt_tool: str,
    arguments: Union[AttributeDict, Config],
):
    """
    .onnx/.tf/.caffemodel -> relay -> optimize -> .onnx -> ppq -> relay -> qnn

    .onnx -> ppq -> relay -> qnn

    Parameters
    ----------
    model_file : List[str]
        Model files
    opt_tool : str
        Tool to optimize model.
    opt_level : int
        Optimization level:
    """
    if opt_tool != "ppq":
        return model_file
    assert isinstance(
        arguments, (AttributeDict, Config)
    ), f"Parameters 'arguments' must be AttributeDict or Config, but get {type(arguments)}"
    assert isinstance(
        model_file, (tuple, list)
    ), f"model_file should be list, but get {type(model_file)}"
    is_onnx = False
    if len(model_file) == 1 and os.path.splitext(model_file[0])[-1] == ".onnx":
        is_onnx = True
    args = arguments if isinstance(arguments, AttributeDict) else arguments.convert_to_arguments()
    opt_level = args.opt_level

    new_model_path = model_file
    new_input_name = args.input_name
    new_input_shape = args.input_shape
    new_output_name = args.output_name
    if opt_level == 3 or not is_onnx:
        logger.log(
            LOG,
            "Current model is not from onnx or you set opt_level=3, "
            "so start to optimize model with qnn...",
        )
        # convert to relay ir
        mod, params = import_model(
            model_file,
            args.model_format,
            new_input_name,
            new_input_shape,
            new_output_name,
        )
        inter_new_args = copy.deepcopy(args)
        inter_filter = ArgumentFilter(inter_new_args)
        all_filters = [
            collect_quantization_config,
            set_quantize_params_by_board,
            collect_codegen_config,
            set_codegen_config,
        ]
        extra_args = AttributeDict()
        extra_args.input_shape = new_input_shape
        extra_args.input_num = len(new_input_shape)
        extra_args.output_num = len(new_output_name)
        inter_filter.filter_argument(all_filters, extra=extra_args)
        inter_args = inter_filter.filtered_args
        inter_config = get_config_dict(inter_args)

        with tvm.transform.PassContext(
            opt_level=3, config={"relay.ext.csinn.options": inter_config}
        ):
            # optimize relay ir and convert to qnn ir
            if params:
                mod["main"] = _bind_params(mod["main"], params)
                params = None

            mod = optimization_phase0(mod)
            if args.board in ("th1520", "hth1520") and args.quantization_scheme not in [
                "int16_sym",
                "int8_sym",
            ]:
                mod = InsertNOp(mod)

            if args.board in ("th1520", "hth1520"):
                # fix sigmoid + mul acc bug in th1520 npu
                mod = InsertRelu(mod)
            mod = check_bn_variance(mod)
            mod = save_const_output(mod, args.output)
            _ = _check_unsupported_ops(args.board, mod)
            mod = convert_to_csi_qnn(
                mod,
                None,
                inter_config["channel_quantization"],
                inter_config["channel_quantization_ratio_threshold"],
            )
            mod = fuse_layer(mod, inter_config)
            mod = QNNConvertReshapeToFlatten()(mod)
            mod = relay.transform.InferType()(mod)

        # convert to onnx
        qnn_onnx_path = os.path.join(args.output, "model_qnn_opt.onnx")
        onnx_model = qnn_to_onnx(mod, {}, "qnn_csi")
        # simplify onnx
        onnx_model_sim, check = simplify(onnx_model)
        if check:
            onnx.save(onnx_model_sim, qnn_onnx_path)
        else:
            onnx.save(onnx_model, qnn_onnx_path)
            logger.warning("Fail to optimize onnx with onnxsim, back to qnn onnx.")
        logger.log(LOG, "Optimized model with qnn optimization is save in %s", qnn_onnx_path)

        new_model_path = [qnn_onnx_path]
        new_input_name, new_input_shape, new_output_name, _ = get_io_info_from_onnx(qnn_onnx_path)

    # create calibrate dataset
    args_filter = ArgumentFilter(args)
    all_filters = [
        collect_preprocess_config,
        set_preprocess_params,
    ]
    extra_args = AttributeDict()
    extra_args.input_shape = new_input_shape
    args_filter.filter_argument(all_filters, extra=extra_args)
    args = args_filter.filtered_args

    dataset_list = []
    dl = DatasetLoader(
        args.calibrate_dataset, args.preprocess_config, new_input_shape, new_input_name
    )
    dataset = dl.get_data()
    for d in dataset:
        inter = []
        for name in new_input_name:
            inter.append(d[name])
        dataset_list.append(inter)

    device = args.quant_device
    # quantize model with ppq
    from ..tools.ppq_quantization import quantize_ppq
    from ppq.api import ENABLE_CUDA_KERNEL

    if device == "cuda":
        with ENABLE_CUDA_KERNEL():
            new_model_path, _ = quantize_ppq(
                new_model_path[0],
                new_input_shape,
                args,
                dataset_list,
                batch_size=args.cali_batch,
                device=device,
                output_dir=args.output,
                target=args.board,
            )
    else:
        new_model_path, _ = quantize_ppq(
            new_model_path[0],
            new_input_shape,
            args,
            dataset_list,
            batch_size=args.cali_batch,
            device=device,
            output_dir=args.output,
            target=args.board,
        )
    return new_model_path


def print_model_info(model_file, input_name, input_shape, output_name, prefix=""):
    """Print model info"""
    info_str = "\n---------" + f"{prefix} Model info ---------\n"
    info_str += f"Model path: {model_file}\n"
    info_str += f"Input name: {input_name}\n"
    info_str += f"Input shape: {input_shape}\n"
    info_str += f"Output name: {output_name}\n"

    logger.log(LOG, info_str)


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

    # optimize model
    if args.quantization_tool == "ppq":
        print_model_info(
            args.model_file,
            args.input_name,
            args.input_shape,
            args.output_name,
            "Before optimization",
        )
        ppq_quantized_file = optimize_model(args.model_file, args.quantization_tool, args)
        # update info
        input_name, input_shape, output_name, _ = get_io_info_from_onnx(ppq_quantized_file)
        args.input_name = input_name
        args.input_shape = input_shape
        args.output_name = output_name
        args.model_file = [ppq_quantized_file]

        print_model_info(
            args.model_file,
            args.input_name,
            args.input_shape,
            args.output_name,
            "After optimization",
        )
        if args.Q:
            return 0

    #######################################################################
    #
    # Execute '-E' command
    #

    if args.llm:
        if args.E:
            logger.log(LOG, "Start import LLM")
            llm_import(args.model_file, args.output)
            logger.log(LOG, "Import model end...")
            return 0

        if args.Q:
            logger.log(LOG, "Start quantize LLM")
            llm_quantize(args.model_file, args.quantization_scheme, args.output)
            logger.log(LOG, "Quantize model end...")
            return 0

    logger.log(LOG, "Start import model.")
    mod, params = import_model(
        args.model_file, args.model_format, args.input_name, args.input_shape, args.output_name
    )

    if args.trace and "relay" in args.trace:
        # generate relay trace
        options = aitrace_options(["cal", "mem"], "")
        result = relay.analysis.get_aitrace_data(mod["main"], options)
        result = convert_tvm_trace2python(result)
        result = HHBIRTrace().imported_from(trace_type="relay", layers=result).to_dict()
        to_json(result, os.path.join(args.output, "model_relay.trace.json"), with_format=False)

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
        quant_scheme, is_per_channel, qnn_dtypes = get_quant_scheme_from_qnn(mod)
        if args.quantization_scheme == "unset" and qnn_dtypes:
            # there is quantize/dequantize op in module, so it is a quantized model.
            if quant_scheme and is_per_channel:
                coverted_quant_scheme = convert_per_channel_scheme(quant_scheme)
                if coverted_quant_scheme is None:
                    raise HHBException(f"Unsupport per-channel quantization for {quant_scheme}\n")
                else:
                    args.quantization_scheme = coverted_quant_scheme
                    logger.log(
                        LOG,
                        "Detect that current model has been quantized with per-channel {}, "
                        "then quantization_scheme is set {}".format(
                            quant_scheme, args.quantization_scheme
                        ),
                    )
            elif quant_scheme:
                args.quantization_scheme = quant_scheme
                logger.log(
                    LOG,
                    "Detect that current model has been quantized with {}, ".format(quant_scheme),
                )
            else:
                raise HHBException(
                    "Can not infer the quantization scheme from original model, please "
                    "specify it by --quantization-scheme.\n"
                )

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

        if args.trace and "qnn" in args.trace:
            # generate qnn trace
            options = aitrace_options(["cal", "mem"], "")
            result = relay.analysis.qnn_aitrace_data(qnn_ir.get_model()[0]["main"], options)
            result = convert_tvm_trace2python(result)
            result = HHBIRTrace().imported_from(trace_type="qnn", layers=result).to_dict()
            to_json(result, os.path.join(args.output, "model_qnn.trace.json"), with_format=False)

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
        "th1520",
        "hth1520",
        "e907",
        "c906",
        "rvm",
        "c908",
        "c920",
        "c920v2",
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
        elif args.board == "c920v2":
            inference_elf(
                "qemu-riscv64 -cpu c920v2 hhb_runtime",
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
