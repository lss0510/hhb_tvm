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
profile the imported model.
"""
import logging
import os

import tvm
from tvm import relay
from tvm.relay.quantize.quantize_hhb import _bind_params, check_bn_variance
from tvm.relay.quantize.quantize_hhb import optimization_phase0
from tvm.relay.quantize.quantize_hhb import save_const_output, _check_unsupported_ops
from tvm.relay.quantize._convert_to_csi import convert_to_csi_qnn, fuse_layer

from .core.frontend_manage import import_model
from .core.common import hhb_register_parse, ensure_dir, AttributeDict, HHBException
from .core.common import generate_config_file, ALL_ARGUMENTS_DESC, collect_arguments_info
from .core.common import hhb_deprecated_check, to_json, from_json
from .core.arguments_manage import (
    add_common_argument,
    add_import_argument,
    add_profiler_argument,
    ArgumentFilter,
    add_quantize_argument,
    add_hardware_argument,
    add_optimize_argument,
    add_codegen_argument,
    add_postprocess_argument,
)
from .core.profiler_manage import convert_tvm_trace2python, aitrace_options
from .core.profiler_manage import dump_profile_result, profile_trace_data, profile_acc_loss
from .core.quantization_manage import (
    collect_quantization_config,
    set_quantize_params_by_board,
    get_config_dict,
)
from .core.codegen_manage import (
    collect_codegen_config,
    set_codegen_config,
)
from .core.hhbir_manage import (
    get_input_info_from_relay,
    get_output_info_from_relay,
)

from .analysis.trace import merge_trace, HHBTrace, HHBIRTrace


# pylint: disable=invalid-name
LOG = 25
logger = logging.getLogger("HHB")


@hhb_register_parse
def add_profiler_parser(subparsers):
    """Include parser for 'profiler' subcommand"""

    parser = subparsers.add_parser("profiler", help="profile model")
    parser.set_defaults(func=driver_profiler)

    add_import_argument(parser)
    add_profiler_argument(parser)
    add_common_argument(parser)
    add_quantize_argument(parser)
    add_hardware_argument(parser)
    add_optimize_argument(parser)
    add_codegen_argument(parser)
    add_postprocess_argument(parser)

    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity")
    parser.add_argument(
        "-f", "--model-file", nargs="+", help="Path to the input model file, can pass multi files"
    )

    ALL_ARGUMENTS_DESC["profiler"] = collect_arguments_info(parser._actions)


def driver_profiler(args_filter: ArgumentFilter):
    """Driver profiler command"""
    args = args_filter.filtered_args
    args.output = ensure_dir(args.output)

    hhb_deprecated_check("--indicator", "3.0")

    if args.generate_config:
        generate_config_file(os.path.join(args.output, "cmd_profiler_params.yml"))

    if os.path.exists(args.model_file[0]) and args.model_file[0].endswith(".json"):
        # branch that deal with trace data

        if args.merge_trace:
            for path in args.model_file:
                if not os.path.exists(path) or not path.endswith(".json"):
                    raise HHBException(f"File is not exists or is not .json file: {path}")

            assert len(args.model_file) == 2, "Only support for 2 json files."
            merged_trace = merge_trace(args.model_file)
            target_path = os.path.join(args.output, "model_merge.trace.json")
            to_json(merged_trace.to_dict(), target_path)

            # just need to process the merged data
            profile_trace_data(
                merged_trace, args.profile_method, args.output, args.output_type, args.topk
            )
        elif "accuracy_loss" in args.profile_method:
            assert (
                len(args.model_file) == 2
            ), f"Analysis accuracy need two .json files, but get {len(args.model_file)}"
            trace1 = from_json(args.model_file[0])
            trace2 = from_json(args.model_file[1])

            display = False
            if "print" in args.output_type or "all" in args.output_type:
                display = True
            to_csv = None
            if "all" in args.output_type or "csv" in args.output_type:
                to_csv = os.path.join(args.output, "model_acc.csv")
            profile_acc_loss(trace1, trace2, topk=args.topk, to_csv=to_csv)
        else:
            for path in args.model_file:
                if not os.path.exists(path) or not path.endswith(".json"):
                    logger.warning("File is not exists or is not .json file: %s" % path)
                    continue

                json_data = from_json(path)
                if (
                    not json_data
                    or "otherData" not in json_data
                    or "source" not in json_data["otherData"]
                    or json_data["otherData"]["source"] not in ("hhb", "csinn")
                ):
                    logger.warning("Unsupport for %s" % path)
                    continue

                logger.log(LOG, "Processing %s..." % path)
                output_dir = os.path.join(args.output, os.path.splitext(os.path.basename(path))[0])
                output_dir = ensure_dir(output_dir)
                if "trace_type" in json_data["otherData"]:
                    # for relay/qnn
                    trace_data = HHBIRTrace().from_dict(json_data)

                    display = False
                    if "print" in args.output_type or "all" in args.output_type:
                        display = True
                    tmp_dir = None
                    if "all" in args.output_type or "csv" in args.output_type:
                        tmp_dir = output_dir
                    trace_data.profile(
                        profile_method=args.profile_method,
                        topk=args.topk,
                        display=display,
                        output_dir=tmp_dir,
                    )
                else:
                    # for chrome trace
                    trace_data = HHBTrace().from_dict(json_data)
                    profile_trace_data(
                        trace_data, args.profile_method, output_dir, args.output_type, args.topk
                    )

    else:
        # branch that deal with original model
        target_arch = args.arch
        if not target_arch:
            target_arch = args.ir_type
            hhb_deprecated_check("--ir-type", "3.0", "--arch")

        if target_arch == "relay":
            # relay ir should do InferType pass before profiling
            from tvm.relay import transform as _transform
            from tvm.ir import transform

            mod, params = import_model(
                args.model_file,
                args.model_format,
                args.input_name,
                args.input_shape,
                args.output_name,
            )

            options = aitrace_options(args.indicator, "")
            logger.debug('profile model with: "%s"', str(options))

            if params:
                mod["main"] = _bind_params(mod["main"], params)
                params = None

            opt_seq = [
                _transform.SimplifyInference(),
                _transform.DynamicToStatic(),
                _transform.FoldConstant(),
                _transform.SimplifyExpr(),
                _transform.InferType(),
            ]
            mod = transform.Sequential(opt_seq, opt_level=3)(mod)

            result = relay.analysis.get_aitrace_data(mod["main"], options)
            result = convert_tvm_trace2python(result)

            dump_profile_result(result, args.output_type, args.indicator, target_arch, args.output)

        elif target_arch == "qnn":
            if args.board == "unset":
                args.board = "x86_ref"
                logger.debug("reset defualt board=x86_ref")
            if args.quantization_scheme == "unset":
                args.quantization_scheme = "float32"
                logger.debug("reset defualt quantization_scheme=float32")

            logger.log(LOG, "Import model into relay ir - started...")
            mod, params = import_model(
                args.model_file,
                args.model_format,
                args.input_name,
                args.input_shape,
                args.output_name,
            )
            logger.log(LOG, "Import model into relay ir - finished")

            input_name_list, input_shape_list, _ = get_input_info_from_relay(mod, params)
            output_shape_list, _ = get_output_info_from_relay(mod, params)
            # filter arguments and prepare all needed args
            all_filters = [
                collect_quantization_config,
                set_quantize_params_by_board,
                collect_codegen_config,
                set_codegen_config,
            ]
            extra_args = AttributeDict()
            extra_args.input_shape = input_shape_list
            extra_args.input_num = len(input_shape_list)
            extra_args.output_num = len(output_shape_list)
            extra_args.model_save = "save_and_run"  # default value
            args_filter.filter_argument(all_filters, extra=extra_args)
            args = args_filter.filtered_args
            logger.log(LOG, "Convert model into qnn ir and optimize graph - started...")
            config_dict = get_config_dict(args)
            with tvm.transform.PassContext(
                opt_level=3, config={"relay.ext.csinn.options": config_dict}
            ):
                # optimize relay ir and convert to qnn ir
                if params:
                    mod["main"] = _bind_params(mod["main"], params)
                    params = None

                mod = optimization_phase0(mod)
                mod = check_bn_variance(mod)
                mod = save_const_output(mod, args.output)
                _ = _check_unsupported_ops(args.board, mod)
                mod = convert_to_csi_qnn(
                    mod,
                    None,
                    config_dict["channel_quantization"],
                    config_dict["channel_quantization_ratio_threshold"],
                )
                mod = fuse_layer(mod, config_dict)
                mod = relay.transform.InferType()(mod)

            logger.log(LOG, "Convert model into qnn ir and optimize graph - finished")
            options = aitrace_options(args.indicator, "")
            logger.debug('profile model with: "%s"', str(options))
            result = relay.analysis.qnn_aitrace_data(mod["main"], options)
            result = convert_tvm_trace2python(result)
            dump_profile_result(result, args.output_type, args.indicator, target_arch, args.output)

        elif target_arch == "npuperf":
            from .tools.npuperf_profiling import generate_trace

            if args.board == "unset":
                args.board = "x86_ref"
                logger.debug("reset defualt board=x86_ref")
            if args.quantization_scheme == "unset":
                args.quantization_scheme = "float32"
                logger.debug("reset defualt quantization_scheme=float32")

            logger.log(LOG, "Import model into relay ir - started...")
            mod, params = import_model(
                args.model_file,
                args.model_format,
                args.input_name,
                args.input_shape,
                args.output_name,
            )
            logger.log(LOG, "Import model into relay ir - finished")

            input_name_list, input_shape_list, _ = get_input_info_from_relay(mod, params)
            output_shape_list, _ = get_output_info_from_relay(mod, params)
            # filter arguments and prepare all needed args
            all_filters = [
                collect_quantization_config,
                set_quantize_params_by_board,
                collect_codegen_config,
                set_codegen_config,
            ]
            extra_args = AttributeDict()
            extra_args.input_shape = input_shape_list
            extra_args.input_num = len(input_shape_list)
            extra_args.output_num = len(output_shape_list)
            extra_args.model_save = "save_and_run"  # default value
            args_filter.filter_argument(all_filters, extra=extra_args)
            args = args_filter.filtered_args

            logger.log(LOG, "Convert model into qnn ir and optimize graph - started...")
            config_dict = get_config_dict(args)
            with tvm.transform.PassContext(
                opt_level=3, config={"relay.ext.csinn.options": config_dict}
            ):
                # optimize relay ir and convert to qnn ir
                if params:
                    mod["main"] = _bind_params(mod["main"], params)
                    params = None

                mod = optimization_phase0(mod)
                mod = check_bn_variance(mod)
                mod = save_const_output(mod, args.output)
                _ = _check_unsupported_ops(args.board, mod)
                mod = convert_to_csi_qnn(
                    mod,
                    None,
                    config_dict["channel_quantization"],
                    config_dict["channel_quantization_ratio_threshold"],
                )
                mod = fuse_layer(mod, config_dict)
                mod = relay.transform.InferType()(mod)

            logger.log(LOG, "Convert model into qnn ir and optimize graph - finished")

            logger.log(LOG, "Generate model data for npuperf - started...")
            from .tools.npuperf_profiling import convert_qnn_ir_to_npm_input

            model_data = convert_qnn_ir_to_npm_input(mod, target_layout=config_dict["layout"])
            logger.log(LOG, "Generate model data for npuperf - finished")

            if logger.level <= logging.DEBUG:
                to_json(
                    model_data,
                    os.path.join(args.output, "model_qnn.npuperf.json"),
                    with_format=True,
                )

            save_temps = True if logger.level <= logging.DEBUG else False
            trace_data = generate_trace(
                model_data, args.arch_config, save_temps=save_temps, output_dir=args.output
            )
            to_json(
                trace_data, os.path.join(args.output, "model.npuperf.trace.json"), with_format=True
            )
        else:
            raise HHBException(f"Unsupport for {target_arch}")
