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
""" HHB Command Line Tools """
import argparse
import logging
import sys
import json
import os
from typing import List

import onnx
from onnxsim import simplify

from tvm import relay
from tvm.contrib.target.onnx import to_onnx
from tvm.relay.quantize.quantize_hhb import _bind_params
from tvm.relay.quantize.quantize_hhb import optimization_phase0

from .core.arguments_manage import ArgumentManage, CommandType, HHBException, ArgumentFilter
from .core.arguments_manage import update_arguments_by_file
from .core.arguments_manage import Config
from .core.common import collect_arguments_info, to_json, from_json
from .core.common import ALL_ARGUMENTS_DESC, ensure_dir
from .core.hhbir_manage import HHBRelayIR, HHBQNNIR, reorder_pixel_format, HHBBoardBuildRuntime
from .core.preprocess_manage import hhb_preprocess
from .core.frontend_manage import insert_preprocess_node
from .core.profiler_manage import (
    aitrace_options,
    convert_tvm_trace2python,
    profile_trace_data,
    profile_acc_loss,
)
from .core.quantization_manage import get_quant_scheme_from_qnn, convert_per_channel_scheme
from .core.main_command_manage import print_model_info
from .core.frontend_manage import get_io_info_from_onnx
from .importer import hhb_import
from .quantizer import hhb_quantize
from .codegen import hhb_codegen
from .simulate import hhb_runner, hhb_inference

from hhb.analysis.trace import merge_trace, HHBTrace, HHBIRTrace


LOG = 25
logging.addLevelName(LOG, "LOG")


def set_debug_level(level="LOG"):
    """Set debug level.

    Parameters
    ----------
    level : str
        The debug level string, select from: LOG, DEBUG, INFO, WARNING and ERROR.

    """
    if level == "LOG":
        level_num = 25
    elif level == "INFO":
        level_num = 20
    elif level == "DEBUG":
        level_num = 10
    elif level == "WARNING":
        level_num = 30
    else:
        level_num = 40
    logging.basicConfig(
        format="[%(asctime)s] (%(name)s %(levelname)s): %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger("HHB")
    logger.setLevel(level_num)


class Compiler(object):
    """Object for compiling original DL models into specific format."""

    def __init__(self, board="unset") -> None:
        self.config = Config(board)
        self.relay_ir = None
        self.qnn_ir = None
        self.codegen_ir = None
        self.executor = None

        self._is_init = False

    def create_relay_ir(self):
        """Create empty HHBRelayIR object"""
        return HHBRelayIR()

    def create_qnn_ir(self):
        """Create empty HHBQNNIR object"""
        return HHBQNNIR()

    def _init_session(self):
        """Do some init operations for current session."""
        if self.relay_ir is None:
            raise HHBException("Please import model first.")
        if self._is_init:
            logger = logging.getLogger("HHB")
            logger.warning("Initialization completed, no need to initialize again.")
            return
        self.config.update_config_from_module(self.relay_ir)
        self._is_init = True

    def preprocess(self, data_path: str, is_generator=False):
        """Preprocess data with provided data files.

        Parameters
        ----------
        data_path : str
            Data file path.
        is_generator : bool, optional
            return generator for data if set.

        Returns
        -------
        out : list[dict[name, numpy.ndarray]] or generator
            Processed data.
        """
        if not self._is_init:
            raise HHBException("Please initialize session by _init_session() first.")
        self.config.generate_cmd_config()

        out = hhb_preprocess(data_path, self.config, is_generator)
        return out

    def import_model(
        self,
        path,
        model_format=None,
        input_name=None,
        input_shape=None,
        output_name=None,
        save_to_dir=None,
    ):
        """Import a model from a supported framework into relay ir.

        Parameters
        ----------
        path : str or list[str]
            Path to a model file. There may be two files(.caffemodel, .prototxt) for Caffe model
        model_format : str, optional
            A string representing input model format
        input_name : list[str], optional
            The names of input node in the graph
        input_shape : list[list[int]], optional
            The shape of input node in the graph
        output_name : list[str], optional
            The name of output node in the graph
        save_to_dir : str, optional
            save model into specified directory
        """
        # update config
        self.config.main.model_file.value = path
        if model_format:
            self.config.import_config.model_format.value = model_format
        if input_name:
            self.config.import_config.input_name.value = input_name
        if input_shape:
            self.config.import_config.input_shape.value = input_shape
        if output_name:
            self.config.import_config.output_name.value = output_name
        # update hhb ir
        self.relay_ir = hhb_import(
            path, model_format, input_name, input_shape, output_name, save_to_dir
        )

        self._init_session()

    def quantize(self, calibrate_data=None, save_to_dir=None):
        """Quantize model and convert relay ir into qnn ir.

        Parameters
        ----------
        calibrate_data : List[Dict[str, numpy.ndarray]]
            The calibration data for quantization. It includes batches of data.
        save_to_dir : str, optional
            save model into specified directory
        """
        if not self._is_init:
            raise HHBException("Please initialize session by _init_session() first.")
        if self.relay_ir is None:
            raise HHBException("Please import model by import_model() first.")

        logger = logging.getLogger("HHB")

        if self.config.quantize.quantization_tool.value == "ppq":
            model_path = self.config.main.model_file.value

            ppq_calibrate_data = []
            for d in calibrate_data:
                inter = []
                for name in self.config.import_config.input_name.value:
                    inter.append(d[name])
                ppq_calibrate_data.append(inter)

            is_onnx = False
            if isinstance(model_path, str):
                is_onnx = os.path.splitext(model_path)[-1] == ".onnx"
            if self.config.optimize.opt_level == 3 or not is_onnx:
                # convert original relay ir into onnx
                logger.log(
                    LOG,
                    "Original model is not onnx or need to be optimized, convert relay to onnx...",
                )
                mod, params = self.relay_ir.get_model()
                if params:
                    mod["main"] = _bind_params(mod["main"], params)
                    params = None
                # optimize relay ir
                mod = optimization_phase0(mod)
                mod = relay.transform.InferType()(mod)

                # convert to onnx
                if save_to_dir:
                    if not os.path.exists(save_to_dir):
                        os.makedirs(save_to_dir)
                    relay_onnx_path = os.path.join(save_to_dir, "model_relay_opt.onnx")
                else:
                    relay_onnx_path = "model_relay_opt.onnx"
                onnx_model = to_onnx(mod, {}, "relay")
                # simplify onnx
                onnx_model_sim, check = simplify(onnx_model)
                if check:
                    onnx.save(onnx_model_sim, relay_onnx_path)
                else:
                    onnx.save(onnx_model, relay_onnx_path)
                    logger.warning("Fail to optimize onnx with onnxsim, back to relay onnx.")
                logger.debug("New model with relay optimization is save in %s", relay_onnx_path)

                model_path = relay_onnx_path
                new_input_name, new_input_shape, new_output_name, _ = get_io_info_from_onnx(
                    model_path
                )
                self.config.import_config.input_name.value = new_input_name
                self.config.import_config.input_shape.value = new_input_shape
                self.config.import_config.output_name.value = new_output_name

            print_model_info(
                model_path,
                self.config.import_config.input_name.value,
                self.config.import_config.input_shape.value,
                self.config.import_config.output_name.value,
                "Before optimization with ppq",
            )

            device = self.config.quantize.quant_device.value
            from .tools.ppq_quantization import quantize_ppq
            from ppq.api import ENABLE_CUDA_KERNEL

            output_dir = "."
            if save_to_dir:
                output_dir = save_to_dir
                if not os.path.exists(save_to_dir):
                    os.makedirs(save_to_dir)
            if device == "cuda":
                with ENABLE_CUDA_KERNEL():
                    new_model_path, _ = quantize_ppq(
                        model_path,
                        self.config.import_config.input_shape.value,
                        self.config.quantize,
                        ppq_calibrate_data,
                        batch_size=self.config.quantize.cali_batch.value,
                        device=device,
                        output_dir=output_dir,
                        target=self.config.optimize.board.value,
                    )
            else:
                new_model_path, _ = quantize_ppq(
                    model_path,
                    self.config.import_config.input_shape.value,
                    self.config.quantize,
                    ppq_calibrate_data,
                    batch_size=self.config.quantize.cali_batch.value,
                    device=device,
                    output_dir=output_dir,
                    target=self.config.optimize.board.value,
                )
            self.config.main.model_file.value = [new_model_path]
            new_input_name, new_input_shape, new_output_name, _ = get_io_info_from_onnx(
                new_model_path
            )
            self.config.import_config.input_name.value = new_input_name
            self.config.import_config.input_shape.value = new_input_shape
            self.config.import_config.output_name.value = new_output_name

            print_model_info(
                new_model_path,
                self.config.import_config.input_name.value,
                self.config.import_config.input_shape.value,
                self.config.import_config.output_name.value,
                "After optimization with ppq",
            )

            self.import_model(
                self.config.main.model_file.value,
                input_name=self.config.import_config.input_name.value,
                input_shape=self.config.import_config.input_shape.value,
                output_name=self.config.import_config.output_name.value,
                save_to_dir=save_to_dir,
            )

        quant_scheme, is_per_channel, qnn_dtypes = get_quant_scheme_from_qnn(
            self.relay_ir.get_model()[0]
        )
        if self.config.quantize.quantization_scheme.value == "unset" and qnn_dtypes:
            # there is quantize/dequantize op in module, so it is a quantized model.
            if quant_scheme and is_per_channel:
                coverted_quant_scheme = convert_per_channel_scheme(quant_scheme)
                if coverted_quant_scheme is None:
                    raise HHBException(f"Unsupport per-channel quantization for {quant_scheme}\n")
                else:
                    self.config.quantize.quantization_scheme.value = coverted_quant_scheme
                    logger.log(
                        LOG,
                        "Detect that current model has been quantized with per-channel {}, "
                        "then quantization_scheme is set {}".format(
                            quant_scheme, self.config.quantize.quantization_scheme.value
                        ),
                    )
            elif quant_scheme:
                self.config.quantize.quantization_scheme.value = quant_scheme
                logger.log(
                    LOG,
                    "Detect that current model has been quantized with {}, ".format(quant_scheme),
                )
            else:
                raise HHBException(
                    "Can not infer the quantization scheme from original model, please "
                    "specify it by --quantization-scheme.\n"
                )

        # update cmd config
        self.config.generate_cmd_config()
        self.qnn_ir = hhb_quantize(self.relay_ir, self.config, calibrate_data, save_to_dir)

    def codegen(self, hhb_ir=None):
        """Codegen hhb model.

        Parameters
        ----------
        hhb_ir : HHBIRBase
            HHB ir wrapper that holds module and params
        """
        # update cmd config first
        if not self._is_init:
            raise HHBException("Please initialize session by _init_session() first.")
        self.config.generate_cmd_config()

        if hhb_ir is None:
            if self.qnn_ir is None:
                hhb_ir = self.relay_ir
            else:
                hhb_ir = self.qnn_ir
        if hhb_ir is None:
            raise HHBException("There is no any hhb ir exists, please import model first.")
        self.codegen_ir = hhb_codegen(hhb_ir, self.config)

    def create_executor(self):
        """Wrapper for hhb runner."""
        if self.codegen_ir is None:
            raise HHBException("Please codegen model first.")
        if not self._is_init:
            raise HHBException("Please initialize session by _init_session() first.")
        self.config.generate_cmd_config()
        self.executor = hhb_runner(self.codegen_ir, self.config)

    def inference(self, data):
        """Inference for hhb model on x86 platform.

        Parameters
        ----------
        data : Dict[str, numpy.ndarray]
            The input data

        Returns
        -------
        output : List[numpy.ndarray]
            The output data.
        """
        if self.executor is None:
            raise HHBException("Please create executor first.")
        out = hhb_inference(self.executor, data)
        return out

    def deploy(self):
        """Cross-compile codegen output for specified target."""
        if self.codegen_ir is None:
            raise HHBException("Please codegen model first.")
        if not self._is_init:
            raise HHBException("Please initialize session by _init_session() first.")
        self.config.generate_cmd_config()

        intrinsic = False
        if self.config._cmd_config.ahead_of_time == "intrinsic":
            intrinsic = True
        platform_deploy = HHBBoardBuildRuntime(
            self.config._cmd_config.board,
            self.config._cmd_config.output,
            intrinsic,
            self.config._cmd_config.link_lib,
        )

        # build all c source files to .o
        platform_deploy.build_c()
        # link_elf for linux platform
        platform_deploy.link_elf()

    def reorder_pixel_format(self):
        """If original model's input data pixel format is rgb, then covert it to bgr,
        otherwise, then convert it to rgb."""
        if self.relay_ir is None:
            raise HHBException("Please import model by import_model() first.")
        new_mod, new_params = reorder_pixel_format(*self.relay_ir.get_model())
        self.relay_ir.set_model(new_mod, new_params)

        # update config
        self.config.import_config.reorder_pixel_format.value = True
        if self.config.preprocess.pixel_format.value == "RGB":
            self.config.preprocess.pixel_format.value = "BGR"
        else:
            self.config.preprocess.pixel_format.value = "RGB"
        if self.config.preprocess.data_mean.value:
            self.config.preprocess.data_mean.value = self.config.preprocess.data_mean.value[::-1]

    def insert_preprocess_node(self):
        """Insert preprocess nodes into the head of model."""
        if self.relay_ir is None:
            raise HHBException("Please import model by import_model() first.")
        if not self._is_init:
            raise HHBException("Please initialize session by _init_session() first.")
        self.config.generate_cmd_config()
        mod, params = self.relay_ir.get_model()
        mod, params = insert_preprocess_node(
            mod,
            params,
            self.config._cmd_config.preprocess_config.data_mean,
            self.config._cmd_config.preprocess_config.data_scale,
        )
        self.relay_ir.set_model(mod, params)

        self.config.preprocess.add_preprocess_node.value = True


class Profiler(object):
    """Collections of profiler tools for HHB."""

    def __init__(self, compile_obj: Compiler = None) -> None:
        self.compile_obj = compile_obj

    def get_cal_total(self, data):
        """Statistics of all calculations of macc and flops"""
        from .core.profiler_manage import get_cal_total_info

        total = get_cal_total_info(data)
        macc = total["fused_mul_add"]
        flops = 0
        for k, v in total.items():
            if k != "fused_mul_add":
                flops += v
        return macc, flops

    def get_mem_total_byte(self, data):
        """Statistics of all memory requirement of params and output of all ops."""
        from .core.profiler_manage import get_mem_total_info

        total = get_mem_total_info(data)
        params = total["params"] * 4
        output = total["output"] * 4
        return params, output

    def analyse_model(self, model_type="relay", indicator="all", tofile=None):
        """Analyse model with specified indicator.

        Parameters
        ----------
        model_type : str
            Model type, selected from ["relay", ]
        indicator : str or list[str]
            Specified indicator data that will be extracted from model, selected from
            ["cal", "mem", "all"]
        tofile : str
            Save result data into file, support for .json format.

        Returns
        -------
        result : list[dict[str, dict[str, object]]]
            Result data
        """
        from tvm.relay import transform as _transform
        from tvm.ir import transform

        logger = logging.getLogger("HHB")
        if model_type == "relay":
            if self.compile_obj.relay_ir is None:
                raise HHBException("Please compile model by Compiler first.")
            mod, params = self.compile_obj.relay_ir.get_model()

            supported_ind = ["cal", "mem", "all"]
            if not indicator:
                indicator = ["all"]
            if isinstance(indicator, str):
                indicator = [indicator]
            if set(indicator) - set(supported_ind):
                raise HHBException(
                    "Unsupport for {}".format(list(set(indicator) - set(supported_ind)))
                )
            options = aitrace_options(indicator, "")
            logger.debug('profile model with: "%s"', str(options))

            if params:
                from tvm.relay.quantize.quantize_hhb import _bind_params

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

            if tofile and tofile.endswith(".json"):
                with open(tofile, "w") as f:
                    json.dump(result, f, indent=2)
            elif tofile:
                raise HHBException("Unsupport for output file format: {}".format(tofile))
            return result
        else:
            raise HHBException("Cannot analyse {} model".format(model_type))

    def merge_trace(self, trace_files: List[str], output_dir="."):
        for path in trace_files:
            if not os.path.exists(path) or not path.endswith(".json"):
                raise HHBException(f"File is not exists or is not .json file: {path}")

        assert len(trace_files) == 2, "Only support for 2 json files."
        merged_trace = merge_trace(trace_files)
        target_path = os.path.join(output_dir, "model_merge.trace.json")
        to_json(merged_trace.to_dict(), target_path)
        return merged_trace

    def profile_accuracy(self, trace_files: List[str], topk: int = 10, display=True, to_csv=None):
        assert (
            len(trace_files) == 2
        ), f"Analysis accuracy need two .json files, but get {len(trace_files)}"
        trace1 = from_json(trace_files[0])
        trace2 = from_json(trace_files[1])

        profile_acc_loss(trace1, trace2, topk=topk, to_csv=to_csv)

    def profile_trace(
        self,
        trace_files: List[str],
        profile_method: List[str] = None,
        topk: int = 10,
        output_type: List[str] = ["all"],
        output_dir=".",
    ):
        logger = logging.getLogger("HHB")
        for path in trace_files:
            if not os.path.exists(path) or not path.endswith(".json"):
                logger.warning("File is not exists or is not .json file, skipped: %s" % path)
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
            output_dir = os.path.join(output_dir, os.path.splitext(os.path.basename(path))[0])
            output_dir = ensure_dir(output_dir)
            if "trace_type" in json_data["otherData"]:
                # for relay/qnn
                trace_data = HHBIRTrace().from_dict(json_data)

                display = False
                if "print" in output_type or "all" in output_type:
                    display = True
                tmp_dir = None
                if "all" in output_type or "csv" in output_type:
                    tmp_dir = output_dir
                trace_data.profile(
                    profile_method=profile_method,
                    topk=topk,
                    display=display,
                    output_dir=tmp_dir,
                )
            else:
                # for chrome trace
                trace_data = HHBTrace().from_dict(json_data)
                profile_trace_data(trace_data, profile_method, output_dir, output_type, topk)


def _main(argv):
    """HHB commmand line interface."""
    arg_manage = ArgumentManage(argv)
    arg_manage.check_cmd_arguments()

    from .core.common import HHBArgumentParser

    parser = HHBArgumentParser(
        prog="HHB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="HHB command line tools",
        epilog=__doc__,
        allow_abbrev=False,
        add_help=False,
    )

    # add command line parameters
    curr_command_type = arg_manage.get_command_type()
    if curr_command_type == CommandType.SUBCOMMAND:
        arg_manage.set_subcommand(parser)
    else:
        arg_manage.set_main_command(parser)
        ALL_ARGUMENTS_DESC["main_command"] = collect_arguments_info(parser._actions)

    # print help info
    if arg_manage.have_help:
        arg_manage.print_help_info(parser)
        return 0

    # generate readme file
    if arg_manage.have_generate_readme:
        arg_manage.generate_readme(parser)
        return 0

    # parse command line parameters
    args = parser.parse_args(arg_manage.origin_argv[1:])
    # save hhb command
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    with open(os.path.join(args.output, "hhb_origin_cmd.txt"), "w") as f:
        f.write(" ".join(argv))
    if args.config_file:
        update_arguments_by_file(args, arg_manage.origin_argv[1:])
    args_filter = ArgumentFilter(args)

    # config logger
    logging.basicConfig(
        format="[%(asctime)s] (%(name)s %(levelname)s): %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger("HHB")
    logger.setLevel(25 - args.verbose * 10)

    # run command
    arg_manage.run_command(args_filter, curr_command_type)


def main():
    try:
        argv = sys.argv
        sys.exit(_main(argv))
    except KeyboardInterrupt:
        print("\nCtrl-C detected.")


if __name__ == "__main__":
    main()
