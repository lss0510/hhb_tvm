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
import os
import json

from tvm.relay.quantize.quantize_hhb import detect_quantized_model
from tvm import relay

from .core.arguments_manage import ArgumentManage, CommandType, HHBException, ArgumentFilter
from .core.arguments_manage import update_arguments_by_file, get_default_config, AttributeDict
from .core.arguments_manage import Config, ArgumentsBase
from .core.common import ensure_dir, import_module_for_register, collect_arguments_info
from .core.common import ArgInfo, ALL_ARGUMENTS_DESC, ARGS_DEST_TO_OPTIONS_STRING
from .core.main_command_manage import driver_main_command
from .core.hhbir_manage import HHBRelayIR, HHBQNNIR, reorder_pixel_format, HHBBoardBuildRuntime
from .core.preprocess_manage import hhb_preprocess
from .core.frontend_manage import insert_preprocess_node
from .core.profiler_manage import aitrace_options, convert_tvm_trace2python
from .importer import hhb_import
from .quantizer import hhb_quantize
from .codegen import hhb_codegen
from .simulate import hhb_runner, hhb_inference


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


def hhb_compile(hhb_config_file, **kwargs):
    """It includes all procedures that compile model with hhb: import->quantize->codegen->run.

    Parameters
    ----------
    hhb_config_file : str
        The path of config file.

    kwargs : dict
        Other values which will overwirte values in hhb_config_file.
    """
    base_config, _ = get_default_config()

    with open(hhb_config_file, "r") as f:
        usr_config = json.load(f)

    for sec, value in usr_config.items():
        base_config[sec].update(value)

    # unroll arguments
    unroll_config = AttributeDict()
    for sec, value in base_config.items():
        unroll_config.update(value)

    if kwargs:
        unroll_config.update(kwargs)

    args_filter = ArgumentFilter(unroll_config)
    driver_main_command(args_filter)


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

        detected_quant_type = detect_quantized_model(self.relay_ir.get_model()[0])
        if detected_quant_type:
            if len(detected_quant_type) == 1:
                detected_quant_type = detected_quant_type.pop()
                if detected_quant_type == "uint8":
                    self.config.quantize.quantization_scheme.value = "uint8_asym"
                elif detected_quant_type == "int8":
                    self.config.quantize.quantization_scheme.value = "int8_asym"
                else:
                    raise HHBException(
                        "Unsupport quantization type:{}.\n".format(detected_quant_type)
                    )
                logger = logging.getLogger("HHB")
                logger.log(
                    LOG,
                    "Detect that current model has been quantized with {}, "
                    "--quantization-scheme will be overwritten to {}".format(
                        detected_quant_type, self.config.quantize.quantization_scheme.value
                    ),
                )
            else:
                logger.warning("Detect that there are multi quantization types in model.")

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

    def __init__(self, compile_obj: Compiler) -> None:
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
            Save result data into file, support for .aitrace and .json format.
            .aitrace: binary format for result data that defiled by HHB;
            .json: JSON format.

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
            if tofile and tofile.endswith(".aitrace"):
                options = aitrace_options(indicator, tofile)
            else:
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
