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
Optimize the imported model.
"""
import logging
import os
from tvm.relay.quantize.quantize_hhb import detect_quantized_model

from .core.common import (
    hhb_register_parse,
    HHBException,
    AttributeDict,
    ensure_dir,
    generate_config_file,
    ALL_ARGUMENTS_DESC,
    collect_arguments_info,
)
from .core.frontend_manage import insert_preprocess_node
from .core.arguments_manage import (
    add_preprocess_argument,
    add_quantize_argument,
    add_hardware_argument,
    add_codegen_argument,
    add_common_argument,
    add_optimize_argument,
    ArgumentFilter,
)
from .core.hhbir_manage import (
    HHBRelayIR,
    HHBQNNIR,
    get_input_info_from_relay,
    get_output_info_from_relay,
)
from .core.quantization_manage import (
    collect_quantization_config,
    set_quantize_params_by_board,
    get_config_dict,
    quantize_model,
    update_hybrid_layer,
    ignore_layers_from_auto_quant,
)
from .core.codegen_manage import (
    collect_codegen_config,
    set_codegen_config,
)
from .core.preprocess_manage import collect_preprocess_config, set_preprocess_params, DatasetLoader


# pylint: disable=invalid-name
LOG = 25
logger = logging.getLogger("HHB")


def save_quantized_model(mod, output_dir="."):
    """Save quantized model into file.

    Parameters
    ----------
    mod : tvm.IRModule
        The qnn module for compilation
    output_dir : str
        The output directory holding file.
    """
    mod_path = os.path.join(output_dir, "qnn.txt")

    with open(mod_path, "w") as f:
        f.write(mod.astext())


def hhb_quantize(relay_ir: HHBRelayIR, config, calibrate_data=None, save_to_dir=None) -> HHBQNNIR:
    """Quantize model and convert relay ir into qnn ir.

    Parameters
    ----------
    relay_ir : HHBRelayIR
        Relay ir wrapper that holds module and params
    config : Config
        All config for HHB
    calibrate_data : List[Dict[str, numpy.ndarray]]
        The calibration data for quantization. It includes batches of data.
    save_to_dir : str, optional
        save model into specified directory

    Returns
    -------
    relay_ir : HHBQNNIR
        Qnn ir wrapper that holds module and params
    """
    mod, params = relay_ir.get_model()
    hhb_config = config._cmd_config
    inter_hhb_config = get_config_dict(hhb_config)
    inter_hhb_config["target"] = hhb_config.board
    inter_hhb_config["params_path"] = os.path.join(inter_hhb_config["params_path"], "qnn.params")

    qnn_ir = HHBQNNIR()
    qnn_ir._curr_mod = quantize_model(
        mod, params, inter_hhb_config, calibrate_data, hhb_config.board
    )

    # update auto-qaunt layers
    if inter_hhb_config["auto_hybrid_quantization"]:
        update_hybrid_layer(hhb_config.quantize_config, hhb_config.output)

        limited_layer = ignore_layers_from_auto_quant(qnn_ir._curr_mod, hhb_config.board)
        logger.info(
            "These layers will be removed from hybrid quant list: {}".format(
                set(hhb_config.quantize_config["hybrid_layer_name"]) & set(limited_layer)
            )
        )
        hhb_config.quantize_config["hybrid_layer_name"] = list(
            set(hhb_config.quantize_config["hybrid_layer_name"]) - set(limited_layer)
        )

        if hhb_config.quantize_config.ignore_hybrid_layer:
            hhb_config.quantize_config["hybrid_layer_name"] = list(
                set(hhb_config.quantize_config["hybrid_layer_name"])
                - set(hhb_config.quantize_config.ignore_hybrid_layer)
            )

    if save_to_dir is not None:
        save_to_dir = ensure_dir(save_to_dir)
        save_quantized_model(qnn_ir._curr_mod, save_to_dir)

    return qnn_ir


@hhb_register_parse
def add_quantize_parser(subparsers):
    """Include parser for 'quantize' subcommand"""

    parser = subparsers.add_parser("quantize", help="Quantize the imported model")
    parser.set_defaults(func=driver_quantize)

    add_preprocess_argument(parser)
    add_quantize_argument(parser)
    add_hardware_argument(parser)
    add_optimize_argument(parser)
    add_codegen_argument(parser)
    add_common_argument(parser)

    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity")
    parser.add_argument("FILE", help="Directory to the model file")

    ALL_ARGUMENTS_DESC["quantize"] = collect_arguments_info(parser._actions)


def driver_quantize(args_filter: ArgumentFilter):
    """Driver quantize command"""
    args = args_filter.filtered_args
    if args.board == "th1520" and (args.hybrid_computing or args.auto_hybrid_quantization):
        args.board = "hth1520"
    if not os.path.exists(args.FILE) or not os.path.isdir(args.FILE):
        raise HHBException("The directory is not exists: {}".format(args.FILE))
    relay_ir = HHBRelayIR()
    relay_ir.load_model(args.FILE)
    input_mod, input_params = relay_ir.get_model()
    input_name_list, input_shape_list, _ = get_input_info_from_relay(input_mod, input_params)
    output_shape_list, _ = get_output_info_from_relay(input_mod, input_params)

    detected_quant_type = detect_quantized_model(input_mod)
    if detected_quant_type:
        if len(detected_quant_type) == 1:
            detected_quant_type = detected_quant_type.pop()
            if detected_quant_type == "uint8":
                args.quantization_scheme = "uint8_asym"
            elif detected_quant_type == "int8":
                args.quantization_scheme = "int8_asym"
            else:
                raise HHBException("Unsupport quantization type:{}.\n".format(detected_quant_type))
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
    extra_args.model_save = "save_and_run"  # default value
    args_filter.filter_argument(all_filters, extra=extra_args)
    args = args_filter.filtered_args

    # add preprocess node into mod
    if args.preprocess_config.add_preprocess_node:
        input_mod, input_params = insert_preprocess_node(
            input_mod,
            input_params,
            args.preprocess_config.data_mean,
            args.preprocess_config.data_scale,
        )
        logger.debug("Insert preprocess node into model successfully!")

    # get calibrate dataset
    dataset_list = []
    if args.calibrate_dataset:
        logger.info("get calibrate dataset from %s", args.calibrate_dataset)
        dl = DatasetLoader(
            args.calibrate_dataset, args.preprocess_config, input_shape_list, input_name_list
        )
        dataset = dl.get_data()
        for d in dataset:
            dataset_list.append(d)

    config_dict = get_config_dict(args)

    qnn_ir = HHBQNNIR()
    qnn_ir.convert((input_mod, input_params), config_dict, dataset_list, args.board)
    args.output = ensure_dir(args.output)

    if args.generate_config:
        generate_config_file(os.path.join(args.output, "cmd_quantizer_params.yml"))

    pre_params = args.preprocess_config
    qnn_ir.save_model(args.output, pre_params, config_dict)
