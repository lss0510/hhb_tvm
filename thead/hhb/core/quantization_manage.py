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
"""Manage quantization"""
import logging
import os
import json

import tvm
from tvm import relay
from tvm.relay.expr import Call, Constant

from tvm.relay.quantize.hhb_quantize import get_quantized_model, quantize_hhb
from tvm.relay.quantize.quantization.hybird_quantize import HybridQuantizationInfo

from .common import argument_filter_helper
from .common import ALL_ARGUMENTS_INFO
from .common import AttributeDict
from .common import HHBException
from .common import hhb_exit, hhb_deprecated_check


# pylint: disable=invalid-name
LOG = 25
logger = logging.getLogger("HHB")


def convert_per_channel_scheme(quant_scheme):
    """Get supported per-channel quant scheme."""
    convert = {
        "int4_asym": "int4_asym_w_sym",
        "int8_asym": "int8_asym_w_sym",
    }
    return convert[quant_scheme] if quant_scheme in convert else None


def update_hybrid_layer(config_dict, output_dir):
    quant_file = os.path.join(output_dir, "model.quant.json")
    if not os.path.exists(quant_file):
        return
    with open(quant_file, "r") as f:
        data = json.load(f)
    if "hybrid_layers" not in data or not data["hybrid_layers"]:
        return
    hqi = HybridQuantizationInfo()
    hqi.from_dict(data["hybrid_layers"])

    hybrid_layer = hqi.get_hybrid_layers()
    if config_dict["hybrid_layer_name"]:
        config_dict["hybrid_layer_name"] += hybrid_layer
    else:
        config_dict["hybrid_layer_name"] = hybrid_layer
    if config_dict["hybrid_quantization_scheme"] == "unset":
        config_dict["hybrid_quantization_scheme"] = "int16_sym"

    if config_dict["hybrid_layer_name"]:
        logger.log(
            LOG,
            "The following layers will be quantized with %s:",
            config_dict["hybrid_quantization_scheme"],
        )
        logger.log(LOG, "\t{}".format(config_dict["hybrid_layer_name"]))
    else:
        logger.log(LOG, "There is no layer to do hybrid quantization.")


def ignore_layers_from_auto_quant(module, board):
    """disable some ops hybrid quant: op before concat."""

    class GetLimitedLayers(tvm.relay.ExprVisitor):
        """Get the limited layers"""

        def __init__(self, board):
            super(GetLimitedLayers, self).__init__()
            self.op_lists = []
            self.board = board

        def visit_call(self, call):
            _ = [self.visit(arg) for arg in call.args]
            if call.op.name == "qnn.csi.concatenate":
                # concat should be ignored
                curr_name = str(call.attrs.layer_name)
                self.op_lists.append(curr_name)

                for pre_call in call.args[0]:
                    if isinstance(pre_call, Call):
                        # the op before concat should be ignored
                        curr_name = str(pre_call.attrs.layer_name)
                        self.op_lists.append(curr_name)

                        # the op before before concat
                        pp_call = pre_call.args[0]
                        if (
                            pp_call
                            and isinstance(pp_call, Call)
                            and pp_call.op.name == "qnn.csi.relu"
                        ):
                            curr_name = str(pp_call.attrs.layer_name)
                            self.op_lists.append(curr_name)

                            ppp_call = pp_call.args[0]
                            if (
                                ppp_call
                                and isinstance(ppp_call, Call)
                                and ppp_call.op.name == "qnn.csi.conv2d"
                            ):
                                curr_name = str(ppp_call.attrs.layer_name)
                                self.op_lists.append(curr_name)
            else:
                for pre_call in call.args:
                    if isinstance(pre_call, Call):
                        if pre_call.op.name == "qnn.csi.concatenate":
                            # the op after concat should be ignore
                            curr_name = str(call.attrs.layer_name)
                            self.op_lists.append(curr_name)

    limited_layers = GetLimitedLayers(board)
    limited_layers.visit(module["main"])
    return limited_layers.op_lists


def deprecated_check(args):
    if args.quantize_config.channel_quantization:
        hhb_deprecated_check("--channel-quantization")


def get_config_dict(args):
    deprecated_check(args)
    config_dict = {
        "nbit_input": args.quantize_config.num_bit_input,
        "nbit_weight": args.quantize_config.num_bit_weight,
        "nbit_activation": args.quantize_config.num_bit_activation,
        "dtype_input": args.quantize_config.dtype_input,
        "dtype_weight": args.quantize_config.dtype_weight,
        "dtype_activation": args.quantize_config.dtype_activation,
        "calibrate_mode": args.quantize_config.calibrate_mode,
        "activate_quantized_type": args.quantize_config.activate_quantized_type,
        "weight_quantized_type": args.quantize_config.weight_quantized_type,
        "weight_scale": args.quantize_config.weight_scale,
        "fuse_relu": args.quantize_config.fuse_relu,
        "fuse_clip": args.quantize_config.fuse_clip,
        "fuse_conv_relu": args.quantize_config.fuse_conv_relu,
        "fuse_reshape_dense": args.quantize_config.fuse_reshape_dense,
        "broadcast_quantization": args.quantize_config.broadcast_quantization,
        "channel_quantization_ratio_threshold": args.quantize_config.channel_quantization_ratio_threshold,
        "fuse_mul_before_conv": args.quantize_config.fuse_mul_before_conv,
        "fuse_mul_after_conv": args.quantize_config.fuse_mul_after_conv,
        "fuse_add_before_conv": args.quantize_config.fuse_add_before_conv,
        "fuse_add_after_conv": args.quantize_config.fuse_add_after_conv,
        "layout": args.quantize_config.target_layout,
        "quantization_scheme": args.quantize_config.quantization_scheme,
        "fuse_zp2bias": args.quantize_config.fuse_zp2bias,
        "use_custom_fusion": args.quantize_config.use_custom_fusion,
        "convert_to_relay": args.quantize_config.convert_to_relay,
        "hybrid_quantization_scheme": args.quantize_config.hybrid_quantization_scheme,
        "hybrid_layer_name": args.quantize_config.hybrid_layer_name,
        "h_sram_size": args.hardware_sram_size,
        "h_max_groups": args.hardware_max_groups,
        "h_max_out_channel": args.hardware_max_out_channel,
        "h_max_kernel_size": args.hardware_max_kernel_size,
        "h_contain_weight": args.hardware_contain_weight,
        "h_align": args.hardware_alignment,
        "auto_hybrid_quantization": args.quantize_config.auto_hybrid_quantization,
        "quantization_loss_algorithm": args.quantize_config.quantization_loss_algorithm,
        "quantization_loss_threshold": args.quantize_config.quantization_loss_threshold,
        "dump_quantization_loss": args.quantize_config.dump_quantization_loss,
        "params_path": args.output,
        "model_save": args.model_save,
        "trace_strategy": args.codegen_config.trace_strategy,
        "input_memory_type": args.codegen_config.input_memory_type,
        "output_memory_type": args.codegen_config.output_memory_type,
        "model_priority": args.codegen_config.model_priority,
        "matrix_extension_mlen": args.matrix_extension_mlen,
        "target": args.board,
        "loss_threshold_type": args.quantize_config.loss_threshold_type,
        "from_quant_file": args.quantize_config.from_quant_file,
        "conv2d_algorithm": args.codegen_config.conv2d_algorithm,
        "ahead_of_time": args.codegen_config.ahead_of_time,
        "dynamic_shape": args.codegen_config.dynamic_shape,
        "low_bound_scale": args.quantize_config.low_bound_scale,
        "high_bound_scale": args.quantize_config.high_bound_scale,
        "show_session_run_time": args.show_session_run_time,
        "trace": args.trace if args.trace else [],
    }
    if args.quantize_config.low_bound_scale < 1:
        hhb_exit("--low-bound-scale should be >= 1.")
    if args.quantize_config.high_bound_scale <= 0 or args.quantize_config.high_bound_scale > 1:
        hhb_exit("--high-bound-scale should be (0, 1].")
    th1520_input_fix_size = args.codegen_config.th1520_input_fix_size
    if len(th1520_input_fix_size) == 2:
        config_dict["th1520_input_fix_height"] = th1520_input_fix_size[0]
        config_dict["th1520_input_fix_width"] = th1520_input_fix_size[1]

    config_dict["debug_level"] = "WARNING"
    if args.verbose >= 3:
        config_dict["debug_level"] = "INFO"

    if args.codegen_config.ahead_of_time == "intrinsic":
        if args.quantize_config.quantization_scheme not in ["float32", "float16"]:
            raise HHBException("--ahead-of-time intrinsic only support float32 or float16.\n")

    if args.quantize_config.quantization_scheme in [
        "int4_asym_w_sym",
        "int8_asym_w_sym",
        "float16_w_int8",
    ]:
        config_dict["channel_quantization"] = True
    else:
        config_dict["channel_quantization"] = False
    return config_dict


@argument_filter_helper
def collect_quantization_config(filtered_args, extra=None):
    """add quantize_config item for hold quantization info"""
    unexpected_params = ["calibrate_dataset"]
    all_true_quantize_params = [
        k for k in ALL_ARGUMENTS_INFO["quantize"] if k not in unexpected_params
    ]
    filtered_args.quantize_config = AttributeDict()
    for k in all_true_quantize_params:
        filtered_args.quantize_config[k] = filtered_args[k]


@argument_filter_helper
def set_quantize_params_by_board(filtered_args, extra=None):
    if not hasattr(filtered_args, "board"):
        raise HHBException("There is no board args in filtered_args\n")
    if not hasattr(filtered_args, "quantize_config"):
        raise HHBException("Please execute 'collect_quantization_config' filter first.\n")

    if filtered_args.board in ("th1520", "hth1520"):
        new_values = {
            "num_bit_input": 8,
            "num_bit_weight": 8,
            "num_bit_activation": 32,
            "dtype_input": "int8",
            "dtype_weight": "int8",
            "dtype_activation": "int32",
            "calibrate_mode": "maxmin",
            "weight_quantized_type": "asym",
            "activate_quantized_type": "asym",
            "weight_scale": "maxmin",
            "fuse_relu": False,
            # "fuse_reshape": False,
            "fuse_mul_add_to_conv": True,
            # "channel_quantization": False,
            "broadcast_quantization": True,
        }
        if filtered_args.quantize_config.channel_quantization:
            if filtered_args.quantize_config.quantization_scheme != "int8_asym_w_sym":
                hhb_exit(
                    "th1520 channel quantization only support with int8_asym_w_sym quantization scheme."
                )
        if filtered_args.quantize_config.channel_quantization:
            new_values["calibrate_mode"] = "maxmin"
            new_values["weight_scale"] = "maxmin"
        if filtered_args.quantize_config.quantization_scheme == "unset":
            new_values["quantization_scheme"] = "int8_asym"
        elif filtered_args.quantize_config.quantization_scheme == "int8_sym":
            new_values["quantization_scheme"] = "int8_sym"
            filtered_args.quantize_config.quantization_scheme = "unset"
        elif filtered_args.quantize_config.quantization_scheme == "int8_asym_w_sym":
            new_values["quantization_scheme"] = "int8_asym_w_sym"
        elif filtered_args.quantize_config.quantization_scheme == "int8_original":
            raise HHBException(
                f"quantization_scheme 'int8_original' has been removed. Please use the actual quantification of the model."
            )
        elif filtered_args.quantize_config.quantization_scheme == "uint8_asym":
            new_values["quantization_scheme"] = "uint8_asym"
        elif filtered_args.quantize_config.quantization_scheme == "int8_asym":
            new_values["quantization_scheme"] = "int8_asym"
            new_values["calibrate_mode"] = "pow2"
            new_values["weight_scale"] = "pow2"
        elif filtered_args.quantize_config.quantization_scheme == "int16_sym":
            new_values["quantization_scheme"] = "int16_sym"
            new_values["num_bit_input"] = 16
            new_values["num_bit_weight"] = 16
            filtered_args.quantize_config.quantization_scheme = "unset"
        else:
            raise HHBException(
                f"Unsupport quantization scheme '{filtered_args.quantize_config.quantization_scheme}' on th1520\n"
            )
        if filtered_args.quantize_config.quantization_scheme in ("float16",):
            raise HHBException("th1520 unsupport float16\n")
    elif filtered_args.board == "e907":
        new_values = {
            "num_bit_input": 8,
            "num_bit_weight": 8,
            # "num_bit_activation": 32,
            "dtype_input": "int8",
            "dtype_weight": "int8",
            "dtype_activation": "int32",
            "calibrate_mode": "maxmin",
            "weight_quantized_type": "asym",
            "activate_quantized_type": "asym",
            "weight_scale": "maxmin",
            "fuse_relu": False,
            # "fuse_reshape": False,
            "fuse_mul_add_to_conv": True,
            # "channel_quantization": False,
            # "broadcast_quantization": False,
        }
    elif filtered_args.board == "c906":
        new_values = {
            "num_bit_input": 16,
            "num_bit_weight": 16,
            # "num_bit_activation": 32,
            "dtype_input": "float16",
            "dtype_weight": "float16",
            "dtype_activation": "float16",
            # "calibrate_mode": "maxmin",
            "weight_quantized_type": "sym",
            "activate_quantized_type": "sym",
            "weight_scale": "maxmin",
            "fuse_relu": False,
            # "fuse_reshape": False,
            "fuse_mul_add_to_conv": True,
            # "channel_quantization": False,
            # "broadcast_quantization": False,
        }
    elif filtered_args.board == "rvm":
        new_values = {
            "num_bit_input": 16,
            "num_bit_weight": 16,
            # "num_bit_activation": 32,
            "dtype_input": "float16",
            "dtype_weight": "float16",
            "dtype_activation": "float16",
            # "calibrate_mode": "maxmin",
            "weight_quantized_type": "sym",
            "activate_quantized_type": "sym",
            "weight_scale": "maxmin",
            "fuse_relu": False,
            # "fuse_reshape": False,
            "fuse_mul_add_to_conv": True,
            # "channel_quantization": False,
            # "broadcast_quantization": False,
        }
    elif filtered_args.board == "c908":
        new_values = {
            "num_bit_input": 8,
            "num_bit_weight": 8,
            "num_bit_activation": 32,
            "dtype_input": "int8",
            "dtype_weight": "int8",
            "dtype_activation": "int32",
            # "calibrate_mode": "maxmin",
            "weight_quantized_type": "asym",
            "activate_quantized_type": "asym",
            "weight_scale": "maxmin",
            "fuse_relu": False,
            # "fuse_reshape": False,
            "fuse_mul_add_to_conv": True,
            # "channel_quantization": False,
            # "broadcast_quantization": False,
        }
    elif filtered_args.board == "c920":
        new_values = {
            "num_bit_input": 16,
            "num_bit_weight": 16,
            # "num_bit_activation": 32,
            "dtype_input": "float16",
            "dtype_weight": "float16",
            "dtype_activation": "float16",
            # "calibrate_mode": "maxmin",
            "weight_quantized_type": "sym",
            "activate_quantized_type": "sym",
            "weight_scale": "maxmin",
            "fuse_relu": False,
            # "fuse_reshape": False,
            "fuse_mul_add_to_conv": True,
            # "channel_quantization": False,
            # "broadcast_quantization": False,
        }
    elif filtered_args.board == "c920v2":
        new_values = {
            "num_bit_input": 8,
            "num_bit_weight": 8,
            "num_bit_activation": 32,
            "dtype_input": "int8",
            "dtype_weight": "int8",
            "dtype_activation": "int32",
            # "calibrate_mode": "maxmin",
            "weight_quantized_type": "asym",
            "activate_quantized_type": "asym",
            "weight_scale": "maxmin",
            "fuse_relu": False,
            # "fuse_reshape": False,
            "fuse_mul_add_to_conv": True,
            # "channel_quantization": False,
            # "broadcast_quantization": False,
        }
    elif filtered_args.board == "x86_ref":
        new_values = {
            "num_bit_input": 8,
            "num_bit_weight": 8,
            "num_bit_activation": 32,
            "dtype_input": "int8",
            "dtype_weight": "int8",
            "dtype_activation": "int32",
            # "calibrate_mode": "maxmin",
            "weight_quantized_type": "asym",
            "activate_quantized_type": "asym",
            "weight_scale": "maxmin",
            "fuse_relu": False,
            # "fuse_reshape": False,
            "fuse_mul_add_to_conv": True,
            # "channel_quantization": False,
            # "broadcast_quantization": False,
        }
    else:
        raise HHBException("Unset --board\n")

    if filtered_args.quantize_config.quantization_scheme in ["int4_asym_w_sym"]:
        new_values["num_bit_input"] = 4
        new_values["num_bit_weight"] = 4
        new_values["num_bit_activation"] = 32
        new_values["dtype_input"] = "int4"
        new_values["dtype_weight"] = "int4"
        new_values["dtype_activation"] = "int32"
        new_values["activate_quantized_type"] = "asym"
        new_values["weight_quantized_type"] = "sym"
        if filtered_args.target_layout == "NCHW":
            raise HHBException("Unsupport target_layout=NCHW for int4.\n")
    elif filtered_args.quantize_config.quantization_scheme == "uint8_asym":
        new_values["num_bit_input"] = 8
        new_values["num_bit_weight"] = 8
        new_values["num_bit_activation"] = 32
        new_values["dtype_input"] = "uint8"
        new_values["dtype_weight"] = "uint8"
        new_values["dtype_activation"] = "int32"
        new_values["activate_quantized_type"] = "asym"
        new_values["weight_quantized_type"] = "asym"
    elif filtered_args.quantize_config.quantization_scheme in ["int8_sym"]:
        new_values["num_bit_input"] = 8
        new_values["num_bit_weight"] = 8
        new_values["num_bit_activation"] = 32
        new_values["dtype_input"] = "int8"
        new_values["dtype_weight"] = "int8"
        new_values["dtype_activation"] = "int32"
        new_values["activate_quantized_type"] = "sym"
        new_values["weight_quantized_type"] = "sym"
    elif filtered_args.quantize_config.quantization_scheme in ["int8_asym_w_sym"]:
        new_values["num_bit_input"] = 8
        new_values["num_bit_weight"] = 8
        new_values["num_bit_activation"] = 32
        new_values["dtype_input"] = "int8"
        new_values["dtype_weight"] = "int8"
        new_values["dtype_activation"] = "int32"
        new_values["activate_quantized_type"] = "asym"
        new_values["weight_quantized_type"] = "sym"
        # new_values["channel_quantization"] = True
    elif filtered_args.quantize_config.quantization_scheme == "int8_asym":
        new_values["num_bit_input"] = 8
        new_values["num_bit_weight"] = 8
        new_values["num_bit_activation"] = 32
        new_values["dtype_input"] = "int8"
        new_values["dtype_weight"] = "int8"
        new_values["dtype_activation"] = "int32"
        new_values["activate_quantized_type"] = "asym"
        new_values["weight_quantized_type"] = "asym"
    elif filtered_args.quantize_config.quantization_scheme == "int16_sym":
        new_values["num_bit_input"] = 16
        new_values["num_bit_weight"] = 16
        new_values["num_bit_activation"] = 32
        new_values["dtype_input"] = "int16"
        new_values["dtype_weight"] = "int16"
        new_values["dtype_activation"] = "int32"
        new_values["activate_quantized_type"] = "sym"
        new_values["weight_quantized_type"] = "sym"
    elif filtered_args.quantize_config.quantization_scheme == "float16":
        new_values["num_bit_input"] = 16
        new_values["num_bit_weight"] = 16
        new_values["num_bit_activation"] = 16
        new_values["dtype_input"] = "float16"
        new_values["dtype_weight"] = "float16"
        new_values["dtype_activation"] = "float16"
        new_values["activate_quantized_type"] = "sym"
        new_values["weight_quantized_type"] = "sym"
    elif filtered_args.quantize_config.quantization_scheme == "float16_w_int8":
        new_values["num_bit_input"] = 16
        new_values["num_bit_weight"] = 16
        new_values["num_bit_activation"] = 16
        new_values["dtype_input"] = "float16"
        # w_int8 only for matmul now
        new_values["dtype_weight"] = "float16"
        new_values["dtype_activation"] = "float16"
        new_values["activate_quantized_type"] = "sym"
        new_values["weight_quantized_type"] = "sym"
    elif filtered_args.quantize_config.quantization_scheme == "float32":
        new_values["num_bit_input"] = 32
        new_values["num_bit_weight"] = 32
        new_values["num_bit_activation"] = 32
        new_values["dtype_input"] = "float32"
        new_values["dtype_weight"] = "float32"
        new_values["dtype_activation"] = "float32"
        new_values["activate_quantized_type"] = "sym"
        new_values["weight_quantized_type"] = "sym"
    elif filtered_args.quantize_config.quantization_scheme == "int8_original":
        raise HHBException(
            f"quantization_scheme 'int8_original' has been removed. Please use the actual quantification of the model."
        )
    else:
        if "quantization_scheme" not in new_values:
            raise HHBException("Unsupport quantization scheme.\n")

    if filtered_args.quantize_config.num_bit_input != 0:
        new_values["num_bit_input"] = filtered_args.quantize_config.num_bit_input

    if filtered_args.quantize_config.num_bit_weight != 0:
        new_values["num_bit_weight"] = filtered_args.quantize_config.num_bit_weight

    if filtered_args.quantize_config.num_bit_activation != 0:
        new_values["num_bit_activation"] = filtered_args.quantize_config.num_bit_activation

    if filtered_args.quantize_config.dtype_input != "unset":
        new_values["dtype_input"] = filtered_args.quantize_config.dtype_input + str(
            filtered_args.quantize_config.num_bit_input
        )

    if filtered_args.quantize_config.dtype_weight != "unset":
        new_values["dtype_weight"] = filtered_args.quantize_config.dtype_weight + str(
            filtered_args.quantize_config.num_bit_weight
        )

    if filtered_args.quantize_config.dtype_activation != "unset":
        new_values["dtype_activation"] = filtered_args.quantize_config.dtype_activation + str(
            filtered_args.quantize_config.num_bit_activation
        )

    if filtered_args.quantize_config.weight_quantized_type != "unset":
        new_values["weight_quantized_type"] = filtered_args.quantize_config.weight_quantized_type

    if filtered_args.quantize_config.activate_quantized_type != "unset":
        new_values[
            "activate_quantized_type"
        ] = filtered_args.quantize_config.activate_quantized_type

    filtered_args.quantize_config.update(new_values)

    if (
        filtered_args.quantize_config.broadcast_quantization
        and filtered_args.quantize_config.fuse_relu
    ):
        raise HHBException(
            "--broadcast-quantization and --fuse_relu can only be used simultaneously.\n"
        )


def get_quant_scheme_from_qnn(mod):
    """Obtain quant_scheme from quantized model."""

    class InterHelper(relay.ExprVisitor):
        """Internal helper class"""

        def __init__(self):
            super(InterHelper, self).__init__()
            self.memo_map = {}
            self.quant_scheme = set()
            self.per_channel = False

        def visit_call(self, call):
            _ = [self.visit(arg) for arg in call.args]
            if call.op.name == "qnn.quantize":
                self.quant_scheme.add(call.attrs.out_dtype)

                scale_node = call.args[1]
                zp_node = call.args[2]
                if scale_node.data.numpy().size > 1 or zp_node.data.numpy().size > 1:
                    self.per_channel = True
            elif call.op.name == "qnn.dequantize":
                if isinstance(call.args[0], Constant):
                    self.quant_scheme.add(call.args[0].data.numpy().dtype.name)

    ih = InterHelper()
    ih.visit(mod["main"])

    # delete bias's dtype
    dtypes = ih.quant_scheme - set(("int32",))
    dtype2scheme = {
        "int4": "int4_asym",
        "int8": "int8_asym",
        "uint8": "uint8_asym",
        "float16": "float16",
        "float32": "float32",
    }

    final_scheme = None
    if len(dtypes) == 1:
        d = dtypes.pop()
        if d in dtype2scheme:
            final_scheme = dtype2scheme[d]
    return final_scheme, ih.per_channel, ih.quant_scheme


def quantize_model(mod, params, qconfig, dataset=None, target="x86_ref"):
    """Quantize the imported relay module.

    Parameters
    ----------
    mod : tvm.IRModule
        The relay module for compilation
    params : dict of str to tvm.NDArray
        The parameter dict to be used by relay
    qconfig : tvm.relay.quantize.QConfig
        The config parameter for quantization
    dataset : data generator
        The dict of input_name(str) to numpy.ndarray

    Returns
    -------
    qfunc : Function
        The graph after quantization
    """
    with tvm.transform.PassContext(opt_level=3, config={"relay.ext.csinn.options": qconfig}):
        logger.debug("current quantize config:")
        logger.debug(qconfig)
        _, _, qnn_dtypes = get_quant_scheme_from_qnn(mod)
        if qnn_dtypes:
            qfunc = get_quantized_model(mod, params, qconfig, target)
        else:
            qfunc = quantize_hhb(mod, params, qconfig, dataset=dataset, target=target)
    return qfunc
