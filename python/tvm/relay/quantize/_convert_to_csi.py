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
# pylint: disable=invalid-name, unused-argument, too-many-lines, import-outside-toplevel
# pylint: disable=consider-using-enumerate, no-else-return, unused-variable
# pylint: disable=inconsistent-return-statements, logging-not-lazy, arguments-differ
# pylint: disable=too-many-nested-blocks, no-else-continue
"""Find scales for quantization on the dataset."""
from __future__ import absolute_import
import logging
import numpy as np
from tqdm import tqdm
import tvm
from tvm import relay
from tvm.ir import IRModule
from tvm.relay.transform import function_pass
from .asy_kl_divergence import _find_scale_by_asy_kl
from .kl_divergence import _find_scale_by_kl
from ..expr import Var, Call, TupleGetItem, Constant, Tuple, const
from .. import function, analysis
from ...ir import transform
from ..frontend.common import infer_shape as _infer_shape

LOG = 25
logger = logging.getLogger("HHB")

# type
CONST = 0
ACTIVATION = 1

# q mode
PER_TENSOR = 0
PER_CHANNEL = 1

# value type
USE_MINMAX = 0
USE_SCALE = 1


def current_csinn_config():
    return transform.PassContext.current().config["relay.ext.csinn.options"]


def _find_minmax(stats, axes=None):
    min_value = np.min(stats, axes)
    max_value = np.max(stats, axes)
    return min_value.astype("float"), max_value.astype("float")


def _find_pow2_minmax(stats, axes=None):
    min_value = np.min(stats, axes)
    max_value = np.max(stats, axes)
    valid_range = np.power(2, 8 - 1) - 1
    abs_max = np.max([np.abs(min_value), np.abs(max_value)])
    scale = valid_range / abs_max
    exponent = np.frexp(scale)[1]
    scale = 1.0 / np.power(2.0, exponent - 1)
    new_min = scale * -128
    new_max = scale * 127

    return new_min.astype("float"), new_max.astype("float")


def _find_abs_minmax(stats, axes=None):
    abs_stats = np.abs(stats)
    return _find_minmax(abs_stats, axes)


statistical_func_map = {
    "maxmin": {
        "asym": _find_minmax,
        "sym": _find_minmax,
    },
    "kl_divergence": {
        "asym": _find_scale_by_asy_kl,
        "sym": _find_scale_by_kl,
    },
    "pow2": {
        "asym": _find_minmax,
        "sym": _find_pow2_minmax,
    },
    "scale": {
        "asym": _find_abs_minmax,
        "sym": _find_abs_minmax,
    },
}


def get_weight_params(weight_val, curr_config):
    """
    Channel quantization only supports NCHW layout model.
    For constants, all dimensions except the first one will be seem as a whole.
    We're sure that this is right in the case of four dimensions,
    and in other cases there may be some problems.
    """

    def _shape_quant_map(data):
        return [data] if len(data.shape) == 0 else data

    calibrate_mode = curr_config["weight_scale"]
    channel_quantize = curr_config["channel_quantization"]
    quantize_type = curr_config["weight_quantized_type"]
    if calibrate_mode == "defult":
        calibrate_mode = curr_config["calibrate_mode"]
    # Check for legitimacy
    if calibrate_mode not in statistical_func_map:
        raise Exception(f"Weight not support this calibrate mode: {calibrate_mode}")

    statistical_func = statistical_func_map[calibrate_mode][quantize_type]
    if channel_quantize:
        quant_data = _shape_quant_map(weight_val)
        min_max_value = [x for data in quant_data for x in statistical_func(data)]
        min_max_value = [PER_CHANNEL] + min_max_value
    else:
        min_max_value = [PER_TENSOR] + [float(x) for x in statistical_func(weight_val)]
    return [CONST, USE_MINMAX] + min_max_value


def get_out_params(outs, calibrate_mode, quantize_type):
    """
    Channel quantization only supports NCHW layout model.
    For every layer's inputs and oututs, all dimensions except the second one
    will be seem as a whole.
    """

    def _shape_quant_map(datas):
        axis = 2 if len(datas.shape) > 2 else 1
        tp_axes = [i for i, _ in enumerate(datas.shape)]
        tp_axes[axis], tp_axes[0] = 0, axis
        datas = np.transpose(datas, tp_axes).reshape([datas.shape[axis], -1])
        return datas

    def _get_quant_axes(datas):
        pop_axis = 2 if len(datas.shape) > 2 else 1
        q_axes = [i for i, _ in enumerate(datas.shape)]
        q_axes.pop(pop_axis)
        return tuple(q_axes)

    elem_size = outs[0].size
    datas = np.array(outs)

    # (Fixme @chenf:) activation/input can not be per-channel quantizaiton.
    channel_quantize = False

    # Check for legitimacy
    if calibrate_mode not in statistical_func_map:
        raise Exception(f"Not support this calibrate mode: {calibrate_mode}")

    statistical_func = statistical_func_map[calibrate_mode][quantize_type]
    if channel_quantize:
        integral = np.sum([1 if elem_size == i else 0 for i in datas.shape[1:]])
        if integral:
            min_max_value = list(statistical_func(datas))
        else:
            if calibrate_mode == "kl_divergence":
                quant_data = _shape_quant_map(datas)
                min_max_value = [x for data in quant_data for x in statistical_func(data)]
            else:
                q_axes = _get_quant_axes(datas)
                mins, maxs = statistical_func(datas, q_axes)
                min_max_value = []
                for x, y in zip(mins, maxs):
                    min_max_value += [x] + [y]
        min_max_value = [PER_CHANNEL] + min_max_value
    else:
        min_max_value = [PER_TENSOR] + [float(x) for x in statistical_func(datas)]

    return [ACTIVATION, USE_MINMAX] + min_max_value


def get_layer_name(call, layer_index):
    layer_name = call.op.name.split(".")[-1]
    if call.span:
        layer_name = layer_name + "_" + call.span.source_name.name
    layer_name = layer_name + "_" + layer_index
    return layer_name


def calibration(module, dataset, curr_config):
    """Calibration: normal scale for uint8 asymmetric quantization,
        only use max and min value, to calculate scale and zero point.

    Parameters
    ---------
    module: Module
        The original module.

    dataset: list of dict of Var -> NDArray
        The calibration dataset.

    Returns
    -------
    ret: dict
        The nodes append quantization information

    """

    class GetLayerCount(relay.ExprVisitor):
        """get layer count"""

        def __init__(self):
            super(GetLayerCount, self).__init__()
            self.elem_count = {}
            self.layer_count = 0

        def enter_dict(self, hash_call):
            if hash_call in self.elem_count:
                self.elem_count[hash_call] += 1
            else:
                self.elem_count[hash_call] = 0

        def visit_call(self, call):
            _ = [self.visit(arg) for arg in call.args]
            self.layer_count += 1
            for i, arg in enumerate(call.args):
                if isinstance(arg, Tuple):
                    len_tuple = len(arg)
                    for j in range(len_tuple):
                        self.enter_dict(hash(arg.fields[j]))
                else:
                    self.enter_dict(hash(arg))

    class Calibration(relay.ExprVisitor):
        """get calibration params"""

        def __init__(self, inputs, pool, elem_count, layer_count, curr_config):
            super(Calibration, self).__init__()
            self.config = curr_config
            self.outs_map = {}
            self.quant_params = {}
            self.inputs = inputs
            self.input_count = len(self.inputs)
            self.pool = pool
            self.elem_count = elem_count
            if LOG >= logger.getEffectiveLevel():
                self.pbar = tqdm(total=layer_count)
                self.pbar.set_description_str("Calibrating")

        def clear_mem(self, call):
            hash_call = hash(call)
            if self.elem_count[hash_call] == 0:
                del self.outs_map[call]
                self.elem_count[hash_call] -= 1
            elif self.elem_count[hash_call] > 0:
                self.elem_count[hash_call] -= 1

        def _get_quant_params(self, call, data, kind):
            """
            kind:
                0: weights
                1: activation
            """
            if kind == ACTIVATION:
                s = get_out_params(
                    data,
                    self.config["calibrate_mode"],
                    self.config["activate_quantized_type"],
                )
            else:
                s = get_weight_params(data.data.asnumpy(), self.config)

            self.quant_params[call].append(s)

        def visit_var(self, var):
            quant_data = []
            new_args = []
            hash_call = hash(var)
            self.quant_params[hash_call] = []
            for in_data in self.inputs:
                data = in_data[var.name_hint]
                new_args.append(const(data, data.dtype))
                quant_data.append(data)
            self.outs_map[var] = new_args
            self._get_quant_params(hash_call, quant_data, ACTIVATION)

        def set_last_frame_to_sequence(self, init_sequence, last_frames, unavailable_frames):
            """prepare sequence frame for fsmn"""

            def _set_in_sequence(seq_fram, last_frame):
                seq_fram_data = seq_fram.data.asnumpy()
                last_frame_data = last_frame.data.asnumpy()
                out = np.zeros_like(seq_fram_data)
                out[:-1] = seq_fram_data[1:]
                out[-1] = last_frame_data
                return const(out, "float32")

            outputs = init_sequence
            # data in last_frames from one to self.input_count is the input frame
            for i in range(self.input_count):
                # get availabel frames
                if i > unavailable_frames:
                    for j, frame in enumerate(last_frames):
                        # only past-available-frame should be pushed in.
                        if i > j >= unavailable_frames:
                            outputs[i] = _set_in_sequence(outputs[i], frame)
            return outputs

        def generate_fsmn_constant(self, idx, sequence_block, frames, unavailable_frames):
            if idx == 3:
                init_sequence = [sequence_block for _ in range(self.input_count)]
                return self.set_last_frame_to_sequence(init_sequence, frames, unavailable_frames)
            elif idx == 4:
                return [const(i, "int32") for i in range(self.input_count)]

        def visit_call(self, call):
            """recursive traversal call"""
            assert call.op.name != "nn.batch_normal"
            _ = [self.visit(arg) for arg in call.args]
            if LOG >= logger.getEffectiveLevel():
                self.pbar.update(1)
            new_args = [[] for arg in call.args]
            hash_call = hash(call)
            self.quant_params[hash_call] = []
            for i, arg in enumerate(call.args):
                quant_data = []
                if isinstance(arg, Constant):
                    if call.op.name == "nn.fsmn" and i in (3, 4):
                        new_args[i] = self.generate_fsmn_constant(
                            i, arg, new_args[0], call.attrs.unavailable_frames
                        )
                    else:
                        new_args[i] = [arg for j in range(self.input_count)]
                    self._get_quant_params(hash_call, arg, CONST)

                elif isinstance(arg, (Call, TupleGetItem, Var)):
                    if arg in self.outs_map:
                        arg_val_list = self.outs_map[arg]
                        self.clear_mem(arg)
                    else:
                        raise Exception("can't find input.")
                    new_args[i] = arg_val_list
                    self.quant_params[hash_call].append(self.quant_params[hash(arg)][-1])

                elif isinstance(arg, Tuple):
                    len_tuple = len(arg)
                    field_val_lists = [[] for x in range(len_tuple)]
                    for j in range(len_tuple):
                        if arg.fields[j] in self.outs_map:
                            tuple_val_list = self.outs_map[arg.fields[j]]
                            self.clear_mem(arg.fields[j])
                        elif isinstance(arg.fields[j], Constant):
                            tuple_val_list = [arg.fields[j] for i in range(self.input_count)]
                            hash_const = hash(arg.fields[j])
                            self.quant_params[hash_const] = []
                            self._get_quant_params(hash_const, arg.fields[j], CONST)
                        else:
                            raise Exception("can't find input.")
                        field_val_lists[j] = tuple_val_list
                        self.quant_params[hash_call].append(
                            self.quant_params[hash(arg.fields[j])][-1]
                        )
                    for j in range(self.input_count):
                        new_tuple = Tuple([x[j] for x in field_val_lists])
                        new_args[i].append(new_tuple)

            self.outs_map[call] = []
            quant_data = []
            mo_flag = False
            nargs = []
            for x in new_args:
                if isinstance(x[0], Tuple):
                    etuple = []
                    for e in x[0]:
                        etuple.append(relay.var("var", shape=e.data.shape, dtype=e.data.dtype))
                    ntuple = Tuple(etuple)
                    nargs.append(ntuple)
                else:
                    nargs.append(relay.var("var", shape=x[0].data.shape, dtype=x[0].data.dtype))
            ncall = Call(call.op, nargs, call.attrs)
            mod = IRModule.from_expr(ncall)
            exc = relay.create_executor("graph", mod=mod, device=tvm.cpu(), target="llvm")
            infer_func = exc.evaluate()

            for i in range(self.input_count):
                args = []
                for x in new_args:
                    if isinstance(x[i], Tuple):
                        for c in x[i]:
                            args.append(c.data)
                    else:
                        args.append(x[i].data)
                value = infer_func(*args)
                if isinstance(value, tvm.nd.NDArray):
                    self.outs_map[call].append(const(value))
                    quant_data.append(value.asnumpy())
                else:
                    self.outs_map[call].append(value)
                    if not mo_flag:
                        quant_data = [[] for _ in value]
                    mo_flag = True
                    for j, x in enumerate(value):
                        data = x.asnumpy()
                        quant_data[j].append(data)
            if mo_flag:
                for data in quant_data:
                    self._get_quant_params(hash_call, data, ACTIVATION)
            else:
                self._get_quant_params(hash_call, quant_data, ACTIVATION)

        def visit_tuple_getitem(self, t):
            self.visit(t.tuple_value)
            hash_call = hash(t)
            if t.tuple_value in self.outs_map:
                tuple_value = self.outs_map[t.tuple_value]
            else:
                raise Exception("tuple getitem not find input.")
            self.outs_map[t] = []
            quant_data = []
            for i in range(self.input_count):
                data = tuple_value[i][t.index]
                self.outs_map[t].append(const(data))
                quant_data.append(data.asnumpy())
            self.quant_params[hash_call] = []
            self._get_quant_params(hash_call, quant_data, ACTIVATION)

    optimizer = GetLayerCount()
    optimizer.visit(module["main"])
    elem_count, layer_count = optimizer.elem_count, optimizer.layer_count
    get_out = Calibration(dataset, None, elem_count, layer_count, curr_config)
    get_out.visit(module["main"])
    if LOG >= logger.getEffectiveLevel():
        get_out.pbar.close()
    return get_out.quant_params


class csi_op:
    """All qnn csi ops"""

    def __init__(self):
        self.conv_handle = {
            "qnn.csi.conv1d": relay.qnn.op.csi_conv1d,
            "qnn.csi.conv2d": relay.qnn.op.csi_conv2d,
            "qnn.csi.conv2d_channel": relay.qnn.op.csi_conv2d_channel,
            "qnn.csi.conv2d_relu": relay.qnn.op.csi_conv2d_relu,
            "qnn.csi.conv2d_relu_channel": relay.qnn.op.csi_conv2d_relu_channel,
            "qnn.csi.conv2d_relu6": relay.qnn.op.csi_conv2d_relu6,
            "qnn.csi.conv2d_relu6_channel": relay.qnn.op.csi_conv2d_relu6_channel,
            "qnn.csi.conv3d": relay.qnn.op.csi_conv3d,
            "qnn.csi.deconv2d": relay.qnn.op.csi_deconv2d,
            "qnn.csi.deconv3d": relay.qnn.op.csi_deconv3d,
        }

        self.siso_handle = {
            "qnn.csi.abs": relay.qnn.op.csi_abs,
            "qnn.csi.acos": relay.qnn.op.csi_acos,
            "qnn.csi.acosh": relay.qnn.op.csi_acosh,
            "qnn.csi.argmax": relay.qnn.op.csi_argmax,
            "qnn.csi.argmin": relay.qnn.op.csi_argmin,
            "qnn.csi.asin": relay.qnn.op.csi_asin,
            "qnn.csi.asinh": relay.qnn.op.csi_asinh,
            "qnn.csi.atan": relay.qnn.op.csi_atan,
            "qnn.csi.atanh": relay.qnn.op.csi_atanh,
            "qnn.csi.avgpool2d": relay.qnn.op.csi_avgpool2d,
            "qnn.csi.avgpool3d": relay.qnn.op.csi_avgpool3d,
            "qnn.csi.batch_to_space_nd": relay.qnn.op.csi_batch_to_space_nd,
            "qnn.csi.broadcast_to": relay.qnn.op.csi_broadcast_to,
            "qnn.csi.cast": relay.qnn.op.csi_cast,
            "qnn.csi.ceil": relay.qnn.op.csi_ceil,
            "qnn.csi.clip": relay.qnn.op.csi_clip,
            "qnn.csi.cos": relay.qnn.op.csi_cos,
            "qnn.csi.cosh": relay.qnn.op.csi_cosh,
            "qnn.csi.depth_to_space": relay.qnn.op.csi_depth_to_space,
            "qnn.csi.erf": relay.qnn.op.csi_erf,
            "qnn.csi.exp": relay.qnn.op.csi_exp,
            "qnn.csi.expand_dims": relay.qnn.op.csi_expand_dims,
            "qnn.csi.flatten": relay.qnn.op.csi_flatten,
            "qnn.csi.floor": relay.qnn.op.csi_ceil,
            "qnn.csi.global_avgpool2d": relay.qnn.op.csi_global_avgpool2d,
            "qnn.csi.global_maxpool2d": relay.qnn.op.csi_global_maxpool2d,
            "qnn.csi.leaky_relu": relay.qnn.op.csi_leaky_relu,
            "qnn.csi.log": relay.qnn.op.csi_log,
            "qnn.csi.log_softmax": relay.qnn.op.csi_log_softmax,
            "qnn.csi.lrn": relay.qnn.op.csi_lrn,
            "qnn.csi.max": relay.qnn.op.csi_max,
            "qnn.csi.maxpool2d": relay.qnn.op.csi_maxpool2d,
            "qnn.csi.maxpool3d": relay.qnn.op.csi_maxpool3d,
            "qnn.csi.maxpool2d_locat": relay.qnn.op.csi_maxpool2d_locat,
            "qnn.csi.maxpool2d_with_argmax": relay.qnn.op.csi_maxpool2d_with_argmax,
            "qnn.csi.mean": relay.qnn.op.csi_mean,
            "qnn.csi.min": relay.qnn.op.csi_min,
            "qnn.csi.negative": relay.qnn.op.csi_negative,
            "qnn.csi.nn_deinit": relay.qnn.op.csinn_deinit,
            "qnn.csi.nn_init": relay.qnn.op.csinn_init,
            "qnn.csi.pad": relay.qnn.op.csi_pad,
            "qnn.csi.prod": relay.qnn.op.csi_prod,
            "qnn.csi.relu": relay.qnn.op.csi_relu,
            "qnn.csi.relu6": relay.qnn.op.csi_relu6,
            "qnn.csi.reshape": relay.qnn.op.csi_reshape,
            "qnn.csi.reverse": relay.qnn.op.csi_reverse,
            "qnn.csi.round": relay.qnn.op.csi_round,
            "qnn.csi.rsqrt": relay.qnn.op.csi_rsqrt,
            "qnn.csi.sigmoid": relay.qnn.op.csi_sigmoid,
            "qnn.csi.sign": relay.qnn.op.csi_sign,
            "qnn.csi.sin": relay.qnn.op.csi_sin,
            "qnn.csi.sinh": relay.qnn.op.csi_sinh,
            "qnn.csi.softmax": relay.qnn.op.csi_softmax,
            "qnn.csi.space_to_batch_nd": relay.qnn.op.csi_space_to_batch_nd,
            "qnn.csi.space_to_depth": relay.qnn.op.csi_space_to_depth,
            "qnn.csi.sqrt": relay.qnn.op.csi_sqrt,
            "qnn.csi.squeeze": relay.qnn.op.csi_squeeze,
            "qnn.csi.strided_slice": relay.qnn.op.csi_strided_slice,
            "qnn.csi.sum": relay.qnn.op.csi_sum,
            "qnn.csi.tan": relay.qnn.op.csi_tan,
            "qnn.csi.tanh": relay.qnn.op.csi_tanh,
            "qnn.csi.tile": relay.qnn.op.csi_tile,
            "qnn.csi.topk": relay.qnn.op.csi_topk,
            "qnn.csi.transpose": relay.qnn.op.csi_transpose,
            "qnn.csi.upsampling": relay.qnn.op.csi_upsampling,
            "qnn.csi.variance": relay.qnn.op.csi_variance,
        }

        self.diso_handle = {
            "qnn.csi.add": relay.qnn.op.csi_add,
            "qnn.csi.bias_add": relay.qnn.op.csi_bias_add,
            "qnn.csi.div": relay.qnn.op.csi_div,
            "qnn.csi.equal": relay.qnn.op.csi_equal,
            "qnn.csi.floor_div": relay.qnn.op.csi_floor_div,
            "qnn.csi.floor_mod": relay.qnn.op.csi_floor_mod,
            "qnn.csi.left_shift": relay.qnn.op.csi_left_shift,
            "qnn.csi.less": relay.qnn.op.csi_less,
            "qnn.csi.maximum": relay.qnn.op.csi_maximum,
            "qnn.csi.minimum": relay.qnn.op.csi_minimum,
            "qnn.csi.mod": relay.qnn.op.csi_mod,
            "qnn.csi.mul": relay.qnn.op.csi_mul,
            "qnn.csi.power": relay.qnn.op.csi_power,
            "qnn.csi.right_shift": relay.qnn.op.csi_right_shift,
            "qnn.csi.segment_max": relay.qnn.op.csi_segment_max,
            "qnn.csi.segment_mean": relay.qnn.op.csi_segment_mean,
            "qnn.csi.segment_min": relay.qnn.op.csi_segment_min,
            "qnn.csi.segment_prod": relay.qnn.op.csi_segment_prod,
            "qnn.csi.segment_sum": relay.qnn.op.csi_segment_sum,
            "qnn.csi.subtract": relay.qnn.op.csi_subtract,
            "qnn.csi.matmul": relay.qnn.op.csi_matmul,
        }

        self.other_handle = {
            "qnn.csi.bn": relay.qnn.op.csi_batch_norm,
            "qnn.csi.concatenate": relay.qnn.op.csi_concatenate,
            "qnn.csi.crop_resize": relay.qnn.op.csi_crop_resize,
            "qnn.csi.dense": relay.qnn.op.csi_dense,
            "qnn.csi.dilation2d": relay.qnn.op.csi_dilation2d,
            "qnn.csi.full": relay.qnn.op.csi_full,
            "qnn.csi.one_hot": relay.qnn.op.csi_one_hot,
            "qnn.csi.prelu": relay.qnn.op.csi_prelu,
            "qnn.csi.proposal": relay.qnn.op.csi_proposal,
            "qnn.csi.psroipooling": relay.qnn.op.csi_psroipooling,
            "qnn.csi.roipooling": relay.qnn.op.csi_roipooling,
            "qnn.csi.scatter_nd": relay.qnn.op.csi_scatter_nd,
            "qnn.csi.split": relay.qnn.op.csi_split,
            "qnn.csi.take": relay.qnn.op.csi_take,
            "qnn.csi.unpooling": relay.qnn.op.csi_unpooling,
            "qnn.csi.fsmn": relay.qnn.op.csi_fsmn,
            "qnn.csi.cache_matmul": relay.qnn.op.csi_cache_matmul,
            "qnn.csi.where": relay.qnn.op.csi_where,
            "qnn.csi.where_softmax": relay.qnn.op.csi_where_softmax,
            "qnn.csi.quantize": relay.qnn.op.csi_quantize,
            "qnn.csi.dequantize": relay.qnn.op.csi_dequantize,
            "qnn.csi.layer_norm": relay.qnn.op.csi_layer_norm,
        }

        self.all_handle = self._get_all_handle()

    def conv_op(self, name):
        return name in self.conv_handle

    def conv_handler(self, name):
        return self.conv_handle[name]

    def siso_op(self, name):
        return name in self.siso_handle

    def siso_handler(self, name):
        return self.siso_handle[name]

    def diso_op(self, name):
        return name in self.diso_handle

    def diso_handler(self, name):
        return self.diso_handle[name]

    def _get_all_handle(self):
        res = dict()
        res.update(**self.conv_handle, **self.siso_handle, **self.diso_handle, **self.other_handle)
        return res


def convert_to_csi_qnn(
    mod, quant_params, channel_quantization, channel_quantization_ratio_threshold
):
    """The convert_to_csi_qnn convert add ops to qnn.csi.* ops.

    Returns
    -------
    ret: Function
        The module pass function.
    """

    class ConvertToCSIMutator(relay.ExprMutator):
        """Convert tvm ops into csi ops"""

        def __init__(self):
            super(ConvertToCSIMutator, self).__init__()
            self.channel_quant = channel_quantization
            self.bias_init = [0, 0, 1, 0.0, 0.0] if self.channel_quant else [0, 0, 0, 0.0, 0.0]
            if quant_params:
                self.layer_index = list(enumerate(quant_params))
                self.quantitative_threshold = channel_quantization_ratio_threshold
                if self.quantitative_threshold and self.channel_quant:
                    logger.warning(
                        "Quantitative parameters optimizer will be used. "
                        + "In general, this optimizer will improve the accuracy, "
                        + "but not absolutely."
                    )

            self._idx = 0

        def get_lay_index(self, hash_call):
            """Get layer index."""
            res = ""
            if quant_params:
                for i, j in self.layer_index:
                    if j == hash_call:
                        res += str(i)
                        break
            if res == "":
                res = str(self._idx)
                self._idx += 1
            return res

        def q_params_optimizer(self, q_params, current_args, op_name):
            """Quantitative parameters optimizer for channel quantization.
            In general, this optimizer will improve the accuracy, but not absolutely.

            logic code:
                for (channel_min/max) in quantitative_params:
                    ratio = (channel_min/max) / (per_tensor_min/max)
                    if ratio < threshold:
                        channel_min/max *= 2
            """

            if q_params[0][0] != 1:
                logger.warning(
                    "%s is not quantized by channel quantize, it will not be optimized.", op_name
                )
                return
            for j in range(len(q_params)):
                q_param = q_params[j]
                if len(q_param) <= 3:
                    continue
                if j < len(current_args):
                    if isinstance(current_args[j], Constant):
                        continue
                min_ = np.min(q_param[3:])
                max_ = np.max(q_param[3:])
                for i in range(3, len(q_param)):
                    if i % 2 == 1:
                        ratio = q_param[i] / min_ if min_ != 0 else 0
                    else:
                        ratio = q_param[i] / max_ if max_ != 0 else 0
                    if ratio != 0 and ratio <= self.quantitative_threshold:
                        q_params[j][i] = q_param[i] * 2

        def get_quant_params(self, hash_call, op_args, call):
            """get quant info"""
            if quant_params:
                q_params = quant_params[hash_call]
                if self.quantitative_threshold and self.channel_quant:
                    self.q_params_optimizer(q_params, op_args, call.op.name)
            else:
                len_op = len(op_args)
                if len_op == 1 and isinstance(op_args[0], Tuple):
                    len_op = len(op_args[0])
                q_params = [[1, 0, 0, 0.0, 0.0]] * (len_op + 1)
            return q_params

        def visit_call(self, call):
            op_args = [self.visit(arg) for arg in call.args]
            cts = call.attrs
            hash_call = hash(call)
            q_params = self.get_quant_params(hash_call, op_args, call)
            layer_index = self.get_lay_index(hash_call)

            if call.op.name == "nn.conv2d":
                data = op_args[0]
                weight = op_args[1]
                bias = relay.expr.const(0, dtype="float32")
                q_params.insert(2, self.bias_init)
                new_call = relay.qnn.op.csi_conv2d(
                    data,
                    weight,
                    bias,
                    cts.strides,
                    cts.padding,
                    cts.dilation,
                    cts.groups,
                    cts.channels,
                    cts.kernel_size,
                    cts.data_layout,
                    cts.kernel_layout,
                    cts.out_layout,
                    cts.out_dtype,
                    q_params,
                    layer_name=get_layer_name(call, layer_index),
                )
            elif call.op.name == "nn.conv1d":
                data = op_args[0]
                weight = op_args[1]
                bias = relay.expr.const(0, dtype="float32")
                q_params.insert(2, self.bias_init)
                new_call = relay.qnn.op.csi_conv1d(
                    data,
                    weight,
                    bias,
                    cts.strides,
                    cts.padding,
                    cts.dilation,
                    cts.groups,
                    cts.channels,
                    cts.kernel_size,
                    cts.data_layout,
                    cts.kernel_layout,
                    cts.out_layout,
                    cts.out_dtype,
                    q_params,
                    layer_name=get_layer_name(call, layer_index),
                )
            elif call.op.name == "nn.conv3d":
                data = op_args[0]
                weight = op_args[1]
                bias = relay.expr.const(0, dtype="float32")
                q_params.insert(2, self.bias_init)
                new_call = relay.qnn.op.csi_conv3d(
                    data,
                    weight,
                    bias,
                    cts.strides,
                    cts.padding,
                    cts.dilation,
                    cts.groups,
                    cts.channels,
                    cts.kernel_size,
                    cts.data_layout,
                    cts.kernel_layout,
                    cts.out_layout,
                    cts.out_dtype,
                    q_params,
                    layer_name=get_layer_name(call, layer_index),
                )
            elif call.op.name == "image.dilation2d":
                data = op_args[0]
                weight = op_args[1]
                new_call = relay.qnn.op.csi_dilation2d(
                    data,
                    weight,
                    cts.strides,
                    cts.padding,
                    cts.dilations,
                    cts.data_layout,
                    cts.kernel_layout,
                    cts.out_dtype,
                    q_params,
                    layer_name=get_layer_name(call, layer_index),
                )
            elif call.op.name == "nn.dense":
                data = op_args[0]
                weight = op_args[1]
                units = cts.units
                if units is None:
                    units = _infer_shape(weight)[0]
                else:
                    units = int(units)
                bias = relay.expr.const(np.zeros([units], dtype=np.float32), dtype="float32")
                q_params.insert(2, self.bias_init)
                new_call = relay.qnn.op.csi_dense(
                    data,
                    weight,
                    bias,
                    cts.units,
                    "float32",
                    q_params,
                    layer_name=get_layer_name(call, layer_index),
                )
            elif call.op.name == "nn.bias_add":
                lhs = op_args[0]
                rhs = op_args[1]
                in_shape = list(_infer_shape(lhs))
                if (
                    (
                        not isinstance(lhs, Call)
                        or (
                            isinstance(lhs, Call)
                            and lhs.op.name not in ("qnn.csi.conv2d", "qnn.csi.deconv2d")
                        )
                    )
                    and isinstance(rhs, Constant)
                    and len(in_shape) == 4
                ):
                    rhs_data = rhs.data.asnumpy()
                    shape_map = {
                        0: (-1, 1, 1, 1),
                        1: (1, -1, 1, 1),
                        2: (1, 1, -1, 1),
                        3: (1, 1, 1, -1),
                    }
                    new_rhs_data = np.reshape(rhs_data, shape_map[cts.axis])
                    new_rhs = relay.expr.const(new_rhs_data, dtype="float32")
                    new_call = relay.qnn.op.csi_add(
                        lhs, new_rhs, q_params, layer_name=get_layer_name(call, layer_index)
                    )
                else:
                    new_call = relay.qnn.op.csi_bias_add(
                        lhs, rhs, q_params, layer_name=get_layer_name(call, layer_index)
                    )
            elif call.op.name == "nn.relu":
                pre_call = op_args[0]
                new_call = relay.qnn.op.csi_relu(
                    pre_call, "float32", q_params, layer_name=get_layer_name(call, layer_index)
                )
            elif call.op.name == "sin":
                data = op_args[0]
                new_call = relay.qnn.op.csi_sin(
                    data, "float32", q_params, layer_name=get_layer_name(call, layer_index)
                )
            elif call.op.name == "cos":
                data = op_args[0]
                new_call = relay.qnn.op.csi_cos(
                    data, "float32", q_params, layer_name=get_layer_name(call, layer_index)
                )
            elif call.op.name == "tan":
                data = op_args[0]
                new_call = relay.qnn.op.csi_tan(
                    data, "float32", q_params, layer_name=get_layer_name(call, layer_index)
                )
            elif call.op.name == "asin":
                data = op_args[0]
                new_call = relay.qnn.op.csi_asin(
                    data, "float32", q_params, layer_name=get_layer_name(call, layer_index)
                )
            elif call.op.name == "acos":
                data = op_args[0]
                new_call = relay.qnn.op.csi_acos(
                    data, "float32", q_params, layer_name=get_layer_name(call, layer_index)
                )
            elif call.op.name == "atan":
                data = op_args[0]
                new_call = relay.qnn.op.csi_atan(
                    data, "float32", q_params, layer_name=get_layer_name(call, layer_index)
                )
            elif call.op.name == "sinh":
                data = op_args[0]
                new_call = relay.qnn.op.csi_sinh(
                    data, "float32", q_params, layer_name=get_layer_name(call, layer_index)
                )
            elif call.op.name == "cosh":
                data = op_args[0]
                new_call = relay.qnn.op.csi_cosh(
                    data, "float32", q_params, layer_name=get_layer_name(call, layer_index)
                )
            elif call.op.name == "tanh":
                data = op_args[0]
                new_call = relay.qnn.op.csi_tanh(
                    data, "float32", q_params, layer_name=get_layer_name(call, layer_index)
                )
            elif call.op.name == "asinh":
                data = op_args[0]
                new_call = relay.qnn.op.csi_asinh(
                    data, "float32", q_params, layer_name=get_layer_name(call, layer_index)
                )
            elif call.op.name == "acosh":
                data = op_args[0]
                new_call = relay.qnn.op.csi_acosh(
                    data, "float32", q_params, layer_name=get_layer_name(call, layer_index)
                )
            elif call.op.name == "atanh":
                data = op_args[0]
                new_call = relay.qnn.op.csi_atanh(
                    data, "float32", q_params, layer_name=get_layer_name(call, layer_index)
                )
            elif call.op.name == "segment_max":
                data = op_args[0]
                segment_ids = op_args[1]
                new_call = relay.qnn.op.csi_segment_max(
                    data,
                    segment_ids,
                    cts.num_segments,
                    "float32",
                    q_params,
                    layer_name=get_layer_name(call, layer_index),
                )
            elif call.op.name == "segment_min":
                data = op_args[0]
                segment_ids = op_args[1]
                new_call = relay.qnn.op.csi_segment_min(
                    data,
                    segment_ids,
                    cts.num_segments,
                    "float32",
                    q_params,
                    layer_name=get_layer_name(call, layer_index),
                )
            elif call.op.name == "segment_mean":
                data = op_args[0]
                segment_ids = op_args[1]
                new_call = relay.qnn.op.csi_segment_mean(
                    data,
                    segment_ids,
                    cts.num_segments,
                    "float32",
                    q_params,
                    layer_name=get_layer_name(call, layer_index),
                )
            elif call.op.name == "segment_prod":
                data = op_args[0]
                segment_ids = op_args[1]
                new_call = relay.qnn.op.csi_segment_prod(
                    data,
                    segment_ids,
                    cts.num_segments,
                    "float32",
                    q_params,
                    layer_name=get_layer_name(call, layer_index),
                )
            elif call.op.name == "segment_sum":
                data = op_args[0]
                segment_ids = op_args[1]
                new_call = relay.qnn.op.csi_segment_sum(
                    data,
                    segment_ids,
                    cts.num_segments,
                    "float32",
                    q_params,
                    layer_name=get_layer_name(call, layer_index),
                )
            elif call.op.name == "nn.batch_norm":
                data = op_args[0]
                gamma = op_args[1]
                beta = op_args[2]
                moving_mean = op_args[3]
                moving_var = op_args[4]
                new_call = relay.qnn.op.csi_batch_norm(
                    data,
                    gamma,
                    beta,
                    moving_mean,
                    moving_var,
                    cts.axis,
                    cts.epsilon,
                    cts.center,
                    cts.scale,
                    q_params,
                    layer_name=get_layer_name(call, layer_index),
                )
            elif call.op.name == "nn.batch_matmul":
                data_a = op_args[0]
                data_b = op_args[1]
                bias = relay.expr.const(0, dtype="float32")
                q_params.insert(2, self.bias_init)
                new_call = relay.qnn.op.csi_matmul(
                    data_a,
                    data_b,
                    bias,
                    cts.transpose_a,
                    cts.transpose_b,
                    "float32",
                    q_params,
                    layer_name=get_layer_name(call, layer_index),
                )
            elif call.op.name == "nn.adaptive_avg_pool1d":
                data = op_args[0]
                if cts.output_size[0] == 1:
                    new_call = relay.qnn.op.csi_mean(
                        data,
                        [2],
                        True,
                        False,
                        "float32",
                        q_params,
                        layer_name=get_layer_name(call, layer_index),
                    )
                else:
                    raise ValueError("Cannot convert op:", call.op.name)
            elif call.op.name == "nn.avg_pool2d":
                data = op_args[0]
                new_call = relay.qnn.op.csi_avgpool2d(
                    data,
                    "float32",
                    cts.strides,
                    cts.padding,
                    cts.dilation,
                    cts.pool_size,
                    cts.ceil_mode,
                    cts.count_include_pad,
                    cts.layout,
                    q_params,
                    layer_name=get_layer_name(call, layer_index),
                )
            elif call.op.name == "nn.avg_pool3d":
                data = op_args[0]
                new_call = relay.qnn.op.csi_avgpool3d(
                    data,
                    "float32",
                    cts.strides,
                    cts.padding,
                    cts.pool_size,
                    cts.ceil_mode,
                    cts.count_include_pad,
                    cts.layout,
                    q_params,
                    layer_name=get_layer_name(call, layer_index),
                )
            elif call.op.name == "nn.max_pool3d":
                data = op_args[0]
                new_call = relay.qnn.op.csi_maxpool3d(
                    data,
                    "float32",
                    cts.strides,
                    cts.padding,
                    cts.pool_size,
                    cts.ceil_mode,
                    cts.layout,
                    q_params,
                    layer_name=get_layer_name(call, layer_index),
                )
            elif call.op.name == "nn.global_avg_pool2d":
                data = op_args[0]
                new_call = relay.qnn.op.csi_global_avgpool2d(
                    data,
                    cts.layout,
                    "float32",
                    q_params,
                    layer_name=get_layer_name(call, layer_index),
                )
            elif call.op.name == "nn.global_max_pool2d":
                data = op_args[0]
                new_call = relay.qnn.op.csi_global_maxpool2d(
                    data,
                    cts.layout,
                    "float32",
                    q_params,
                    layer_name=get_layer_name(call, layer_index),
                )
            elif call.op.name == "nn.max_pool2d":
                data = op_args[0]
                new_call = relay.qnn.op.csi_maxpool2d(
                    data,
                    "float32",
                    cts.strides,
                    cts.padding,
                    cts.dilation,
                    cts.pool_size,
                    cts.ceil_mode,
                    cts.layout,
                    q_params,
                    layer_name=get_layer_name(call, layer_index),
                )
            elif call.op.name == "reshape":
                data = op_args[0]
                new_call = relay.qnn.op.csi_reshape(
                    data,
                    cts.newshape,
                    "float32",
                    q_params,
                    layer_name=get_layer_name(call, layer_index),
                )
            elif call.op.name == "squeeze":
                data = op_args[0]
                ishape = _infer_shape(data)
                new_shape = []
                if cts.axis is None:
                    for x in ishape:
                        if x != 1:
                            new_shape.append(x)
                else:
                    dims = len(ishape)
                    for x in range(dims):
                        if x not in cts.axis:
                            new_shape.append(ishape[x])
                new_call = relay.qnn.op.csi_reshape(
                    data,
                    new_shape,
                    "float32",
                    q_params,
                    layer_name=get_layer_name(call, layer_index),
                )
            elif call.op.name == "nn.softmax":
                data = op_args[0]
                new_call = relay.qnn.op.csi_softmax(
                    data,
                    cts.axis,
                    "float32",
                    q_params,
                    layer_name=get_layer_name(call, layer_index),
                )
            elif call.op.name == "scatter_nd":
                data = op_args[0]
                indices = op_args[1]
                updates = op_args[2]
                new_call = relay.qnn.op.csi_scatter_nd(
                    data,
                    indices,
                    updates,
                    "float32",
                    q_params,
                    layer_name=get_layer_name(call, layer_index),
                )
            elif call.op.name == "reverse":
                data = op_args[0]
                axis = cts.axis.value
                new_call = relay.qnn.op.csi_reverse(
                    data,
                    axis,
                    "float32",
                    q_params,
                    layer_name=get_layer_name(call, layer_index),
                )
            elif call.op.name == "negative":
                data = op_args[0]
                new_call = relay.qnn.op.csi_negative(
                    data, "float32", q_params, layer_name=get_layer_name(call, layer_index)
                )
            elif call.op.name == "nn.log_softmax":
                data = op_args[0]
                new_call = relay.qnn.op.csi_log_softmax(
                    data,
                    cts.axis,
                    "float32",
                    q_params,
                    layer_name=get_layer_name(call, layer_index),
                )
            elif call.op.name == "nn.lrn":
                data = op_args[0]
                new_call = relay.qnn.op.csi_lrn(
                    data,
                    cts.size,
                    cts.axis,
                    cts.alpha,
                    cts.beta,
                    cts.bias,
                    cts.norm_region,
                    "float32",
                    q_params,
                    layer_name=get_layer_name(call, layer_index),
                )
            elif call.op.name == "concatenate":
                data = op_args[0]
                axis = cts.axis
                if axis < 0:
                    in_shape = _infer_shape(call)
                    axis += len(in_shape)
                new_call = relay.qnn.op.csi_concatenate(
                    data, axis, q_params, layer_name=get_layer_name(call, layer_index)
                )
            elif call.op.name == "add":
                lhs = op_args[0]
                rhs = op_args[1]
                if isinstance(lhs, Constant):
                    lhs, rhs = rhs, lhs
                    q_params[0], q_params[1] = q_params[1], q_params[0]
                if isinstance(rhs, Constant):
                    rhs_value = rhs.data.asnumpy()
                    rhs_shape = list(rhs_value.shape)
                    lhs_shape = _infer_shape(lhs)

                    if len(rhs_shape) < len(lhs_shape):
                        left_axis = len(lhs_shape) - len(rhs_shape)
                        for i in range(left_axis):
                            rhs_shape.insert(0, 1)
                        rhs_value = np.reshape(rhs_value, rhs_shape)
                        rhs = relay.expr.const(rhs_value, "float32")
                new_call = relay.qnn.op.csi_add(
                    lhs, rhs, q_params, layer_name=get_layer_name(call, layer_index)
                )
            elif call.op.name == "equal":
                new_call = relay.qnn.op.csi_equal(
                    *op_args, q_params, layer_name=get_layer_name(call, layer_index)
                )
            elif call.op.name == "subtract":
                lhs = op_args[0]
                rhs = op_args[1]
                new_call = relay.qnn.op.csi_subtract(
                    lhs, rhs, q_params, layer_name=get_layer_name(call, layer_index)
                )
                if isinstance(rhs, Constant):
                    rhs_value = rhs.data.asnumpy()
                    len_shape = len(rhs_value.shape)
                    if len_shape in [0, 1]:
                        if len_shape == 1:
                            rhs_value = rhs_value[0]
                        if abs(rhs_value - 0) < 1e-5:
                            new_call = lhs
            elif call.op.name == "nn.leaky_relu":
                data = op_args[0]
                new_call = relay.qnn.op.csi_leaky_relu(
                    data,
                    cts.alpha,
                    "float32",
                    q_params,
                    layer_name=get_layer_name(call, layer_index),
                )
            elif call.op.name == "nn.upsampling":
                data = op_args[0]
                new_call = relay.qnn.op.csi_upsampling(
                    data,
                    cts.scale_h,
                    cts.scale_w,
                    cts.align_corners,
                    cts.method,
                    "float32",
                    cts.layout,
                    q_params,
                    layer_name=get_layer_name(call, layer_index),
                )
            elif call.op.name == "image.resize2d":
                data = op_args[0]
                origin_shape = (call.type_args)[0].concrete_shape
                assert len(origin_shape) == 4, "Only support 4-dim shape of image.resize"
                scale_h = int(cts.size[0]) / origin_shape[2]
                scale_w = int(cts.size[1]) / origin_shape[3]
                align_corners = cts.coordinate_transformation_mode == "align_corners"
                new_call = relay.qnn.op.csi_upsampling(
                    data,
                    scale_h,
                    scale_w,
                    align_corners,
                    cts.method,
                    "float32",
                    cts.layout,
                    q_params,
                    layer_name=get_layer_name(call, layer_index),
                )

            elif call.op.name == "nn.conv2d_transpose":
                data = op_args[0]
                weight = op_args[1]
                bias = relay.expr.const(0, dtype="float32")
                q_params.insert(2, self.bias_init)
                new_call = relay.qnn.op.csi_deconv2d(
                    data,
                    weight,
                    bias,
                    cts.strides,
                    cts.padding,
                    cts.dilation,
                    cts.groups,
                    cts.channels,
                    cts.kernel_size,
                    cts.data_layout,
                    cts.kernel_layout,
                    cts.out_layout,
                    cts.output_padding,
                    "float32",
                    q_params,
                    layer_name=get_layer_name(call, layer_index),
                )
            elif call.op.name == "nn.conv3d_transpose":
                data = op_args[0]
                weight = op_args[1]
                bias = relay.expr.const(0, dtype="float32")
                q_params.insert(2, self.bias_init)
                new_call = relay.qnn.op.csi_deconv3d(
                    data,
                    weight,
                    bias,
                    cts.strides,
                    cts.padding,
                    cts.dilation,
                    cts.groups,
                    cts.channels,
                    cts.kernel_size,
                    cts.data_layout,
                    cts.kernel_layout,
                    cts.out_layout,
                    cts.output_padding,
                    "float32",
                    q_params,
                    layer_name=get_layer_name(call, layer_index),
                )
            elif call.op.name == "transpose":
                data = op_args[0]
                new_call = relay.qnn.op.csi_transpose(
                    data,
                    cts.axes,
                    "float32",
                    q_params,
                    layer_name=get_layer_name(call, layer_index),
                )
            elif call.op.name == "nn.batch_flatten":
                data = op_args[0]
                in_shape = _infer_shape(data)
                new_shape = [in_shape[0], -1]
                new_call = relay.qnn.op.csi_reshape(
                    data,
                    new_shape,
                    "float32",
                    q_params,
                    layer_name=get_layer_name(call, layer_index),
                )
            elif call.op.name == "sigmoid":
                data = op_args[0]
                new_call = relay.qnn.op.csi_sigmoid(
                    data, "float32", q_params, layer_name=get_layer_name(call, layer_index)
                )
            elif call.op.name == "vision.proposal":
                cls_prob = op_args[0]
                bbox_pred = op_args[1]
                im_info = op_args[2]
                new_call = relay.qnn.op.csi_proposal(
                    cls_prob,
                    bbox_pred,
                    im_info,
                    cts.scales,
                    cts.ratios,
                    cts.feature_stride,
                    cts.threshold,
                    cts.rpn_pre_nms_top_n,
                    cts.rpn_post_nms_top_n,
                    cts.rpn_min_size,
                    cts.iou_loss,
                    "float32",
                    q_params,
                    layer_name=get_layer_name(call, layer_index),
                )
            elif call.op.name == "vision.psroipooling":
                cls_prob = op_args[0]
                roi = op_args[1]
                new_call = relay.qnn.op.csi_psroipooling(
                    cls_prob,
                    roi,
                    cts.spatial_scale,
                    cts.output_dim,
                    cts.group_size,
                    "float32",
                    q_params,
                    layer_name=get_layer_name(call, layer_index),
                )
            elif call.op.name == "vision.roi_pool":
                data = op_args[0]
                roi = op_args[1]
                new_call = relay.qnn.op.csi_roipooling(
                    data,
                    roi,
                    cts.pooled_size,
                    cts.spatial_scale,
                    "float32",
                    q_params,
                    layer_name=get_layer_name(call, layer_index),
                )
            elif call.op.name == "multiply":
                lhs = op_args[0]
                rhs = op_args[1]
                if isinstance(lhs, Constant):
                    lhs, rhs = rhs, lhs
                    q_params[0], q_params[1] = q_params[1], q_params[0]
                if isinstance(rhs, Constant):
                    rhs_value = rhs.data.asnumpy()
                    rhs_shape = list(rhs_value.shape)
                    lhs_shape = _infer_shape(lhs)

                    if len(rhs_shape) < len(lhs_shape):
                        left_axis = len(lhs_shape) - len(rhs_shape)
                        for i in range(left_axis):
                            rhs_shape.insert(0, 1)
                        rhs_value = np.reshape(rhs_value, rhs_shape)
                        rhs = relay.expr.const(rhs_value, "float32")
                new_call = relay.qnn.op.csi_mul(
                    lhs, rhs, q_params, layer_name=get_layer_name(call, layer_index)
                )
                if isinstance(rhs, Constant):
                    rhs_value = rhs.data.asnumpy()
                    len_shape = len(rhs_value.shape)
                    if len_shape in [0, 1]:
                        if len_shape == 1:
                            rhs_value = rhs_value[0]
                        if abs(rhs_value - 1) < 1e-5:
                            new_call = lhs
            elif call.op.name == "divide":
                lhs = op_args[0]
                rhs = op_args[1]
                if isinstance(rhs, Constant) and q_params[1][1] == USE_MINMAX:
                    rhs_value = rhs.data.asnumpy()
                    if rhs_value.dtype == "float32":
                        rhs_value = 1.0 / rhs.data.asnumpy()
                        q_params[1][3:] = list(1.0 / np.array(q_params[1][3:], dtype=np.float32))
                        rhs = relay.expr.const(rhs_value, "float32")
                        new_call = relay.qnn.op.csi_mul(
                            lhs, rhs, q_params, layer_name=get_layer_name(call, layer_index)
                        )
                else:
                    new_call = relay.qnn.op.csi_div(
                        lhs, rhs, q_params, layer_name=get_layer_name(call, layer_index)
                    )
            elif call.op.name == "power":
                lhs = op_args[0]
                rhs = op_args[1]
                new_call = relay.qnn.op.csi_power(
                    lhs, rhs, q_params, layer_name=get_layer_name(call, layer_index)
                )
            elif call.op.name == "mod":
                lhs = op_args[0]
                rhs = op_args[1]
                new_call = relay.qnn.op.csi_mod(
                    lhs, rhs, q_params, layer_name=get_layer_name(call, layer_index)
                )
            elif call.op.name == "nn.prelu":
                data = op_args[0]
                alpha = op_args[1]
                new_call = relay.qnn.op.csi_prelu(
                    data,
                    alpha,
                    cts.axis,
                    "float32",
                    q_params,
                    layer_name=get_layer_name(call, layer_index),
                )
            elif call.op.name == "nn.max_pool2d_with_argmax":
                data = op_args[0]
                new_call = relay.qnn.op.csi_maxpool2d_with_argmax(
                    data,
                    "float32",
                    cts.strides,
                    cts.padding,
                    cts.dilation,
                    cts.pool_size,
                    cts.ceil_mode,
                    cts.layout,
                    q_params,
                    layer_name=get_layer_name(call, layer_index),
                )
            elif call.op.name == "mean":
                data = op_args[0]
                new_call = relay.qnn.op.csi_mean(
                    data,
                    cts.axis,
                    cts.keepdims,
                    cts.exclude,
                    "float32",
                    q_params,
                    layer_name=get_layer_name(call, layer_index),
                )
            elif call.op.name == "prod":
                data = op_args[0]
                new_call = relay.qnn.op.csi_prod(
                    data,
                    cts.axis,
                    cts.keepdims,
                    cts.exclude,
                    "float32",
                    q_params,
                    layer_name=get_layer_name(call, layer_index),
                )
            elif call.op.name == "max":
                data = op_args[0]
                new_call = relay.qnn.op.csi_max(
                    data,
                    cts.axis,
                    cts.keepdims,
                    cts.exclude,
                    "float32",
                    q_params,
                    layer_name=get_layer_name(call, layer_index),
                )
            elif call.op.name == "min":
                data = op_args[0]
                new_call = relay.qnn.op.csi_min(
                    data,
                    cts.axis,
                    cts.keepdims,
                    cts.exclude,
                    "float32",
                    q_params,
                    layer_name=get_layer_name(call, layer_index),
                )
            elif call.op.name == "sum":
                data = op_args[0]
                new_call = relay.qnn.op.csi_sum(
                    data,
                    cts.axis,
                    cts.keepdims,
                    cts.exclude,
                    "float32",
                    q_params,
                    layer_name=get_layer_name(call, layer_index),
                )
            elif call.op.name == "argmax":
                data = op_args[0]
                new_call = relay.qnn.op.csi_argmax(
                    data,
                    cts.axis,
                    cts.keepdims,
                    cts.exclude,
                    "float32",
                    q_params,
                    layer_name=get_layer_name(call, layer_index),
                )
            elif call.op.name == "argmin":
                data = op_args[0]
                new_call = relay.qnn.op.csi_argmin(
                    data,
                    cts.axis,
                    cts.keepdims,
                    cts.exclude,
                    "float32",
                    q_params,
                    layer_name=get_layer_name(call, layer_index),
                )
            elif call.op.name == "nn.pad":
                data = op_args[0]
                pad_value = op_args[1]
                new_call = relay.qnn.op.csi_pad(
                    data,
                    pad_value,
                    cts.pad_width,
                    cts.pad_mode,
                    "float32",
                    q_params,
                    layer_name=get_layer_name(call, layer_index),
                )
            elif call.op.name == "clip":
                pre_call = op_args[0]
                new_call = relay.qnn.op.csi_clip(
                    pre_call,
                    cts.a_min,
                    cts.a_max,
                    "float32",
                    q_params,
                    layer_name=get_layer_name(call, layer_index),
                )
                if [cts.a_min, cts.a_max] == [0, 6]:
                    new_call = relay.qnn.op.csi_relu6(
                        pre_call,
                        "float32",
                        q_params,
                        layer_name=get_layer_name(call, layer_index),
                    )
            elif call.op.name == "vision.max_pool2d_location":
                data = op_args[0]
                new_call = relay.qnn.op.csi_maxpool2d_locat(
                    data,
                    cts.strides,
                    cts.padding,
                    cts.pool_size,
                    cts.ceil_mode,
                    "float32",
                    cts.layout,
                    q_params,
                    layer_name=get_layer_name(call, layer_index),
                )
            elif call.op.name == "vision.unpooling":
                data = op_args[0]
                mask = op_args[1]
                scale = [cts.scale_h, cts.scale_w]
                out_padding = [cts.pad_out_h, cts.pad_out_w]
                new_call = relay.qnn.op.csi_unpooling(
                    data,
                    mask,
                    scale,
                    out_padding,
                    "float32",
                    cts.layout,
                    q_params,
                    layer_name=get_layer_name(call, layer_index),
                )
            elif call.op.name == "strided_slice":
                data = op_args[0]
                if len(cts.strides) == 0:
                    strides = [1] * len(cts.begin)
                else:
                    strides = cts.strides
                begin = [int(i) for i in cts.begin]
                end = [int(i) for i in cts.end]
                if cts.slice_mode == "size":
                    end = list(map(lambda x: x[0] + x[1], zip(begin, end)))

                if cts.axes is not None:
                    input_shape = list(_infer_shape(data))
                    expand_begin = [0 for i in input_shape]
                    expand_end = list(input_shape)
                    expand_strides = [1 for i in input_shape]

                    for idx, axes in enumerate(list(cts.axes)):
                        expand_begin[int(axes)] = begin[idx]
                        expand_end[int(axes)] = end[idx]
                        expand_strides[int(axes)] = strides[idx]

                    begin = expand_begin
                    end = expand_end
                    strides = expand_strides
                if cts.axes is None and len(cts.strides) != len(cts.begin):
                    strides = [1] * len(cts.begin)
                new_call = relay.qnn.op.csi_strided_slice(
                    data,
                    begin,
                    end,
                    strides,
                    "float32",
                    q_params,
                    layer_name=get_layer_name(call, layer_index),
                )
            elif call.op.name == "split":
                data = op_args[0]
                if not quant_params:
                    if isinstance(cts.indices_or_sections, tvm.tir.IntImm):
                        q_params_len = cts.indices_or_sections.value + 1
                    else:
                        q_params_len = len(cts.indices_or_sections) + 2
                    q_params = [[1, 0, 0, 0.0, 0.0]] * q_params_len
                new_call = relay.qnn.op.csi_split(
                    data,
                    cts.indices_or_sections,
                    cts.axis,
                    "float32",
                    q_params,
                    layer_name=get_layer_name(call, layer_index),
                )
            elif call.op.name == "variance":
                data = op_args[0]
                new_call = relay.qnn.op.csi_variance(
                    data,
                    cts.axis,
                    cts.keepdims,
                    cts.exclude,
                    "float32",
                    q_params,
                    layer_name=get_layer_name(call, layer_index),
                )
            elif call.op.name == "exp":
                data = op_args[0]
                new_call = relay.qnn.op.csi_exp(
                    data, "float32", q_params, layer_name=get_layer_name(call, layer_index)
                )
            elif call.op.name == "log":
                data = op_args[0]
                new_call = relay.qnn.op.csi_log(
                    data, "float32", q_params, layer_name=get_layer_name(call, layer_index)
                )
            elif call.op.name == "abs":
                data = op_args[0]
                new_call = relay.qnn.op.csi_abs(
                    data, "float32", q_params, layer_name=get_layer_name(call, layer_index)
                )
            elif call.op.name == "expand_dims":
                data = op_args[0]
                new_shape = list(_infer_shape(data))
                axis = (
                    cts.axis
                    if isinstance(cts.axis, (tuple, list))
                    else [
                        cts.axis,
                    ]
                )
                for i in range(cts.num_newaxis):
                    new_shape.insert(cts.axis, 1)
                new_call = relay.qnn.op.csi_reshape(
                    data,
                    new_shape,
                    "float32",
                    q_params,
                    layer_name=get_layer_name(call, layer_index),
                )
            elif call.op.name == "broadcast_to":
                data = op_args[0]
                pre_shape = list(_infer_shape(call.args[0]))
                out_shape = list(cts.shape)
                if pre_shape == out_shape:
                    new_call = data
                else:
                    pre_size = np.array(pre_shape).prod()
                    out_size = np.array(out_shape).prod()
                    if pre_size == out_size:
                        if data.op.name == "qnn.csi.reshape":
                            new_call = relay.qnn.op.csi_reshape(
                                data.args[0],
                                cts.shape,
                                "float32",
                                q_params,
                                layer_name=get_layer_name(call, layer_index),
                            )
                        else:
                            new_call = relay.qnn.op.csi_reshape(
                                data,
                                cts.shape,
                                "float32",
                                q_params,
                                layer_name=get_layer_name(call, layer_index),
                            )
                    else:
                        new_call = relay.qnn.op.csi_broadcast_to(
                            data,
                            cts.shape,
                            "float32",
                            q_params,
                            layer_name=get_layer_name(call, layer_index),
                        )
            elif call.op.name == "cast":
                data = op_args[0]
                new_call = relay.qnn.op.csi_cast(
                    data, cts.dtype, q_params, get_layer_name(call, layer_index)
                )
            elif call.op.name == "ceil":
                data = op_args[0]
                new_call = relay.qnn.op.csi_ceil(
                    data, "float32", q_params, layer_name=get_layer_name(call, layer_index)
                )
            elif call.op.name == "floor":
                data = op_args[0]
                new_call = relay.qnn.op.csi_floor(
                    data, "float32", q_params, layer_name=get_layer_name(call, layer_index)
                )
            elif call.op.name == "round":
                data = op_args[0]
                new_call = relay.qnn.op.csi_round(
                    data, "float32", q_params, layer_name=get_layer_name(call, layer_index)
                )
            elif call.op.name == "minimum":
                lhs = op_args[0]
                rhs = op_args[1]
                new_call = relay.qnn.op.csi_minimum(
                    lhs, rhs, q_params, layer_name=get_layer_name(call, layer_index)
                )
            elif call.op.name == "maximum":
                lhs = op_args[0]
                rhs = op_args[1]
                new_call = relay.qnn.op.csi_maximum(
                    lhs, rhs, q_params, layer_name=get_layer_name(call, layer_index)
                )
            elif call.op.name == "right_shift":
                lhs = op_args[0]
                rhs = op_args[1]
                new_call = relay.qnn.op.csi_right_shift(
                    lhs, rhs, q_params, layer_name=get_layer_name(call, layer_index)
                )
            elif call.op.name == "left_shift":
                lhs = op_args[0]
                rhs = op_args[1]
                new_call = relay.qnn.op.csi_left_shift(
                    lhs, rhs, q_params, layer_name=get_layer_name(call, layer_index)
                )
            elif call.op.name == "floor_divide":
                lhs = op_args[0]
                rhs = op_args[1]
                new_call = relay.qnn.op.csi_floor_div(
                    lhs, rhs, q_params, layer_name=get_layer_name(call, layer_index)
                )
            elif call.op.name == "floor_mod":
                lhs = op_args[0]
                rhs = op_args[1]
                new_call = relay.qnn.op.csi_floor_mod(
                    lhs, rhs, q_params, layer_name=get_layer_name(call, layer_index)
                )
            elif call.op.name == "image.crop_and_resize":
                data = op_args[0]
                boxes = op_args[1]
                box_indices = op_args[2]
                new_call = relay.qnn.op.csi_crop_resize(
                    data,
                    boxes,
                    box_indices,
                    cts.crop_size,
                    cts.layout,
                    cts.method,
                    cts.extrapolation_value,
                    "float32",
                    q_params,
                    layer_name=get_layer_name(call, layer_index),
                )
            elif call.op.name == "nn.depth_to_space":
                data = op_args[0]
                new_call = relay.qnn.op.csi_depth_to_space(
                    data,
                    cts.block_size,
                    cts.layout,
                    cts.mode,
                    "float32",
                    q_params,
                    layer_name=get_layer_name(call, layer_index),
                )
            elif call.op.name == "nn.batch_to_space_nd":
                data = op_args[0]
                new_call = relay.qnn.op.csi_batch_to_space_nd(
                    data,
                    cts.block_shape,
                    cts.crops,
                    "float32",
                    q_params,
                    layer_name=get_layer_name(call, layer_index),
                )
            elif call.op.name == "nn.space_to_batch_nd":
                data = op_args[0]
                new_call = relay.qnn.op.csi_space_to_batch_nd(
                    data,
                    cts.block_shape,
                    cts.paddings,
                    cts.pad_value,
                    "float32",
                    q_params,
                    layer_name=get_layer_name(call, layer_index),
                )
            elif call.op.name == "nn.space_to_depth":
                data = op_args[0]
                new_call = relay.qnn.op.csi_space_to_depth(
                    data,
                    cts.block_size,
                    cts.layout,
                    cts.mode,
                    "float32",
                    q_params,
                    layer_name=get_layer_name(call, layer_index),
                )
            elif call.op.name == "erf":
                data = op_args[0]
                new_call = relay.qnn.op.csi_erf(
                    data, "float32", q_params, layer_name=get_layer_name(call, layer_index)
                )
            elif call.op.name == "sqrt":
                data = op_args[0]
                new_call = relay.qnn.op.csi_sqrt(
                    data, "float32", q_params, layer_name=get_layer_name(call, layer_index)
                )
            elif call.op.name == "rsqrt":
                data = op_args[0]
                new_call = relay.qnn.op.csi_rsqrt(
                    data, "float32", q_params, layer_name=get_layer_name(call, layer_index)
                )
            elif call.op.name == "sign":
                data = op_args[0]
                new_call = relay.qnn.op.csi_sign(
                    data, "float32", q_params, layer_name=get_layer_name(call, layer_index)
                )
            elif call.op.name == "full":
                data = op_args[0]
                new_call = relay.qnn.op.csi_full(
                    data,
                    cts.shape,
                    cts.dtype,
                    q_params,
                    layer_name=get_layer_name(call, layer_index),
                )
            elif call.op.name == "take":
                data = op_args[0]
                indices = op_args[1]
                if isinstance(indices, Var):
                    ctype = indices.checked_type
                    if ctype.dtype != "int64":
                        indices = relay.var(indices.name_hint, shape=ctype.shape, dtype="int64")
                elif isinstance(indices, Constant):
                    indices_data = indices.data.asnumpy()
                    dtype = indices_data.dtype
                    if dtype != "int64":
                        indices_data = indices_data.astype("int64")
                        indices = relay.expr.const(indices_data, "int64")
                axis = cts.axis.value if hasattr(cts.axis, "value") else cts.axis
                new_call = relay.qnn.op.csi_take(
                    data,
                    indices,
                    axis,
                    cts.mode,
                    "float32",
                    q_params,
                    layer_name=get_layer_name(call, layer_index),
                )
            elif call.op.name == "tile":
                data = op_args[0]
                new_call = relay.qnn.op.csi_tile(
                    data,
                    cts.reps,
                    "float32",
                    q_params,
                    layer_name=get_layer_name(call, layer_index),
                )
            elif call.op.name == "topk":
                data = op_args[0]
                k = cts.k.value
                new_call = relay.qnn.op.csi_topk(
                    data,
                    k,
                    cts.axis,
                    cts.ret_type,
                    cts.is_ascend,
                    cts.dtype,
                    "float32",
                    q_params,
                    layer_name=get_layer_name(call, layer_index),
                )
            elif call.op.name == "unravel_index":
                data = op_args[0]
                shape = op_args[1]
                new_call = relay.qnn.op.csi_unravel_index(
                    data,
                    shape,
                    "float32",
                    q_params,
                    layer_name=get_layer_name(call, layer_index),
                )
            # for custom ops
            elif call.op.name == "nn.fsmn":
                frame = op_args[0]
                l_filter = op_args[1]
                r_filter = op_args[2]
                frame_sequence = op_args[3]
                frame_counter = op_args[4]
                if quant_params:
                    # set input quantize params to frame_sequence and remove frame counter
                    q_params[3] = q_params[0]
                else:
                    q_params = [
                        [ACTIVATION, USE_SCALE, PER_TENSOR, 0.0, 0],
                        [CONST, USE_SCALE, PER_TENSOR, 0.0, 0],
                        [CONST, USE_SCALE, PER_TENSOR, 0.0, 0],
                        [ACTIVATION, USE_SCALE, PER_TENSOR, 0.0, 0],
                        [ACTIVATION, USE_SCALE, PER_TENSOR, 0.0, 0],
                    ]
                new_call = relay.qnn.op.csi_fsmn(
                    frame,
                    l_filter,
                    r_filter,
                    frame_sequence,
                    frame_counter,
                    cts.l_order,
                    cts.r_order,
                    cts.l_stride,
                    cts.r_stride,
                    cts.unavailable_frames,
                    "float32",
                    q_params,
                    layer_name=get_layer_name(call, layer_index),
                )
            elif call.op.name == "cache_matmul":
                new_call = relay.qnn.op.csi_cache_matmul(
                    *op_args,
                    cts.cache_shape,
                    cts.shape,
                    cts.axes,
                    "float32",
                    q_params,
                    layer_name=get_layer_name(call, layer_index),
                )
            elif call.op.name == "cache_conv1d":
                new_call = relay.qnn.op.csi_cache_conv1d(
                    *op_args,
                    cts.cache_shape,
                    cts.strides,
                    cts.padding,
                    cts.dilation,
                    cts.groups,
                    cts.channels,
                    cts.kernel_size,
                    cts.data_layout,
                    cts.kernel_layout,
                    cts.out_layout,
                    cts.out_dtype,
                    q_params,
                    layer_name=get_layer_name(call, layer_index),
                )
            elif call.op.name == "nn.layer_norm":
                new_call = relay.qnn.op.csi_layer_norm(
                    *op_args,
                    cts.axis,
                    cts.epsilon,
                    cts.center,
                    cts.scale,
                    "float32",
                    q_params,
                    layer_name=get_layer_name(call, layer_index),
                )
            elif call.op.name == "less":
                new_call = relay.qnn.op.csi_less(
                    *op_args,
                    q_params,
                    layer_name=get_layer_name(call, layer_index),
                )
            elif call.op.name == "one_hot":
                one_value = op_args[1].data.asnumpy()
                off_value = op_args[2].data.asnumpy()
                if one_value == 1.0 and off_value == 0.0:
                    new_call = relay.qnn.op.csi_one_hot(
                        op_args[0],
                        cts.depth,
                        cts.axis,
                        "float32",
                        q_params,
                        layer_name=get_layer_name(call, layer_index),
                    )
                else:
                    raise ValueError("Unsupport one_hot with one_value and off_value")
            elif call.op.name == "where":
                new_call = relay.qnn.op.csi_where(
                    *op_args,
                    "float32",
                    q_params,
                    layer_name=get_layer_name(call, layer_index),
                )
            elif call.op.name == "qnn.quantize":
                data = op_args[0]
                scale = op_args[1]
                zero_point = op_args[2]
                new_call = relay.qnn.op.csi_quantize(
                    data,
                    scale,
                    zero_point,
                    cts.axis,
                    cts.out_dtype,
                    q_params,
                    layer_name=get_layer_name(call, layer_index),
                )
            elif call.op.name == "qnn.dequantize":
                data = op_args[0]
                scale = op_args[1]
                zero_point = op_args[2]
                new_call = relay.qnn.op.csi_dequantize(
                    data,
                    scale,
                    zero_point,
                    cts.axis,
                    "float32",
                    q_params,
                    layer_name=get_layer_name(call, layer_index),
                )
            else:
                raise ValueError("Cannot convert op:", call.op.name)

            return new_call

        def visit_tuple_getitem(self, op):
            tuple_value = self.visit(op.tuple_value)
            if not tuple_value.same_as(op.tuple_value):
                if tuple_value.op.name == "qnn.csi.bn":
                    return tuple_value
                return TupleGetItem(tuple_value, op.index)
            return tuple_value

    func = ConvertToCSIMutator().visit(mod["main"])
    func = function.Function(analysis.free_vars(func.body), func.body)
    mod["main"] = func

    return mod


def _get_array_value(data):
    out = []
    for x in data:
        if isinstance(x, tvm.ir.container.Array):
            out.append(_get_array_value(x))
        else:
            out.append(x.value)
    return out


def _qnn_attrs(attrs):
    ret = {}
    for i in dir(attrs):
        if not i.startswith("_") and i not in ["handle", "same_as"]:
            ret[i] = getattr(attrs, i)
            if isinstance(ret[i], tvm.ir.container.Array):
                ret[i] = _get_array_value(ret[i])
            elif isinstance(ret[i], tvm.tir.expr.IntImm):
                ret[i] = ret[i].value

    return ret


def _get_csi_op(name):
    return csi_op().all_handle[name]


@function_pass(opt_level=1)
class FuseBias:
    """Fuse bias class which only valid in NCHW layout"""

    def __init__(self):
        self.target_ops = [
            "qnn.csi.conv2d",
            "qnn.csi.dense",
            "qnn.csi.deconv2d",
            "qnn.csi.conv1d",
        ]

    def get_new_op(self, call, pre_call, op_args):
        new_attrs = _qnn_attrs(pre_call.attrs)
        data = pre_call.args[0]
        weight = pre_call.args[1]
        bias = op_args[1]
        new_attrs["q_params"][-1] = call.attrs.q_params[-1]  # output
        new_attrs["q_params"][2] = call.attrs.q_params[1]  # bias
        new_attrs["layer_name"] += "_fuse_" + call.attrs.layer_name
        return _get_csi_op(pre_call.op.name)(data, weight, bias, **new_attrs)

    def transform_function(self, func, mod, ctx):
        """patten and convert op"""

        func = relay.Function(func.params, func.body, None, func.type_params, func.attrs)
        fuse_bias = self

        class FuseBiasMutator(relay.ExprMutator):
            """Fuse bias"""

            def visit_call(self, call):
                op_args = [self.visit(arg) for arg in call.args]
                pre_call = op_args[0]
                if call.op.name == "qnn.csi.bias_add":
                    if not isinstance(pre_call, Call):
                        return Call(call.op, op_args, call.attrs, call.type_args, call.span)

                    if pre_call.op.name in fuse_bias.target_ops:
                        return fuse_bias.get_new_op(call, pre_call, op_args)

                elif call.op.name == "qnn.csi.add":
                    if not isinstance(pre_call, Call) or not isinstance(op_args[1], Constant):
                        return Call(call.op, op_args, call.attrs, call.type_args, call.span)
                    in_name = pre_call.op.name
                    if in_name not in fuse_bias.target_ops:
                        return Call(call.op, op_args, call.attrs, call.type_args, call.span)

                    bias = op_args[1].data.asnumpy()
                    b_shape = bias.shape
                    in_shape = _infer_shape(pre_call)
                    need_broadcast = False
                    b_rank = len(b_shape)
                    if b_rank == 1:
                        b_size = b_shape[0]
                        need_broadcast = b_shape[0] == 1
                    elif b_rank == 0:
                        need_broadcast = True
                    else:
                        return Call(call.op, op_args, call.attrs, call.type_args, call.span)

                    if need_broadcast:
                        if in_name == "qnn.csi.dense":
                            bias = np.zeros(in_shape[2]) + bias
                            op_args[1] = relay.const(bias)
                            return fuse_bias.get_new_op(call, pre_call, op_args)
                        else:
                            bias = np.zeros(in_shape[1]) + bias
                            op_args[1] = relay.const(bias)
                            return fuse_bias.get_new_op(call, pre_call, op_args)
                    else:
                        if in_name == "qnn.csi.dense":
                            if b_size == in_shape[-1]:
                                return fuse_bias.get_new_op(call, pre_call, op_args)
                        else:
                            if b_size == in_shape[1]:
                                return fuse_bias.get_new_op(call, pre_call, op_args)

                return Call(call.op, op_args, call.attrs, call.type_args, call.span)

        return FuseBiasMutator().visit(func)


@function_pass(opt_level=1)
class FuseConvRelu:
    """Fuse relu layer helper class"""

    def transform_function(self, func, mod, ctx):
        """patten and convert op"""
        func = relay.Function(func.params, func.body, None, func.type_params, func.attrs)

        class FuseConvReluMutator(relay.ExprMutator):
            """Fuse conv and relu"""

            def visit_call(self, call):
                op_args = [self.visit(arg) for arg in call.args]

                if call.op.name == "qnn.csi.relu":
                    pre_call = op_args[0]
                    if isinstance(pre_call, Call) and pre_call.op.name == "qnn.csi.conv2d":
                        new_attrs = _qnn_attrs(pre_call.attrs)
                        data = pre_call.args[0]
                        weight = pre_call.args[1]
                        bias = pre_call.args[2]
                        new_attrs["q_params"][-1] = call.attrs.q_params[-1]
                        new_attrs["layer_name"] += "_fuse_" + call.attrs.layer_name
                        new_call = relay.qnn.op.csi_conv2d_relu(data, weight, bias, **new_attrs)
                        return new_call

                # elif pre_call.op.name == "qnn.csi.dense":
                #     data = pre_call.args[0]
                #     weight = pre_call.args[1]
                #     bias = pre_call.op_args[2]
                #     new_attrs['axis'] = 0
                #     new_attrs['output_scale'] = call.attrs.output_scale
                #     new_attrs['output_zero_point'] = call.attrs.output_zero_point
                #     new_call = relay.qnn.op.csi_dense(data, weight, bias, **new_attrs)
                # elif pre_call.op.name == "qnn.csi.deconv2d":
                #     data = pre_call.args[0]
                #     weight = pre_call.args[1]
                #     bias = pre_call.op_args[2]
                #     new_attrs['output_scale'] = call.attrs.output_scale
                #     new_attrs['output_zero_point'] = call.attrs.output_zero_point
                #     new_call = relay.qnn.op.csi_deconv2d(data, weight, bias, **new_attrs)
                elif call.op.name == "qnn.csi.relu6":
                    pre_call = op_args[0]
                    if pre_call.op.name == "qnn.csi.conv2d":
                        new_attrs = _qnn_attrs(pre_call.attrs)
                        data = pre_call.args[0]
                        weight = pre_call.args[1]
                        bias = pre_call.args[2]
                        new_attrs["q_params"][-1] = call.attrs.q_params[-1]
                        new_attrs["layer_name"] += "_fuse_" + call.attrs.layer_name
                        new_call = relay.qnn.op.csi_conv2d_relu6(data, weight, bias, **new_attrs)
                        return new_call

                new_call = Call(call.op, op_args, call.attrs, call.type_args, call.span)

                return new_call

        return FuseConvReluMutator().visit(func)


@function_pass(opt_level=1)
class FusePad:
    """Fuse pad layer helper class"""

    def transform_function(self, func, mod, ctx):
        """patten and convert op"""
        func = relay.Function(func.params, func.body, None, func.type_params, func.attrs)

        class FusePadMutator(relay.ExprMutator):
            """Fuse pad"""

            def visit_call(self, call):
                op_args = [self.visit(arg) for arg in call.args]

                if call.op.name == "qnn.csi.conv2d":
                    pre_call = op_args[0]
                    if not pre_call or isinstance(pre_call, tvm.relay.expr.Var):
                        new_call = Call(call.op, op_args, call.attrs, call.type_args, call.span)
                        return new_call

                    if isinstance(pre_call, Call) and pre_call.op.name == "qnn.csi.pad":
                        if not pre_call.attrs.pad_mode == "constant":
                            new_call = Call(call.op, op_args, call.attrs, call.type_args, call.span)
                            return new_call

                        new_attrs = _qnn_attrs(call.attrs)
                        data = pre_call.args[0]
                        weight = op_args[1]
                        bias = op_args[2]

                        new_attrs["q_params"][0] = pre_call.attrs.q_params[0]

                        pad_len = len(call.attrs.padding)
                        if pad_len == 4:
                            new_attrs["padding"] = [
                                pre_call.attrs.pad_width[2][0],
                                pre_call.attrs.pad_width[2][1],
                                pre_call.attrs.pad_width[3][0],
                                pre_call.attrs.pad_width[3][1],
                            ]
                        elif pad_len == 2:
                            new_attrs["padding"] = [
                                pre_call.attrs.pad_width[2][0],
                                pre_call.attrs.pad_width[3][0],
                            ]
                        else:
                            raise ValueError("Unsupport padding size:", pad_len)
                        new_attrs["layer_name"] += "_fuse_" + pre_call.attrs.layer_name
                        new_call = relay.qnn.op.csi_conv2d(data, weight, bias, **new_attrs)
                    else:
                        new_call = Call(call.op, op_args, call.attrs, call.type_args, call.span)
                else:
                    new_call = Call(call.op, op_args, call.attrs, call.type_args, call.span)
                return new_call

        return FusePadMutator().visit(func)


@function_pass(opt_level=1)
class FuseReshapeDense:
    """Fuse reshape helper class"""

    def transform_function(self, func, mod, ctx):
        """patten and convert op"""
        func = relay.Function(func.params, func.body, None, func.type_params, func.attrs)

        class FuseReshapeDenseMutator(relay.ExprMutator):
            """Fuse reshape and dense"""

            def visit_call(self, call):
                op_args = [self.visit(arg) for arg in call.args]

                if call.op.name == "qnn.csi.dense":
                    pre_call = op_args[0]
                    new_attrs = _qnn_attrs(call.attrs)
                    if isinstance(pre_call, Call) and pre_call.op.name == "qnn.csi.reshape":
                        data = pre_call.args[0]
                        if isinstance(data, Var):
                            return Call(call.op, op_args, call.attrs, call.type_args, call.span)
                        weight = call.args[1]
                        bias = call.args[2]
                        new_attrs["layer_name"] += "_fuse_" + pre_call.attrs.layer_name
                        return relay.qnn.op.csi_dense(data, weight, bias, **new_attrs)

                return Call(call.op, op_args, call.attrs, call.type_args, call.span)

        return FuseReshapeDenseMutator().visit(func)


@function_pass(opt_level=1)
class FuseReshape:
    """Fuse reshape helper class"""

    def transform_function(self, func, mod, ctx):
        """patten and convert op"""
        func = relay.Function(func.params, func.body, None, func.type_params, func.attrs)

        class FuseReshapeMutator(relay.ExprMutator):
            """Fuse reshape"""

            def visit_call(self, call):
                op_args = [self.visit(arg) for arg in call.args]

                if call.op.name == "qnn.csi.reshape":
                    pre_call = op_args[0]
                    in_shape = _infer_shape(pre_call)
                    curt_shape = _infer_shape(call)
                    if isinstance(pre_call, Call) and pre_call.op.name == "qnn.csi.reshape":
                        pre_attrs = _qnn_attrs(pre_call.attrs)
                        crt_attrs = _qnn_attrs(call.attrs)
                        crt_attrs["newshape"] = curt_shape
                        crt_attrs["q_params"][0] = pre_attrs["q_params"][0]
                        crt_attrs["layer_name"] += "_fuse_" + pre_attrs["layer_name"]
                        return relay.qnn.op.csi_reshape(pre_call.args[0], **crt_attrs)
                    elif in_shape == curt_shape:
                        return pre_call
                return Call(call.op, op_args, call.attrs, call.type_args, call.span)

        return FuseReshapeMutator().visit(func)


@function_pass(opt_level=1)
class FuseClip:
    """Fuse clip helper class"""

    def __init__(self):
        self.changed_layer = {}

    def transform_function(self, func, mod, ctx):
        """patten and convert op"""
        func = relay.Function(func.params, func.body, None, func.type_params, func.attrs)
        fuse_clip = self

        class FuseClipMutator(relay.ExprMutator):
            """Fuse Clip"""

            def visit_call(self, call):
                op_args = [self.visit(arg) for arg in call.args]
                current_attrs = _qnn_attrs(call.attrs)
                if call.op.name in ["qnn.csi.clip", "qnn.csi.relu6"]:
                    pre_call = op_args[0]
                    if isinstance(pre_call, Call):
                        pre_attrs = _qnn_attrs(pre_call.attrs)
                        pre_attrs["q_params"][-1] = current_attrs["q_params"][-1]
                        pre_attrs["layer_name"] += "_fuse_" + call.attrs.layer_name
                        new_call = _get_csi_op(pre_call.op.name)(*pre_call.args, **pre_attrs)
                        fuse_clip.changed_layer[hash(new_call)] = pre_attrs["q_params"][-1]
                    else:
                        new_call = _get_csi_op(call.op.name)(*op_args, **current_attrs)
                else:
                    for idx, arg in enumerate(op_args):
                        hash_arg = hash(arg)
                        if hash_arg in fuse_clip.changed_layer:
                            current_attrs["q_params"][idx] = fuse_clip.changed_layer[hash_arg]
                    new_call = _get_csi_op(call.op.name)(*op_args, **current_attrs)
                return new_call

        return FuseClipMutator().visit(func)


@function_pass(opt_level=1)
class FuseRelu:
    """Fuse relu helper class"""

    def __init__(self):
        self.changed_layer = {}

    def transform_function(self, func, mod, ctx):
        """patten and convert op"""
        func = relay.Function(func.params, func.body, None, func.type_params, func.attrs)
        fuse_relu = self

        class FuseReluMutator(relay.ExprMutator):
            """Fuse relu"""

            def visit_call(self, call):
                op_args = [self.visit(arg) for arg in call.args]
                current_attrs = _qnn_attrs(call.attrs)
                if call.op.name == "qnn.csi.relu":
                    pre_call = op_args[0]
                    if isinstance(pre_call, Call):
                        pre_attrs = _qnn_attrs(pre_call.attrs)
                        pre_attrs["q_params"][-1] = current_attrs["q_params"][-1]
                        pre_attrs["layer_name"] += "_fuse_" + call.attrs.layer_name
                        new_call = _get_csi_op(pre_call.op.name)(*pre_call.args, **pre_attrs)
                        fuse_relu.changed_layer[hash(new_call)] = pre_attrs["q_params"][-1]
                    else:
                        new_call = _get_csi_op(call.op.name)(*op_args, **current_attrs)
                else:
                    for idx, arg in enumerate(op_args):
                        hash_arg = hash(arg)
                        if hash_arg in fuse_relu.changed_layer:
                            current_attrs["q_params"][idx] = fuse_relu.changed_layer[hash_arg]
                    new_call = _get_csi_op(call.op.name)(*op_args, **current_attrs)
                return new_call

        return FuseReluMutator().visit(func)


def fuse_params_add_mul_before_conv(weight, bias, mul_val, add_val):
    """update the params in convolution op while add or/and mul op in front of it."""
    assert len(weight.shape) == 4
    new_weight = weight * mul_val
    new_bias = weight * add_val
    new_bias = np.sum(new_bias, (1, 2, 3))
    new_bias = new_bias + bias

    return new_weight.astype(np.float32), new_bias.reshape(-1).astype(np.float32)


def update_conv_attrs(weight_val, attrs, config):
    """update the attrubutions for conv2d op with new weight value."""
    min_max_val = get_weight_params(weight_val, config)

    attrs["q_params"][1] = min_max_val


@function_pass(opt_level=1)
class FuseAddBeforeConv:
    """Fuse add op in front of the convolution op."""

    def __init__(self, curr_config):
        self.config = curr_config

    def transform_function(self, func, mod, ctx):
        """patten and convert op"""
        func = relay.Function(func.params, func.body, None, func.type_params, func.attrs)
        _fuse_add = self

        class FuseAddBeforeConvMutator(relay.ExprMutator):
            """ "Fuse add op before conv"""

            def visit_call(self, call):
                op_args = [self.visit(arg) for arg in call.args]

                if call.op.name == "qnn.csi.conv2d":
                    new_conv2d_attrs = _qnn_attrs(call.attrs)
                    pre_call = op_args[0]
                    if (
                        isinstance(pre_call, Call)
                        and (pre_call.op.name in ("qnn.csi.add", "qnn.csi.bias_add"))
                        and isinstance(pre_call.args[1], Constant)
                        and sum(new_conv2d_attrs["padding"]) == 0
                    ):
                        data = pre_call.args[0]
                        weight = op_args[1]
                        bias = op_args[2]

                        weight_val = weight.data.asnumpy()
                        bias_val = bias.data.asnumpy()
                        add_rhs_val = pre_call.args[1].data.asnumpy()

                        if len(bias_val.shape) == 0:
                            bias_val = np.zeros(weight_val.shape[0])
                        if len(add_rhs_val.shape) == 1:
                            add_rhs_val = np.reshape(add_rhs_val, (1, add_rhs_val.shape[0], 1, 1))

                        if (
                            add_rhs_val.size != weight_val.shape[1]
                            or new_conv2d_attrs["groups"] > 1
                        ):
                            new_call = Call(call.op, op_args, call.attrs, call.type_args, call.span)
                            return new_call

                        mul_rhs_val = np.ones_like(add_rhs_val)

                        new_weight_val, new_bias_val = fuse_params_add_mul_before_conv(
                            weight_val, bias_val, mul_rhs_val, add_rhs_val
                        )

                        new_conv2d_attrs["q_params"][0] = pre_call.attrs.q_params[0]
                        update_conv_attrs(new_weight_val, new_conv2d_attrs, _fuse_add.config)

                        weight.data.copyfrom(new_weight_val)
                        bias = relay.expr.const(new_bias_val)

                        new_call = relay.qnn.op.csi_conv2d(data, weight, bias, **new_conv2d_attrs)
                        return new_call
                    else:
                        new_call = Call(call.op, op_args, call.attrs, call.type_args, call.span)
                else:
                    new_call = Call(call.op, op_args, call.attrs, call.type_args, call.span)
                return new_call

        return FuseAddBeforeConvMutator().visit(func)


@function_pass(opt_level=1)
class FuseMulBeforeConv:
    """Fuse mul op in front of the convolution op."""

    def __init__(self, curr_config):
        self.config = curr_config

    def transform_function(self, func, mod, ctx):
        """patten and convert op"""
        func = relay.Function(func.params, func.body, None, func.type_params, func.attrs)
        _fuse_mul = self

        class FuseMulBeforeConvMutator(relay.ExprMutator):
            """Fuse mul before conv"""

            def visit_call(self, call):
                op_args = [self.visit(arg) for arg in call.args]

                if call.op.name == "qnn.csi.conv2d":
                    new_conv2d_attrs = _qnn_attrs(call.attrs)
                    pre_call = op_args[0]
                    if (
                        isinstance(pre_call, Call)
                        and pre_call.op.name == "qnn.csi.mul"
                        and isinstance(pre_call.args[1], Constant)
                    ):
                        data = pre_call.args[0]
                        weight = op_args[1]
                        bias = op_args[2]

                        weight_val = weight.data.asnumpy()
                        bias_val = bias.data.asnumpy()
                        mul_rhs_val = pre_call.args[1].data.asnumpy()

                        if len(bias_val.shape) == 0:
                            bias_val = np.zeros(weight_val.shape[0])
                        if len(mul_rhs_val.shape) in [0, 1]:
                            if len(mul_rhs_val.shape) == 1:
                                mul_rhs_val = mul_rhs_val[0]
                            mul_rhs_val = np.full((1, weight_val.shape[1], 1, 1), mul_rhs_val)
                        if (
                            mul_rhs_val.size != mul_rhs_val.shape[1]
                            or new_conv2d_attrs["groups"] > 1
                        ):
                            new_call = Call(call.op, op_args, call.attrs, call.type_args, call.span)
                            return new_call

                        add_rhs_val = np.zeros_like(mul_rhs_val)

                        new_weight_val, new_bias_val = fuse_params_add_mul_before_conv(
                            weight_val, bias_val, mul_rhs_val, add_rhs_val
                        )

                        new_conv2d_attrs["q_params"][0] = pre_call.attrs.q_params[0]

                        update_conv_attrs(new_weight_val, new_conv2d_attrs, _fuse_mul.config)

                        weight.data.copyfrom(new_weight_val)
                        bias = relay.expr.const(new_bias_val)
                        new_conv2d_attrs["layer_name"] += "_fuse_" + pre_call.attrs.layer_name
                        new_call = relay.qnn.op.csi_conv2d(data, weight, bias, **new_conv2d_attrs)
                        return new_call
                    else:
                        new_call = Call(call.op, op_args, call.attrs, call.type_args, call.span)
                else:
                    new_call = Call(call.op, op_args, call.attrs, call.type_args, call.span)
                return new_call

        return FuseMulBeforeConvMutator().visit(func)


def fuse_params_mul_after_conv(weight, mul_val):
    """update the params in convolution op while mul op in behind it."""
    assert len(weight.shape) == 4
    mul_val = np.reshape(mul_val, (-1, 1, 1, 1))
    new_weight = weight * mul_val
    return new_weight.astype(np.float32)


@function_pass(opt_level=1)
class FuseAddAfterConv:
    """Fuse add op in behind the convolution op."""

    def __init__(self, curr_config):
        self.config = curr_config

    def transform_function(self, func, mod, ctx):
        """patten and convert op"""
        func = relay.Function(func.params, func.body, None, func.type_params, func.attrs)
        _fuse_add = self

        class FuseAddAfterConvMutator(relay.ExprMutator):
            """Fuse add after conv"""

            def visit_call(self, call):
                op_args = [self.visit(arg) for arg in call.args]
                if call.op.name in ("qnn.csi.add", "qnn.csi.bias_add") and isinstance(
                    op_args[1], Constant
                ):
                    pre_call = op_args[0]
                    if not isinstance(pre_call, Call):
                        return Call(call.op, op_args, call.attrs, call.type_args, call.span)
                    if pre_call.op.name == "qnn.csi.conv2d":
                        new_conv2d_attrs = _qnn_attrs(pre_call.attrs)
                        data = pre_call.args[0]
                        weight = pre_call.args[1]
                        bias = pre_call.args[2]

                        weight_val = weight.data.asnumpy()
                        bias_val = bias.data.asnumpy()
                        add_rhs_val = op_args[1].data.asnumpy()

                        if add_rhs_val.size != weight_val.shape[0]:
                            new_call = Call(call.op, op_args, call.attrs, call.type_args, call.span)
                            return new_call

                        if len(bias_val.shape) == 0:
                            bias_val = np.zeros(weight_val.shape[0])

                        new_bias_val = add_rhs_val.reshape(-1) + bias_val
                        new_conv2d_attrs["q_params"][-1] = call.attrs.q_params[-1]
                        new_conv2d_attrs["q_params"][2] = get_weight_params(
                            new_bias_val, _fuse_add.config
                        )
                        bias = relay.expr.const(new_bias_val)
                        new_conv2d_attrs["layer_name"] += "_fuse_" + call.attrs.layer_name
                        new_call = relay.qnn.op.csi_conv2d(data, weight, bias, **new_conv2d_attrs)
                        return new_call
                    elif pre_call.op.name == "qnn.csi.dense":
                        new_dense_attrs = _qnn_attrs(pre_call.attrs)
                        data = pre_call.args[0]
                        weight = pre_call.args[1]
                        bias = pre_call.args[2]

                        weight_val = weight.data.asnumpy()
                        bias_val = bias.data.asnumpy()
                        add_rhs_val = op_args[1].data.asnumpy()

                        if add_rhs_val.size != weight_val.shape[0]:
                            new_call = Call(call.op, op_args, call.attrs, call.type_args, call.span)
                            return new_call

                        if len(bias_val.shape) == 0:
                            bias_val = np.zeros(weight_val.shape[0])

                        new_bias_val = add_rhs_val.reshape(bias_val.shape) + bias_val

                        new_dense_attrs["q_params"][-1] = call.attrs.q_params[-1]

                        new_dense_attrs["q_params"][2] = get_weight_params(
                            new_bias_val, _fuse_add.config
                        )
                        bias = relay.expr.const(new_bias_val)
                        new_dense_attrs["layer_name"] += "_fuse_" + call.attrs.layer_name
                        new_call = relay.qnn.op.csi_dense(data, weight, bias, **new_dense_attrs)
                        return new_call
                    else:
                        if call.op.name == "qnn.csi.bias_add":
                            lhs_shape = _infer_shape(pre_call)
                            rhs_shape = op_args[1].checked_type.concrete_shape
                            if len(lhs_shape) == 4 and len(rhs_shape) == 1:
                                newshape = (1, -1, 1, 1)
                                rhs_data = op_args[1].data.asnumpy()
                                rhs_data = np.reshape(rhs_data, newshape)
                                rhs = relay.expr.const(rhs_data)

                                new_attrs = _qnn_attrs(call.attrs)
                                new_call = relay.qnn.op.csi_add(pre_call, rhs, **new_attrs)
                                return new_call
                new_call = Call(call.op, op_args, call.attrs, call.type_args, call.span)
                return new_call

        return FuseAddAfterConvMutator().visit(func)


@function_pass(opt_level=1)
class FuseMulAfterConv:
    """Fuse mul op in behind the convolution op."""

    def __init__(self, curr_config):
        self.config = curr_config

    def transform_function(self, func, mod, ctx):
        """patten and convert op"""
        func = relay.Function(func.params, func.body, None, func.type_params, func.attrs)
        _fuse_mul = self

        class FuseMulAfterConvMutator(relay.ExprMutator):
            """Fuse mul after conv"""

            def visit_call(self, call):
                op_args = [self.visit(arg) for arg in call.args]
                if call.op.name == "qnn.csi.mul" and isinstance(op_args[1], Constant):
                    pre_call = op_args[0]
                    if isinstance(pre_call, Call) and pre_call.op.name == "qnn.csi.conv2d":
                        new_conv2d_attrs = _qnn_attrs(pre_call.attrs)
                        data = pre_call.args[0]
                        weight = pre_call.args[1]
                        bias = pre_call.args[2]

                        weight_val = weight.data.asnumpy()
                        bias_val = bias.data.asnumpy()
                        mul_rhs_val = op_args[1].data.asnumpy()

                        if mul_rhs_val.size != weight_val.shape[0]:
                            new_call = Call(call.op, op_args, call.attrs, call.type_args, call.span)
                            return new_call

                        new_weight_val = fuse_params_mul_after_conv(weight_val, mul_rhs_val)
                        if len(bias_val.shape) != 0:
                            new_bias_val = bias_val * mul_rhs_val.reshape(-1)
                        else:
                            new_bias_val = bias_val

                        new_conv2d_attrs["q_params"][-1] = call.attrs.q_params[-1]
                        new_conv2d_attrs["q_params"][2] = get_weight_params(
                            new_bias_val, _fuse_mul.config
                        )
                        update_conv_attrs(new_weight_val, new_conv2d_attrs, _fuse_mul.config)

                        weight.data.copyfrom(new_weight_val)
                        bias = relay.expr.const(new_bias_val)
                        new_conv2d_attrs["layer_name"] += "_fuse_" + call.attrs.layer_name
                        new_call = relay.qnn.op.csi_conv2d(data, weight, bias, **new_conv2d_attrs)
                        return new_call

                new_call = Call(call.op, op_args, call.attrs, call.type_args, call.span)
                return new_call

        return FuseMulAfterConvMutator().visit(func)


def fuse_layer(mod, current_config):
    """remove unnecessary layer to speed up module.

    Returns
    -------
    ret: Function
        The module pass function.
    """

    # def wrapped_func(mod, ctx): # pylint: disable=unused-argument
    fuse_pass_sequential = [
        {FuseReshape: "default"},
        {FusePad: "default"},
        {FuseBias: "fuse_add_after_conv"},
        {FuseMulAfterConv: "fuse_mul_after_conv"},
        {FuseAddAfterConv: "fuse_add_after_conv"},
        # {FuseAddBeforeConv: "fuse_add_before_conv"},
        {FuseMulBeforeConv: "fuse_mul_before_conv"},
        {FuseClip: "fuse_clip"},
        {FuseRelu: "fuse_relu"},
        {FuseConvRelu: "fuse_conv_relu"},
        {FuseReshapeDense: "fuse_reshape_dense"},
    ]

    for mutator_map in fuse_pass_sequential:
        mutator = list(mutator_map.keys())[0]
        csinn_config = mutator_map[mutator]
        if (
            csinn_config in ["fuse_mul_before_conv", "fuse_mul_after_conv"]
            and current_config[csinn_config]
        ):
            mod = transform.Sequential([mutator(current_config)])(mod)
        elif csinn_config == "fuse_add_after_conv" and current_config[csinn_config]:
            mod = transform.Sequential([FuseBias()])(mod)
            mod = transform.Sequential([FuseAddAfterConv(current_config)])(mod)
        elif csinn_config == "default" or current_config[csinn_config]:
            mod = transform.Sequential([mutator()])(mod)

    return mod


@function_pass(opt_level=1)
class OptimizeShapeCheck:
    """Optimize shape check layer"""

    def transform_function(self, func, mod, ctx):
        """patten and convert op"""
        func = relay.Function(func.params, func.body, None, func.type_params, func.attrs)

        class OptimizeShapeCheckMutator(relay.ExprMutator):
            """optimize shape"""

            def visit_call(self, call):
                op_args = [self.visit(arg) for arg in call.args]

                if call.op.name in [
                    "qnn.csi.add",
                    "qnn.csi.mul",
                    "qnn.csi.subtract",
                    "qnn.csi.div",
                    "qnn.csi.power",
                    "qnn.csi.minimum",
                    "qnn.csi.maximum",
                ]:
                    if isinstance(op_args[1], Constant) and len(_infer_shape(op_args[1])) == 0:
                        dtype = (
                            op_args[1]._checked_type_.dtype
                            if op_args[1]._checked_type_
                            else "float32"
                        )
                        value = op_args[1].data.asnumpy().tolist()
                        op_args[1] = const(np.array([value]).astype(dtype), dtype)
                new_call = Call(call.op, op_args, call.attrs, call.type_args, call.span)
                return new_call

            def visit_function(self, fn):
                new_params = [self.visit(x) for x in fn.params]
                new_body = self.visit(fn.body)
                return function.Function(list(new_params), new_body)

        return OptimizeShapeCheckMutator().visit(func)


@function_pass(opt_level=1)
class UpdataQparams:
    """update attr for layers"""

    def __init__(self, indexd_graph):
        self.indexd_graph = indexd_graph

    def transform_function(self, func, mod, ctx):
        """patten and convert op"""
        func = relay.Function(func.params, func.body, None, func.type_params, func.attrs)
        updata_qparams = self

        class UpdataQparamsMutator(relay.ExprMutator):
            """_summary_

            Args:
                relay (_type_): _description_
            """

            def visit_call(self, call):
                op_args = [self.visit(arg) for arg in call.args]
                node_name = hash(call)
                if node_name in updata_qparams.indexd_graph:
                    attrs = updata_qparams.indexd_graph[node_name].attr
                else:
                    attrs = _qnn_attrs(call.attrs)
                new_call = _get_csi_op(call.op.name)(*op_args, **attrs)
                return new_call

        return UpdataQparamsMutator().visit(func)


@function_pass(opt_level=1)
class InsertAddBeforeConcat:
    """Optimize concat layer"""

    def __init__(self, op_list):
        self.insert_list = ["qnn.csi." + op for op in op_list]
        self.concat_input = []

    def insert_add(self, inputs, q_params):
        """insert op"""

        in_shape = _infer_shape(inputs)
        zeros = np.ones(in_shape, np.float32)
        zeros = relay.expr.const(zeros, dtype="float32")
        add_q = [1, 0, 1, 1.0, 1.0]
        new_q_params = [q_params[-1], add_q, q_params[-1]]
        return relay.qnn.op.csi_mul(inputs, zeros, new_q_params)

    def transform_function(self, func, mod, ctx):
        """patten and convert op"""
        func = relay.Function(func.params, func.body, None, func.type_params, func.attrs)
        insert_add_ = self

        class InsertAddBeforeConcatMutator(relay.ExprMutator):
            """_summary_

            Args:
                relay (_type_): _description_
            """

            def visit_call(self, call):
                op_args = [self.visit(arg) for arg in call.args]
                current_attrs = _qnn_attrs(call.attrs)
                if call.op.name == "qnn.csi.concatenate":
                    new_tuple_args = [[] for _ in op_args[0]]
                    for idx, pre_call in enumerate(op_args[0]):
                        new_tuple_args[idx] = op_args[0][idx]
                        if isinstance(pre_call, Call):
                            if pre_call.op.name in insert_add_.insert_list:
                                pre_attrs = _qnn_attrs(pre_call.attrs)
                                new_pre_call = insert_add_.insert_add(
                                    pre_call, pre_attrs["q_params"]
                                )
                                new_tuple_args[idx] = new_pre_call
                            elif pre_call in insert_add_.concat_input:
                                pre_attrs = _qnn_attrs(pre_call.attrs)
                                new_pre_call = insert_add_.insert_add(
                                    pre_call, pre_attrs["q_params"]
                                )
                                new_tuple_args[idx] = new_pre_call
                            insert_add_.concat_input.append(pre_call)

                    new_current_call = relay.qnn.op.csi_concatenate(
                        Tuple(new_tuple_args), **current_attrs
                    )
                    return new_current_call

                return Call(call.op, op_args, call.attrs, call.type_args, call.span)

        return InsertAddBeforeConcatMutator().visit(func)


def optimize_quantization(mod, broadcast_quantization=False, target=""):
    """Optimize quantization for th1520"""

    class Node:
        """Indexed node"""

        def __init__(self, name, op_name, attr, inputs):
            self.name = name
            self.op_name = op_name
            self.attr = attr
            self.call = None  # for debug
            self.inputs = inputs
            self.outputs = list()
            self.change_in = dict()
            self.change_out = dict()

        def get_input_idx(self, input_name):
            for i, j in enumerate(self.inputs):
                if j == input_name:
                    return i
            raise Exception("Can't find input!.")

    class CreateIndexedGraph(relay.ExprVisitor):
        """create indexed graph"""

        def __init__(self, mod, target):
            super(CreateIndexedGraph, self).__init__()
            self.target = target
            self.indexd_graph = dict()
            self.mod = mod
            self.need_change = set()
            self.visit(self.mod["main"])

        def visit_call(self, call):
            _ = [self.visit(arg) for arg in call.args]
            attrs = _qnn_attrs(call.attrs)
            node_name = hash(call)

            pre_layers = call.args if call.op.name != "qnn.csi.concatenate" else call.args[0]

            inputs = []
            for pre_layer in pre_layers:
                if isinstance(pre_layer, (Var, Constant)):
                    continue
                if isinstance(pre_layer, TupleGetItem):
                    hash_pre = hash(pre_layer.tuple_value)
                else:
                    hash_pre = hash(pre_layer)
                inputs.append(hash_pre)
                if hash_pre not in self.indexd_graph:
                    raise Exception("Can't find pre node.")
                in_node = self.indexd_graph[hash_pre]
                in_node.outputs.append(node_name)
            self.indexd_graph[node_name] = Node(node_name, call.op.name, attrs, inputs)
            self.indexd_graph[node_name].call = call

        def get_graph(self):
            """return indexed graph"""
            return self.indexd_graph

        def update_node_in(self, node_name, in_name, qinfo):
            node = self.indexd_graph[node_name]
            node.change_in[in_name] = qinfo
            self.need_change.add(node_name)

        def update_node_out(self, node_name, in_name, qinfo):
            node = self.indexd_graph[node_name]
            node.change_out[in_name] = qinfo
            self.need_change.add(node_name)

        def th1520_qinfo_mutator(self, in2out_list, out2in_list):
            """qinfo mutator for th1520"""
            for node_name, node in self.indexd_graph.items():
                op_name = node.op_name
                if op_name in in2out_list:
                    if node.attr["q_params"][1] == node.attr["q_params"][0]:
                        continue
                    # in to out
                    node.attr["q_params"][1] = node.attr["q_params"][0]
                    # register for all out
                    for out_name in node.outputs:
                        self.update_node_in(out_name, node.name, node.attr["q_params"][0])

                elif op_name in out2in_list:
                    if op_name == "qnn.csi.concatenate":
                        for idx, in_name in enumerate(node.inputs):
                            node.attr["q_params"][idx] = node.attr["q_params"][-1]
                            in_node = self.indexd_graph[in_name]
                            if in_node.op_name == "qnn.csi.concatenate" and self.target in (
                                "th1520",
                                "hth1520",
                            ):
                                raise Exception("concat try to modifly pre concat out!")
                            if in_node.op_name == "qnn.csi.concatenate":
                                continue
                            self.update_node_out(in_name, node.name, node.attr["q_params"][-1])
                    else:
                        if node.attr["q_params"][1] == node.attr["q_params"][0]:
                            continue
                        # out to in
                        node.attr["q_params"][0] = node.attr["q_params"][1]
                        # for ops in first layer
                        if not node.inputs:
                            continue
                        # register for all inputs
                        for in_name in node.inputs:
                            self.update_node_out(in_name, node.name, node.attr["q_params"][0])

            while self.need_change:
                node_name = self.need_change.pop()
                node = self.indexd_graph[node_name]
                in_changed = False
                out_changed = False
                if len(node.change_out) > 1:
                    if node.op_name != "qnn.csi.split":
                        raise Exception(
                            "Multiple nodes attempt to modify the current node at the same time！"
                        )
                if node.change_in:
                    for in_node_name, qinfo in node.change_in.items():
                        in_idx = node.get_input_idx(in_node_name)
                        node.attr["q_params"][in_idx] = qinfo
                    in_changed = True
                    node.change_in.clear()

                if node.change_out:
                    if node.op_name == "qnn.csi.split":
                        for out_node_name, qinfo in node.change_out.items():
                            out_node = self.indexd_graph[out_node_name]

                            out_node_ins = []
                            for arg in out_node.call.args:
                                if isinstance(arg, Tuple):
                                    for a in arg:
                                        out_node_ins.append(a)
                                else:
                                    out_node_ins.append(arg)
                            for i_out_node in out_node_ins:
                                if not isinstance(i_out_node, TupleGetItem):
                                    continue
                                split_index = i_out_node.index
                                break
                            node.attr["q_params"][1 + split_index] = qinfo

                            for out_name in node.outputs:
                                if out_node_name == out_name:
                                    continue
                                if out_node.op_name in out2in_list:
                                    continue
                                out_node = self.indexd_graph[out_name]

                                out_node_ins = []
                                for arg in out_node.call.args:
                                    if isinstance(arg, Tuple):
                                        for a in arg:
                                            out_node_ins.append(a)
                                    else:
                                        out_node_ins.append(arg)

                                need_change = False
                                for o in out_node_ins:
                                    if not isinstance(o, TupleGetItem):
                                        continue
                                    if split_index == o.index:
                                        need_change = True
                                        break
                                if need_change:
                                    self.update_node_in(
                                        out_name, node.name, node.attr["q_params"][1 + split_index]
                                    )
                        node.change_out.clear()
                        continue
                    else:
                        for _, qinfo in node.change_out.items():
                            node.attr["q_params"][-1] = qinfo
                            break

                    # updat outputs
                    for out_name in node.outputs:
                        out_node = self.indexd_graph[out_name]
                        if out_node.op_name in out2in_list:
                            continue
                        self.update_node_in(out_name, node.name, node.attr["q_params"][-1])
                        out_changed = True
                    node.change_out.clear()

                if in_changed and out_changed:
                    if node.op_name in in2out_list + out2in_list:
                        raise Exception("Input and output qinfo can't be changed at the same time.")
                if node.op_name in ["qnn.csi.concatenate"]:
                    if in_changed:
                        new_min = np.min(node.attr["q_params"])
                        new_max = np.max(node.attr["q_params"])
                        node.attr["q_params"][-1] = node.attr["q_params"][-1][:3] + [
                            new_min,
                            new_max,
                        ]
                        # updata inputs
                        for idx, in_name in enumerate(node.inputs):
                            in_node = self.indexd_graph[in_name]
                            if in_node.op_name == "qnn.csi.concatenate" and self.target in (
                                "th1520",
                                "hth1520",
                            ):
                                raise Exception("concat try to modifly pre concat out!")
                            if in_node.op_name == "qnn.csi.concatenate":
                                break
                            in_node.change_out[node.name] = node.attr["q_params"][-1]
                            node.attr["q_params"][idx] = node.attr["q_params"][-1]
                            self.need_change.add(in_name)

                        # updat outputs
                        for out_name in node.outputs:
                            self.update_node_in(out_name, node.name, node.attr["q_params"][-1])

                    if out_changed:
                        raise Exception("Concat output cannot be modified！")

                elif node.op_name in in2out_list + out2in_list:
                    if in_changed:
                        # in to out
                        if node.attr["q_params"][-1] != node.attr["q_params"][0]:
                            node.attr["q_params"][-1] = node.attr["q_params"][0]
                            for out_name in node.outputs:
                                self.update_node_in(out_name, node.name, node.attr["q_params"][-1])

                    if out_changed:
                        # out to in
                        if node.attr["q_params"][0] != node.attr["q_params"][-1]:
                            node.attr["q_params"][0] = node.attr["q_params"][-1]
                            for in_name in node.inputs:
                                self.update_node_out(in_name, node.name, node.attr["q_params"][-1])

        def cpu_qinfo_mutator(self, in2out_list, out2in_list):
            """qinfo mutator for cpus"""
            for node_name, node in self.indexd_graph.items():
                op_name = node.op_name
                if op_name in in2out_list:
                    if node.attr["q_params"][1] == node.attr["q_params"][0]:
                        continue
                    # in to out
                    node.attr["q_params"][1] = node.attr["q_params"][0]
                    # register for all out
                    for out_name in node.outputs:
                        self.update_node_in(out_name, node.name, node.attr["q_params"][0])

            while self.need_change:
                node_name = self.need_change.pop()
                node = self.indexd_graph[node_name]
                in_changed = False
                out_changed = False
                if len(node.change_out) > 1:
                    if node.op_name != "qnn.csi.split":
                        raise Exception(
                            "Multiple nodes attempt to modify the current node at the same time！"
                        )
                if node.change_in:
                    for in_node_name, qinfo in node.change_in.items():
                        in_idx = node.get_input_idx(in_node_name)
                        node.attr["q_params"][in_idx] = qinfo
                    in_changed = True
                    node.change_in.clear()

                if node.change_out:
                    if node.op_name == "qnn.csi.split":
                        for out_node_name, qinfo in node.change_out.items():
                            out_node = self.indexd_graph[out_node_name]

                            out_node_ins = []
                            for arg in out_node.call.args:
                                if isinstance(arg, Tuple):
                                    for a in arg:
                                        out_node_ins.append(a)
                                else:
                                    out_node_ins.append(arg)
                            for i_out_node in out_node_ins:
                                if not isinstance(i_out_node, TupleGetItem):
                                    continue
                                split_index = i_out_node.index
                                break
                            node.attr["q_params"][1 + split_index] = qinfo

                            for out_name in node.outputs:
                                if out_node_name == out_name:
                                    continue
                                if out_node.op_name in out2in_list:
                                    continue
                                out_node = self.indexd_graph[out_name]

                                out_node_ins = []
                                for arg in out_node.call.args:
                                    if isinstance(arg, Tuple):
                                        for a in arg:
                                            out_node_ins.append(a)
                                    else:
                                        out_node_ins.append(arg)

                                need_change = False
                                for o in out_node_ins:
                                    if not isinstance(o, TupleGetItem):
                                        continue
                                    if split_index == o.index:
                                        need_change = True
                                        break
                                if need_change:
                                    self.update_node_in(
                                        out_name, node.name, node.attr["q_params"][1 + split_index]
                                    )
                        node.change_out.clear()
                        continue
                    else:
                        for _, qinfo in node.change_out.items():
                            node.attr["q_params"][-1] = qinfo
                            break

                    # updat outputs
                    for out_name in node.outputs:
                        out_node = self.indexd_graph[out_name]
                        if out_node.op_name in out2in_list:
                            continue
                        self.update_node_in(out_name, node.name, node.attr["q_params"][-1])
                        out_changed = True
                    node.change_out.clear()

                if in_changed and out_changed:
                    if node.op_name in in2out_list + out2in_list:
                        raise Exception("Input and output qinfo can't be changed at the same time.")

                elif node.op_name in in2out_list + out2in_list:
                    if in_changed:
                        # in to out
                        if node.attr["q_params"][-1] != node.attr["q_params"][0]:
                            node.attr["q_params"][-1] = node.attr["q_params"][0]
                            for out_name in node.outputs:
                                self.update_node_in(out_name, node.name, node.attr["q_params"][-1])
                    if out_changed:
                        # out to in
                        if node.attr["q_params"][0] != node.attr["q_params"][-1]:
                            node.attr["q_params"][0] = node.attr["q_params"][-1]
                            for in_name in node.inputs:
                                self.update_node_out(in_name, node.name, node.attr["q_params"][-1])

        def qinfo_exchange(self, in2out_list, out2in_list, target):
            """change node quant params"""
            in2out_list = ["qnn.csi." + op for op in in2out_list]
            out2in_list = ["qnn.csi." + op for op in out2in_list]
            if target in ("th1520", "hth1520"):
                self.th1520_qinfo_mutator(in2out_list, out2in_list)
            else:
                self.cpu_qinfo_mutator(in2out_list, out2in_list)

    mod = transform.Sequential([OptimizeShapeCheck()])(mod)
    if broadcast_quantization:
        if target in ["th1520", "hth1520"]:
            out2in_list = ["concatenate"]
            in2out_list = [
                "reshape",
                "upsampling",
                "transpose",
                "mean",
                "relu",
                "relu6",
                "avgpool2d",
                "maxpool2d",
                "global_avgpool2d",
                "global_maxpool2d",
                "strided_slice",
            ]
            mod = transform.Sequential([InsertAddBeforeConcat(out2in_list + in2out_list)])(mod)
        else:
            out2in_list = []
            in2out_list = ["transpose", "reshape", "upsampling", "maxpool2d", "strided_slice"]
        index_graph_creater = CreateIndexedGraph(mod, target)
        index_graph_creater.qinfo_exchange(in2out_list, out2in_list, target)
        mod = transform.Sequential([UpdataQparams(index_graph_creater.get_graph())])(mod)

    return mod


def rename_call(mod, call_count):
    """Specify name for call node which has empty layer_name."""

    class RenameCall(relay.ExprMutator):
        """Helper class"""

        def __init__(self, call_count):
            super(RenameCall, self).__init__()
            self.call_count = call_count

        def visit_call(self, call):
            op_args = [self.visit(arg) for arg in call.args]
            if str(call.attrs.layer_name) == "":
                attrs = _qnn_attrs(call.attrs)
                op_name = call.op.name.split(".")[-1]
                attrs["layer_name"] = op_name + "_" + str(self.call_count)
                if call.span:
                    attrs["layer_name"] = call.span.source_name.name
                new_call = _get_csi_op(call.op.name)(*op_args, **attrs)

                self.call_count += 1
            else:
                new_call = Call(call.op, op_args, call.attrs, call.type_args, call.span)
            return new_call

    mod["main"] = RenameCall(call_count).visit(mod["main"])
    return mod


def is_valid(param):
    """param = [1, 1, 0, scale, zp], if scale==0 and zp==0
    this param is not initialized or valid.
    """
    res = False
    if tuple(param[-2:]) != (0, 0):
        res = True
    return res


@function_pass(opt_level=1)
class CurrentInputAndPreOutput:
    """Ensure the input's quant params of current op is the same with
    the output's quant params of previous op.
    """

    def transform_funciton(self, func, mod, ctx):
        """_summary_"""
        func = relay.Function(func.params, func.body, None, func.type_params, func.attrs)

        class CurrentInputAndPreOutputMutator(relay.ExprMutator):
            """_summary_

            Args:
                relay (_type_): _description_
            """

            def visit_call(self, call):
                op_args = [self.visit(arg) for arg in call.args]
                new_op_attrs = _qnn_attrs(call.attrs)

                for i, arg in enumerate(op_args):
                    if isinstance(arg, Call):
                        arg_attrs = _qnn_attrs(arg.attrs)

                        if is_valid(arg_attrs["q_params"][-1]):
                            new_op_attrs["q_params"][i] = arg_attrs["q_params"][-1]
                        elif is_valid(new_op_attrs["q_params"][i]):
                            arg_attrs["q_params"][-1] = new_op_attrs["q_params"][i]
                            op_args[i] = csi_op().all_handle[arg.op.name](*(arg.args), **arg_attrs)
                    # (Fixme@chenf): TupleGetItem and Tuple node's params should be fixed.
                    elif isinstance(arg, TupleGetItem):
                        arg_attrs = _qnn_attrs(arg.tuple_value.attrs)

                        in_num = len(arg.tuple_value.args)
                        if tuple(arg_attrs["q_params"][in_num + arg.index][-2:]) != (0, 0):
                            new_op_attrs["q_params"][i] = arg_attrs["q_params"][in_num + arg.index]
                    elif isinstance(arg, Tuple):
                        for j in range(len(arg)):
                            curr_node = arg.field[j]
                            if isinstance(curr_node, Call):
                                curr_attrs = _qnn_attrs(curr_node.attrs)
                                if tuple(curr_attrs["q_params"][-1][-2:]) != (0, 0):
                                    new_op_attrs["q_params"][j] = curr_attrs["q_params"][-1]
                            elif isinstance(curr_node, TupleGetItem):
                                curr_attrs = _qnn_attrs(curr_node.attrs)

                                in_num = len(arg.tuple_value.args)
                                if tuple(curr_attrs["q_params"][in_num + curr_node.index][-2:]) != (
                                    0,
                                    0,
                                ):
                                    new_op_attrs["q_params"][j] = curr_attrs["q_params"][
                                        in_num + curr_node.index
                                    ]
                return csi_op().all_handle[call.op.name](*op_args, **new_op_attrs)

        return CurrentInputAndPreOutputMutator().visit(func)


@function_pass(opt_level=1)
class CurrentInputAndCurrentOutput:
    """Ensure the input's quant params of current op is the same with
    the output's quant params of current op."""

    def transform_funciton(self, func, mod, ctx):
        """_summary_"""
        func = relay.Function(func.params, func.body, None, func.type_params, func.attrs)

        class CurrentInputAndCurrentOutputMutator(relay.ExprMutator):
            """_summary_

            Args:
                relay (_type_): _description_
            """

            def visit_call(self, call):
                op_args = [self.visit(arg) for arg in call.args]
                new_op_attrs = _qnn_attrs(call.attrs)

                out2in_list = [
                    "qnn.csi.reshape",
                    "qnn.csi.upsampling",
                    "qnn.csi.transpose",
                    "qnn.csi.mean",
                    "qnn.csi.relu",
                    "qnn.csi.relu6",
                    "qnn.csi.avgpool2d",
                    "qnn.csi.maxpool2d",
                    "qnn.csi.global_avgpool2d",
                    "qnn.csi.global_maxpool2d",
                    "qnn.csi.strided_slice",
                ]

                if call.op.name in out2in_list:
                    if (
                        is_valid(new_op_attrs["q_params"][-1])
                        and is_valid(new_op_attrs["q_params"][0])
                        and tuple(new_op_attrs["q_params"][-1][-2:])
                        != tuple(new_op_attrs["q_params"][0][-2:])
                    ):
                        logger.warning(
                            "%s:%s has different quant info for input:%s/output:%s"
                            % (
                                call.op.name,
                                new_op_attrs["layer_name"],
                                tuple(new_op_attrs["q_params"][0][-2:]),
                                tuple(new_op_attrs["q_params"][-1][-2:]),
                            )
                        )
                    if is_valid(new_op_attrs["q_params"][-1]) and (
                        not is_valid(new_op_attrs["q_params"][0])
                    ):
                        new_op_attrs["q_params"][0] = new_op_attrs["q_params"][-1]
                        logger.warning(
                            "The quant info of output is copied to input in %s:%s"
                            % (call.op.name, new_op_attrs["layer_name"])
                        )
                    elif is_valid(new_op_attrs["q_params"][0]) and (
                        not is_valid(new_op_attrs["q_params"][-1])
                    ):
                        new_op_attrs["q_params"][-1] = new_op_attrs["q_params"][0]
                        logger.warning(
                            "The quant info of input is copied to output in %s:%s"
                            % (call.op.name, new_op_attrs["layer_name"])
                        )

                return csi_op().all_handle[call.op.name](*op_args, **new_op_attrs)

        return CurrentInputAndCurrentOutputMutator().visit(func)


def unify_quant_params(mod):
    old_level = logger.getEffectiveLevel()
    logger.setLevel(logging.ERROR)
    mod = transform.Sequential([CurrentInputAndCurrentOutput()])(mod)
    logger.setLevel(old_level)
    mod = transform.Sequential([CurrentInputAndPreOutput()])(mod)
    mod = transform.Sequential([CurrentInputAndCurrentOutput()])(mod)
    mod = transform.Sequential([CurrentInputAndPreOutput()])(mod)
    return mod
