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
"""calibrate function for quantization."""
import logging
import numpy as np
from tqdm import tqdm

import tvm
from tvm import relay
from tvm.ir import IRModule
from tvm.relay.expr import Var, Call, TupleGetItem, Constant, Tuple, const

from .algorithm import _find_scale_by_asy_kl, _find_scale_by_kl

# type
CONST = 0
ACTIVATION = 1

# q mode
PER_TENSOR = 0
PER_CHANNEL = 1

# value type
USE_MINMAX = 0
USE_SCALE = 1


LOG = 25
logger = logging.getLogger("HHB")


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
