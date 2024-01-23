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
# pylint: disable=missing-docstring, unused-import
# pylint: disable=too-many-nested-blocks, no-else-continue
"""Optimization passess for qnn."""
import math
import numpy as np

import tvm
from tvm import relay
from tvm.ir import transform
from tvm.relay.transform import function_pass
from tvm.relay.expr import Var, Call, Constant, Tuple, TupleGetItem
from tvm.relay.expr import RelayExpr
from tvm.relay.dataflow_pattern import DFPatternCallback, is_constant, wildcard, is_op, rewrite
from tvm.relay.frontend.common import infer_shape as _infer_shape

from ..ir.base import _qnn_attrs, _get_csi_op, csi_op
from ..quantization.calibrate import (
    get_weight_params,
    CONST,
    ACTIVATION,
    PER_CHANNEL,
    PER_TENSOR,
    USE_SCALE,
)
from ..quantization.spec import is_invalid_q_params


@function_pass(opt_level=1)
class Conv2dSqueezeAdd:
    r"""fusion pass for qnn

        Input
          |
    qnn.csin.conv2d        qnn.csin.conv2d
          |           -->        |
    qnn.csi.reshape        qnn.csi.reshape
          |
     qnn.csi.add

    """

    def transform_function(self, func, mod, ctx):
        """patten and convert op"""

        class MyCallback(DFPatternCallback):
            def __init__(self):
                super(MyCallback, self).__init__()
                # input
                self.input = wildcard()
                # conv2d
                self.weight_val = is_constant()
                self.bias = is_constant()
                self.conv2d = is_op("qnn.csi.conv2d")(self.input, self.weight_val, self.bias)
                # squeeze
                self.squeeze = is_op("qnn.csi.reshape")(self.conv2d)
                # bias_add
                self.bias_val = is_constant()
                self.bias_add = is_op("qnn.csi.add")(self.squeeze, self.bias_val)

                self.pattern = self.bias_add

            def callback(self, pre, post, node_map):
                """taget op"""
                in_node = node_map[self.input][0]
                weight = node_map[self.weight_val][0]
                conv2d = node_map[self.conv2d][0]
                old_bias = node_map[self.bias][0]
                bias = node_map[self.bias_val][0]
                squeeze = node_map[self.squeeze][0]

                old_b_val = old_bias.data.asnumpy()
                bias_val = bias.data.asnumpy()
                b_shape = bias_val.shape
                bias_size = b_shape[0] if len(b_shape) == 1 else b_shape
                conv_attrs = _qnn_attrs(conv2d.attrs)
                bias_attrs = _qnn_attrs(bias.attrs)
                squeeze_attrs = _qnn_attrs(squeeze.attrs)
                conv_attrs["q_params"][-1] = bias_attrs["q_params"][-1]
                squeeze_attrs["q_params"][0] = bias_attrs["q_params"][-1]
                squeeze_attrs["q_params"][-1] = bias_attrs["q_params"][-1]
                if bias_size == conv_attrs["channels"]:
                    new_bias = bias if not old_b_val else relay.const(bias_val + old_b_val)
                    new_conv2d = relay.qnn.op.csi_conv2d(in_node, weight, new_bias, **conv_attrs)
                    new_node = relay.qnn.op.csi_reshape(new_conv2d, **squeeze_attrs)
                else:
                    new_node = bias

                return new_node

        out = rewrite(MyCallback(), mod["main"].body)
        res = tvm.IRModule.from_expr(out)

        return res["main"]


@function_pass(opt_level=1)
class FuseWhereSoftmax:
    r"""fusion pass for qnn

    Input
      |
    where    -> where_softmax
      |
    softmax

    """

    def transform_function(self, func, mod, ctx):
        """patten and convert op"""

        class MyCallback(DFPatternCallback):
            def __init__(self):
                super(MyCallback, self).__init__()
                # input
                self.conditoin = wildcard()
                self.x = is_constant()
                self.y = wildcard()
                # where
                self.where = is_op("qnn.csi.where")(self.conditoin, self.x, self.y)
                # softmax
                self.softmax = is_op("qnn.csi.softmax")(self.where)
                self.pattern = self.softmax

            def callback(self, pre, post, node_map):
                """taget op"""
                conditoin = node_map[self.conditoin][0]
                x = node_map[self.x][0]
                y = node_map[self.y][0]
                x_data = x.data.asnumpy()
                if len(x_data.shape) != 0 or x_data != -np.Inf:
                    # TODO : fix x is constant but not just a number
                    raise Exception(f"where softmax need single number.")

                where_attrs = _qnn_attrs(node_map[self.where][0].attrs)
                softmax_attrs = _qnn_attrs(node_map[self.softmax][0].attrs)

                where_attrs["minus_inf"] = float(x_data)
                where_attrs["axis"] = softmax_attrs["axis"]
                where_attrs["q_params"][-1] = softmax_attrs["q_params"][-1]

                new_node = relay.qnn.op.csi_where_softmax(conditoin, x, y, **where_attrs)
                return new_node

        out = rewrite(MyCallback(), mod["main"].body)
        res = tvm.IRModule.from_expr(out)

        return res["main"]


@function_pass(opt_level=1)
class Resume4DimsMatMul:
    r"""fusion reshapes in MatMul to be 4 dims matmul

    Input0      Input1
      |           |
    reshape0   reshape1
       \       /
        MatMul          -->   MatMul
         |
      reshape2

    """

    def transform_function(self, func, mod, ctx):
        """patten and convert op"""

        class MyCallback(DFPatternCallback):
            def __init__(self):
                super(MyCallback, self).__init__()
                # input0
                self.input0 = wildcard()
                # input1
                self.input1 = wildcard()
                self.bias = is_constant()
                # reshape0
                self.reshape0 = is_op("qnn.csi.reshape")(self.input0)
                # reshape1
                self.reshape1 = is_op("qnn.csi.reshape")(self.input1)
                # MatMul
                self.matmul = is_op("qnn.csi.matmul")(self.reshape0, self.reshape1, self.bias)
                # reshape1
                self.reshape2 = is_op("qnn.csi.reshape")(self.matmul)
                self.pattern = self.reshape2

            def callback(self, pre, post, node_map):
                """taget op"""
                in_node0 = node_map[self.input0][0]
                in_node1 = node_map[self.input1][0]
                bias = relay.expr.const(0, dtype="float32")
                matmul_attrs = _qnn_attrs(node_map[self.matmul][0].attrs)
                new_node = relay.qnn.op.csi_matmul(in_node0, in_node1, bias, **matmul_attrs)
                reshape2 = node_map[self.reshape2][0]
                reshape_attrs = _qnn_attrs(reshape2.attrs)
                out_new_shape = reshape_attrs["newshape"]
                if len(out_new_shape) not in [3, 4]:
                    new_node = relay.qnn.op.csi_reshape(new_node, **reshape_attrs)

                in_shape0 = _infer_shape(in_node0)
                in_shape1 = _infer_shape(in_node1)
                if len(in_shape0) == len(in_shape1):
                    return new_node
                reshape0 = node_map[self.reshape0][0]
                reshape1 = node_map[self.reshape1][0]

                out_shape = _infer_shape(node_map[self.matmul][0])

                in0 = (
                    in_node0
                    if len(in_shape0) == len(out_shape) and in_shape0[:2] == out_shape[:2]
                    else reshape0
                )
                in1 = in_node1 if len(in_shape1) == len(out_shape) else reshape1
                new_node = relay.qnn.op.csi_matmul(in0, in1, bias, **matmul_attrs)
                new_node = relay.qnn.op.csi_reshape(new_node, **reshape_attrs)
                return new_node

        out = rewrite(MyCallback(), mod["main"].body)
        res = tvm.IRModule.from_expr(out)

        return res["main"]


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


def get_quant_value(data):
    """Extract quantization info values."""
    data_shape = data.shape
    if len(data_shape) == 0 or (len(data_shape) == 1 and data_shape[0] == 1):
        # per-tensor quantization
        data = data.tolist()
        if isinstance(data, (tuple, list)):
            data = data[0]
    else:
        raise NotImplementedError("Detect multi values, per-channel quantization is not supported.")
    return data


@function_pass(opt_level=1)
class FuseActivateQuantInfo:
    r"""Extract quant info from quantize/dequantize ops and fuse them into previous op.

      op
      |
    quantize    ->   op with output quantization info (scale, zero_point)
      |
    dequantize

    """

    def transform_function(self, func, mod, ctx):
        """patten and convert op."""

        class MyCallback(DFPatternCallback):
            def __init__(self):
                super(MyCallback, self).__init__()

                # any call
                self.call_patten = wildcard()(None)

                # quantize op
                self.scale1 = is_constant()
                self.zp1 = is_constant()
                self.quantize = is_op("qnn.csi.quantize")(self.call_patten, self.scale1, self.zp1)

                # dequantize op
                self.scale2 = is_constant()
                self.zp2 = is_constant()
                self.dequantize = is_op("qnn.csi.dequantize")(self.quantize, self.scale2, self.zp2)

                self.pattern = self.dequantize

            def callback(
                self, pre: RelayExpr, post: RelayExpr, node_map: tvm.ir.container.Map
            ) -> RelayExpr:
                call_node = node_map[self.call_patten][0]
                dequantize_node = node_map[self.dequantize][0]
                scale1_node = node_map[self.scale1][0]
                zp1_node = node_map[self.zp1][0]

                scale1_val = scale1_node.data.numpy()
                scale1_val = get_quant_value(scale1_val)
                zp1_val = zp1_node.data.numpy()
                zp1_val = get_quant_value(zp1_val)

                call_attrs = _qnn_attrs(call_node.attrs)
                dequantize_attrs = _qnn_attrs(dequantize_node.attrs)
                # modify output quant params of call_node
                call_attrs["q_params"][-1] = [1, 1, 0, scale1_val, zp1_val]

                call_attrs["layer_name"] = (
                    "fuse_" + call_attrs["layer_name"] + "_" + dequantize_attrs["layer_name"]
                )

                new_node = csi_op().all_handle[call_node.op.name](*call_node.args, **call_attrs)

                return new_node

        out = rewrite(MyCallback(), mod["main"].body)
        res = tvm.IRModule.from_expr(out)

        return res["main"]


@function_pass(opt_level=1)
class FuseInputQuantInfo:
    r"""Extract quant info of input node from quantize/dequantize ops
         and fuse them into subsequent op.

    input          input
      |              |
    quantize   ->   op with input quantization info (scale, zero_point)
      |
    dequantize
      |
     op

    """

    def transform_function(self, func, mod, ctx):
        func = relay.Function(func.params, func.body, None, func.type_params, func.attrs)

        class FuseInputQuantInfoMutator(relay.ExprMutator):
            """Internal helper class"""

            def visit_call(self, call):
                op_args = [self.visit(arg) for arg in call.args]
                new_op_attrs = _qnn_attrs(call.attrs)
                new_args = list(op_args)

                for i, arg in enumerate(op_args):
                    if isinstance(arg, Call):
                        if arg.op.name == "qnn.csi.dequantize":
                            quant_node = arg.args[0]
                            if (
                                quant_node
                                and isinstance(quant_node, Call)
                                and quant_node.op.name == "qnn.csi.quantize"
                            ):
                                var_node = quant_node.args[0]
                                if var_node and isinstance(var_node, Var):
                                    scale_val = arg.args[1].data.numpy()
                                    scale_val = get_quant_value(scale_val)
                                    zp_val = arg.args[2].data.numpy()
                                    zp_val = get_quant_value(zp_val)

                                    new_op_attrs["q_params"][i] = [1, 1, 0, scale_val, zp_val]
                                    new_args[i] = var_node
                    elif isinstance(arg, Tuple):
                        new_tuple = []
                        for j in range(len(arg)):
                            dequant_node = arg.field[j]
                            if (
                                dequant_node
                                and isinstance(dequant_node, Call)
                                and dequant_node.op.name == "qnn.csi.dequantize"
                            ):
                                quant_node = dequant_node.args[0]
                                if (
                                    quant_node
                                    and isinstance(quant_node, Call)
                                    and quant_node.op.name == "qnn.csi.quantize"
                                ):
                                    var_node = quant_node.args[0]
                                    if var_node and isinstance(var_node, Var):
                                        scale_val = dequant_node.args[1].data.numpy()
                                        scale_val = get_quant_value(scale_val)
                                        zp_val = dequant_node.args[2].data.numpy()
                                        zp_val = get_quant_value(zp_val)

                                        new_op_attrs["q_params"][j] = [1, 1, 0, scale_val, zp_val]

                                        new_tuple.append(var_node)
                                        continue
                            new_tuple.append(dequant_node)
                        new_args[i] = Tuple(new_tuple)
                return csi_op().all_handle[call.op.name](*new_args, **new_op_attrs)

        return FuseInputQuantInfoMutator().visit(func)


@function_pass(opt_level=1)
class FuseDequantizeOp:
    r"""Fuse dequantize into op.

    dequantize
      |
      op         ->   op

    """

    def transform_function(self, func, mod, ctx):
        func = relay.Function(func.params, func.body, None, func.type_params, func.attrs)

        class FuseDequantizeOpMutator(relay.ExprMutator):
            """Internal helper class"""

            def visit_call(self, call):
                op_args = [self.visit(arg) for arg in call.args]
                new_op_attrs = _qnn_attrs(call.attrs)
                new_args = list(op_args)

                for i, arg in enumerate(op_args):
                    if isinstance(arg, Call):
                        if arg.op.name == "qnn.csi.dequantize":
                            const_node = arg.args[0]
                            if const_node and isinstance(const_node, Constant):
                                scale_val = arg.args[1].data.numpy()
                                scale_val = get_quant_value(scale_val)
                                zp_value = arg.args[2].data.numpy()
                                zp_value = get_quant_value(zp_value)

                                new_op_attrs["q_params"][i] = [1, 1, 0, scale_val, zp_value]
                                new_args[i] = const_node
                    elif isinstance(arg, Tuple):
                        new_tuple = []
                        for j in range(len(arg)):
                            dequant_node = arg.field[j]
                            if (
                                dequant_node
                                and isinstance(dequant_node, Call)
                                and dequant_node.op.name == "qnn.csi.dequantize"
                            ):
                                const_node = dequant_node.args[0]
                                if const_node and isinstance(const_node, Constant):
                                    scale_val = dequant_node.args[1].data.numpy()
                                    scale_val = get_quant_value(scale_val)
                                    zp_value = dequant_node.args[2].data.numpy()
                                    zp_value = get_quant_value(zp_value)

                                    new_op_attrs["q_params"][j] = [1, 1, 0, scale_val, zp_value]
                                    continue
                            new_tuple.append(dequant_node)
                        new_args[i] = Tuple(new_tuple)
                return csi_op().all_handle[call.op.name](*new_args, **new_op_attrs)

        return FuseDequantizeOpMutator().visit(func)


class QNNNodeMap(object):
    """Convert original qnn module into node map that holds keys information."""

    class Node(object):
        """Internal class holds node info."""

        def __init__(self) -> None:
            # [(call_hash, index), ...]
            self.ins = []
            self.in_q_params = []
            self.out_q_params = []
            self.hash_value = None
            self.call_name = None

        def __eq__(self, __value: object) -> bool:
            return self.hash_value == __value.hash_value

        def __hash__(self) -> int:
            return self.hash_value

    def __init__(self) -> None:
        self.nodes = []
        self.hash2nodes = {}

    def create_empty_node(self) -> Node:
        return self.Node()

    def find_out_node(self, call_hash, index):
        """Get the specified output node of current call."""
        for node in self.nodes:
            for hash_index in node.ins:
                if (call_hash, index) == hash_index:
                    return node
        return None

    def create_map_from_module(self, mod):
        """Create node map with specified qnn module."""
        class_obj = self

        class CreateHashMap(relay.ExprVisitor):
            """Convert QNN ir into hash map"""

            def visit_call(self, call):
                _ = [self.visit(arg) for arg in call.args]
                call_attrs = _qnn_attrs(call.attrs)
                qnn_map_node = class_obj.create_empty_node()
                qnn_map_node.hash_value = hash(call)
                qnn_map_node.call_name = call.op.name

                in_num = 0
                for i, arg in enumerate(call.args):
                    if isinstance(arg, Tuple):
                        in_num += len(arg)
                        for a in arg:
                            if isinstance(a, TupleGetItem):
                                qnn_map_node.ins.append((hash(a.tuple_value), a.index))
                            else:
                                qnn_map_node.ins.append((hash(a), 0))
                    elif isinstance(arg, TupleGetItem):
                        qnn_map_node.ins.append((hash(arg.tuple_value), arg.index))
                        in_num += 1
                    else:
                        qnn_map_node.ins.append((hash(arg), 0))
                        in_num += 1

                for i, q_param in enumerate(call_attrs["q_params"]):
                    true_value = q_param if not is_invalid_q_params(q_param) else None
                    if i < in_num:
                        qnn_map_node.in_q_params.append(true_value)
                    else:
                        qnn_map_node.out_q_params.append(true_value)

                class_obj.nodes.append(qnn_map_node)
                class_obj.hash2nodes[qnn_map_node.hash_value] = qnn_map_node

        chm = CreateHashMap()
        chm.visit(mod["main"])


@function_pass(opt_level=1)
class QNNFuseQDQ:
    """Fuse QDQ nodes into qnn ir.

    .. code-block:: text

        input -> quantize -> dequantize -> qnn_layer1 -> quantize -> dequantize -> qnn_layer2 ->
        quantize -> dequantize -> output

    Would become:

    .. code-block:: text

        input -> qnn_layer1 -> qnn_layer2 -> output

    """

    def __init__(self, config) -> None:
        self.config = config

    def transform_function(self, func, mod, ctx):
        """Helper function to convert qnn ir."""
        func = relay.Function(func.params, func.body, None, func.type_params, func.attrs)

        def _get_qdq_params(scale, zp, tensor_type):
            """Get the quantization params that meet the requirements of qnn."""
            q_param = [tensor_type, USE_SCALE]

            assert (
                scale.size == zp.size
            ), f"Mismatch size between scale:{scale.size} and zero_point:{zp.size}"
            if scale.size == 1:
                q_param += [PER_TENSOR]
            else:
                q_param += [PER_CHANNEL]
            scale = scale.tolist()
            zp = zp.tolist()
            if isinstance(scale, (tuple, list)):
                for s_zp in zip(scale, zp):
                    q_param += list(s_zp)
            else:
                q_param += [scale, zp]
            return q_param

        class FuseActivatiionQDQ(DFPatternCallback):
            r"""Extract quant info from quantize/dequantize ops and fuse them into previous op.

            op
            |
            quantize    ->   op with output quantization info (scale, zero_point)
            |
            dequantize

            """

            def __init__(self):
                super(FuseActivatiionQDQ, self).__init__()

                # any call
                self.call_patten = wildcard()(None)

                # quantize op
                self.scale1 = is_constant()
                self.zp1 = is_constant()
                self.quantize = is_op("qnn.csi.quantize")(self.call_patten, self.scale1, self.zp1)

                # dequantize op
                self.scale2 = is_constant()
                self.zp2 = is_constant()
                self.dequantize = is_op("qnn.csi.dequantize")(self.quantize, self.scale2, self.zp2)

                self.pattern = self.dequantize

            def callback(
                self, pre: RelayExpr, post: RelayExpr, node_map: tvm.ir.container.Map
            ) -> RelayExpr:
                call_node = node_map[self.call_patten][0]
                scale1_node = node_map[self.scale1][0]
                zp1_node = node_map[self.zp1][0]

                scale1_val = scale1_node.data.numpy()
                zp1_val = zp1_node.data.numpy()

                call_attrs = _qnn_attrs(call_node.attrs)
                # modify output quant params of call_node
                call_attrs["q_params"][-1] = _get_qdq_params(scale1_val, zp1_val, ACTIVATION)

                new_node = csi_op().all_handle[call_node.op.name](*call_node.args, **call_attrs)

                return new_node

        class FuseQDQActivation(relay.ExprMutator):
            r"""Extract quant info of input node from quantize/dequantize ops
                and fuse them into subsequent op.

            input          input
            |              |
            quantize   ->   op with input quantization info (scale, zero_point)
            |
            dequantize
            |
            op

            """

            def visit_call(self, call):
                op_args = [self.visit(arg) for arg in call.args]
                new_op_attrs = _qnn_attrs(call.attrs)
                new_args = list(op_args)

                for i, arg in enumerate(op_args):
                    if isinstance(arg, Call):
                        if arg.op.name == "qnn.csi.dequantize":
                            quant_node = arg.args[0]
                            if (
                                quant_node
                                and isinstance(quant_node, Call)
                                and quant_node.op.name == "qnn.csi.quantize"
                            ):
                                pre_node = quant_node.args[0]
                                scale_val = arg.args[1].data.numpy()
                                zp_val = arg.args[2].data.numpy()

                                new_op_attrs["q_params"][i] = _get_qdq_params(
                                    scale_val, zp_val, ACTIVATION
                                )
                                new_args[i] = pre_node
                    elif isinstance(arg, Tuple):
                        new_tuple = []
                        for j in range(len(arg)):
                            dequant_node = arg.fields[j]
                            if (
                                dequant_node
                                and isinstance(dequant_node, Call)
                                and dequant_node.op.name == "qnn.csi.dequantize"
                            ):
                                quant_node = dequant_node.args[0]
                                if (
                                    quant_node
                                    and isinstance(quant_node, Call)
                                    and quant_node.op.name == "qnn.csi.quantize"
                                ):
                                    pre_node = quant_node.args[0]
                                    scale_val = dequant_node.args[1].data.numpy()
                                    zp_val = dequant_node.args[2].data.numpy()

                                    new_op_attrs["q_params"][i + j] = _get_qdq_params(
                                        scale_val, zp_val, ACTIVATION
                                    )
                                    new_tuple.append(pre_node)
                                    continue
                            new_tuple.append(dequant_node)
                        new_args[i] = Tuple(new_tuple)
                return csi_op().all_handle[call.op.name](*new_args, **new_op_attrs)

        class FuseDequantize(relay.ExprMutator):
            r"""Fuse dequantize into op.

            dequantize
            |
            op         ->   op

            """

            def visit_call(self, call):
                op_args = [self.visit(arg) for arg in call.args]
                new_op_attrs = _qnn_attrs(call.attrs)
                new_args = list(op_args)

                for i, arg in enumerate(op_args):
                    if isinstance(arg, Call):
                        if arg.op.name == "qnn.csi.dequantize":
                            const_node = arg.args[0]
                            if const_node and isinstance(const_node, Constant):
                                scale_val = arg.args[1].data.numpy()
                                zp_value = arg.args[2].data.numpy()

                                new_op_attrs["q_params"][i] = _get_qdq_params(
                                    scale_val, zp_value, CONST
                                )
                                new_args[i] = const_node
                    elif isinstance(arg, Tuple):
                        new_tuple = []
                        for j in range(len(arg)):
                            dequant_node = arg.fields[j]
                            if (
                                dequant_node
                                and isinstance(dequant_node, Call)
                                and dequant_node.op.name == "qnn.csi.dequantize"
                            ):
                                const_node = dequant_node.args[0]
                                if const_node and isinstance(const_node, Constant):
                                    scale_val = dequant_node.args[1].data.numpy()
                                    zp_value = dequant_node.args[2].data.numpy()

                                    new_op_attrs["q_params"][i + j] = _get_qdq_params(
                                        scale_val, zp_value, CONST
                                    )
                                    continue
                            new_tuple.append(dequant_node)
                        new_args[i] = Tuple(new_tuple)
                return csi_op().all_handle[call.op.name](*new_args, **new_op_attrs)

        class AlignCurrentInputAndPreOutput(relay.ExprMutator):
            """Ensure the input's quant params of current op is the same with
            the output's quant params of previous op.
            """

            def __init__(self, qnn_map: QNNNodeMap):
                super().__init__()

                self.qnn_map = qnn_map

            def visit_call(self, call):
                op_args = [self.visit(arg) for arg in call.args]
                call_attrs = _qnn_attrs(call.attrs)

                call_node = self.qnn_map.hash2nodes[hash(call)]

                # deal with input quantization params
                new_in_q_params = []
                for i, in_q in enumerate(call_node.in_q_params):
                    if in_q is not None:
                        new_in_q_params.append(in_q)
                        continue
                    in_hash, out_idx = call_node.ins[i]
                    if in_hash not in self.qnn_map.hash2nodes:
                        new_in_q_params.append(call_attrs["q_params"][i])
                        continue
                    in_node = self.qnn_map.hash2nodes[in_hash]
                    if in_node.out_q_params[out_idx] is None:
                        new_in_q_params.append(call_attrs["q_params"][i])
                    else:
                        new_in_q_params.append(in_node.out_q_params[out_idx])

                # deal with output quantization params
                new_out_q_params = []
                for i, out_q in enumerate(call_node.out_q_params):
                    if out_q is not None:
                        new_out_q_params.append(out_q)
                        continue
                    out_node = self.qnn_map.find_out_node(call_node.hash_value, i)
                    if out_node:
                        for in_idx, (in_out_hash, _) in enumerate(out_node.ins):
                            if in_out_hash == call_node.hash_value:
                                new_out_q_params.append(out_node.in_q_params[in_idx])
                    else:
                        in_num = len(new_in_q_params)
                        new_out_q_params.append(call_attrs["q_params"][in_num + i])

                # create new call
                call_attrs["q_params"] = new_in_q_params + new_out_q_params
                return csi_op().all_handle[call.op.name](*op_args, **call_attrs)

        out = rewrite(FuseActivatiionQDQ(), mod["main"].body)
        res = tvm.IRModule.from_expr(out)

        res["main"] = FuseQDQActivation().visit(res["main"])
        res["main"] = FuseDequantize().visit(res["main"])

        res = fuse_layer(res, self.config)

        qnn_map = QNNNodeMap()
        qnn_map.create_map_from_module(res)

        res["main"] = AlignCurrentInputAndPreOutput(qnn_map).visit(res["main"])
        return res["main"]


@function_pass(opt_level=1)
class QNNFuseConvDepthtospace:
    """Fuse conv2d+depth2space into deconv."""

    def transform_function(self, func, mod, ctx):
        """Helper function to convert qnn ir."""
        func = relay.Function(func.params, func.body, None, func.type_params, func.attrs)

        class InterHelper(DFPatternCallback):
            """Helper class"""

            def __init__(self, require_type=False, rewrite_once=False):
                super().__init__(require_type, rewrite_once)

                self.input = wildcard()
                self.weight = is_constant()
                self.bias = is_constant()
                self.conv = is_op("qnn.csi.conv2d")(self.input, self.weight, self.bias).has_attr(
                    {"strides": [1, 1], "groups": 1}
                )
                self.d2s = is_op("qnn.csi.depth_to_space")(self.conv)

                self.pattern = self.d2s

            def callback(self, pre, post, node_map) -> RelayExpr:
                in_call = node_map[self.input][0]

                weight = node_map[self.weight][0]
                bias = node_map[self.bias][0]
                conv_call = node_map[self.conv][0]

                d2s_call = node_map[self.d2s][0]

                conv_attrs = _qnn_attrs(conv_call.attrs)
                d2s_attrs = _qnn_attrs(d2s_call.attrs)
                block_size = d2s_attrs["block_size"]

                hk, wk = conv_attrs["kernel_size"]
                deconv_kernel_size = [hk * block_size, wk * block_size]
                deconv_strides = [block_size, block_size]
                deconv_pad = [
                    (hk - 1 - conv_attrs["padding"][2]) * block_size,
                    (wk - 1 - conv_attrs["padding"][3]) * block_size,
                    (hk - 1 - conv_attrs["padding"][0]) * block_size,
                    (wk - 1 - conv_attrs["padding"][1]) * block_size,
                ]

                Fk, Pk, Hk, Wk = _infer_shape(weight)
                Fk_deconv = Fk // (block_size * block_size)
                Pk_deconv = Pk
                Hk_deconv = Hk * block_size
                Wk_deconv = Wk * block_size
                weight_data = weight.data.numpy()
                # flip weight in x, y
                flipped_weight_data = np.zeros(weight_data.shape, dtype=weight_data.dtype)
                for f in range(Fk):
                    for c in range(Pk):
                        for h in range(Hk):
                            for w in range(Wk):
                                flipped_weight_data[f, c, h, w] = weight_data[
                                    f, c, Hk - 1 - h, Wk - 1 - w
                                ]

                # interleave the weight
                interleaved_weight_data = np.zeros(weight_data.shape, dtype=weight_data.dtype)
                for f in range(Fk):
                    for c in range(Pk):
                        for h in range(Hk):
                            for w in range(Wk):
                                idx = (f % Fk_deconv) * block_size * block_size + f // Fk_deconv
                                interleaved_weight_data[idx, c, h, w] = flipped_weight_data[
                                    f, c, h, w
                                ]

                #  combine weight into deconv weight
                deconv_weight_data_oihw = np.zeros(
                    (Fk_deconv, Pk_deconv, Hk_deconv, Wk_deconv), dtype=weight_data.dtype
                )
                for f in range(Fk):
                    for c in range(Pk):
                        for h in range(Hk):
                            for w in range(Wk):
                                deconv_weight_data_oihw[
                                    f // (block_size * block_size),
                                    c,
                                    h * block_size + f // block_size % block_size,
                                    w * block_size + f % block_size,
                                ] = interleaved_weight_data[f, c, h, w]

                deconv_weight_data_iohw = np.transpose(deconv_weight_data_oihw, (1, 0, 2, 3))
                deconv_weight = relay.const(deconv_weight_data_iohw)

                bias_data = bias.data.numpy()
                if isinstance(bias_data.tolist(), float) and math.isclose(bias_data.tolist(), 0.0):
                    deconv_bias = bias
                else:
                    interleaved_bias_data = np.zeros(bias_data.shape, bias_data.dtype)
                    for f in range(Fk):
                        idx = (f % Fk_deconv) * (block_size * block_size) + f // Fk_deconv
                        interleaved_bias_data[idx] = bias_data[f]
                    deconv_bias = relay.const(interleaved_bias_data)

                deconv_call = relay.qnn.op.csi_deconv2d(
                    in_call,
                    deconv_weight,
                    deconv_bias,
                    strides=deconv_strides,
                    padding=deconv_pad,
                    dilation=(1, 1),
                    groups=1,
                    channels=Fk_deconv,
                    kernel_size=deconv_kernel_size,
                    data_layout="NCHW",
                    kernel_layout="IOHW",
                    out_layout="",
                    output_padding=(0, 0),
                    out_dtype="float32",
                    q_params=conv_attrs["q_params"],
                    layer_name="deconv_" + conv_attrs["layer_name"],
                )
                return deconv_call

        out = rewrite(InterHelper(), mod["main"].body)
        res = tvm.IRModule.from_expr(out)

        return res["main"]
