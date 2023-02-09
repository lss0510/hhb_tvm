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
# pylint: disable=invalid-name, unused-argument, missing-docstring, unused-import
# pylint: disable=import-outside-toplevel
# pylint: disable=too-many-nested-blocks
"""Custom fusion pass."""
import tvm
import numpy as np
from tvm import relay
from tvm.relay.expr import RelayExpr as Expr
from tvm.relay.expr import Call, Var, Tuple, Constant
from ..dataflow_pattern import DFPatternCallback
from ..dataflow_pattern import is_constant, is_var
from ..dataflow_pattern import wildcard, is_op
from ..dataflow_pattern import rewrite, is_tuple
from ..frontend.common import infer_shape
from ..transform import function_pass
from ._convert_to_csi import _qnn_attrs, csi_op


def conv2python(data):
    return [conv2python(x) if isinstance(x, tvm.ir.container.Array) else int(x) for x in data]


@function_pass(opt_level=1)
class FuseCacheMatMul:
    r"""
    (Cache)
    Gather   Other
       \      /
        Concat
          |
        MatMUl               --> CacheMatMul
          |
         Add
          |
       Reshape
          |
       Transpose
    """

    def transform_function(self, func, mod, ctx):
        """patten and convert op"""

        class MyCallback(DFPatternCallback):
            def __init__(self):
                super(MyCallback, self).__init__()
                # Gathe
                self.input = wildcard()
                # concat
                self.concat = is_op("concatenate")(self.input)
                # Matmul
                self.weight = wildcard()
                self.dense = is_op("nn.dense")(self.concat, self.weight)
                self.b = wildcard()
                self.reshape2 = is_op("reshape")(self.dense)
                self.add = is_op("add")(self.reshape2, self.b)
                self.reshape3 = is_op("reshape")(self.add)
                # transpose
                self.transpose = is_op("transpose")(self.reshape3)
                self.pattern = self.transpose

            def callback(self, pre, post, node_map):
                """taget op"""
                cache, in_node = node_map[self.input][0]
                weight = node_map[self.weight][0]
                bias = node_map[self.b][0]
                t_dims = conv2python(node_map[self.transpose][0].attrs.axes)

                cache_shape = infer_shape(cache)
                reshape = infer_shape(node_map[self.reshape3][0])

                new_node = relay.op.custom_op.cache_matmul(
                    in_node, weight, bias, cache_shape, reshape, t_dims
                )
                return new_node

        out = rewrite(MyCallback(), mod["main"].body)
        res = tvm.IRModule.from_expr(out)

        return res["main"]


@function_pass(opt_level=1)
class FuseCacheConv1d:
    r"""
    (Cache)    Input
      |          |
    Gather   Transpose
       \        /
         Concat               --> CacheConv1d
           |
         Conv1d
           |
        BiasAdd
    """

    def transform_function(self, func, mod, ctx):
        """patten and convert op"""

        class MyCallback(DFPatternCallback):
            def __init__(self):
                super(MyCallback, self).__init__()
                # Input
                self.input = wildcard()
                # Gather
                self.gather = is_op("take")(is_var(), wildcard())
                # Transpose
                self.transpose = is_op("transpose")(self.input)
                # Concat
                self.tup = is_tuple([self.gather, self.transpose])
                self.concat = is_op("concatenate")(self.tup)
                # Conv1d
                self.weight = wildcard()
                self.conv1d = is_op("nn.conv1d")(self.concat, self.weight)
                # BiasAdd
                self.bias = wildcard()
                self.bias_add = is_op("nn.bias_add")(self.conv1d, self.bias)
                self.pattern = self.bias_add

            def callback(self, pre, post, node_map):
                """taget op"""
                in_node = node_map[self.input][0]
                weight = node_map[self.weight][0]
                bias = node_map[self.bias][0]
                cache_shape = infer_shape(node_map[self.gather][0])
                new_node = relay.op.custom_op.cache_conv1d(in_node, weight, bias, cache_shape)
                return new_node

        out = rewrite(MyCallback(), mod["main"].body)
        res = tvm.IRModule.from_expr(out)

        return res["main"]


@function_pass(opt_level=1)
class FuseLayerNormal:
    r"""
        input
       /     \
      |     Mean
       \     /
         Sub
       /     \
      |      Power
              |
      |      Mean
              |
      |      Add               --> LayNormal
              |
      |      Sqrt
       \     /
         Div
          |
         Mul
          |
         Add

    """

    def transform_function(self, func, mod, ctx):
        """patten and convert op"""

        class MyCallback(DFPatternCallback):
            def __init__(self):
                super(MyCallback, self).__init__()
                # input
                self.input = wildcard()
                # mean1
                self.mean1 = is_op("mean")(self.input)
                # sub
                self.sub = is_op("subtract")(self.input, self.mean1)
                # power
                self.power_val = is_constant()
                self.power = is_op("power")(self.sub, self.power_val)
                # mean2
                self.mean2 = is_op("mean")(self.power)
                # add1
                self.add1_val = is_constant()
                self.add1 = is_op("add")(self.mean2, self.add1_val)
                # sqrt
                self.sqrt = is_op("sqrt")(self.add1)
                # div
                self.div = is_op("divide")(self.sub, self.sqrt)
                # mul
                self.mul_val = is_constant()
                self.mul = is_op("multiply")(self.div, self.mul_val)
                # add2
                self.add2_val = is_constant()
                self.add2 = is_op("add")(self.mul, self.add2_val)

                self.pattern = self.add2

            def callback(self, pre, post, node_map):
                """taget op"""
                in_node = node_map[self.input][0]
                axis = int(node_map[self.mean1][0].attrs.axis[0])
                eps = node_map[self.add1_val][0].data.asnumpy().tolist()
                gamma = node_map[self.mul_val][0]
                beta = node_map[self.add2_val][0]

                new_node = relay.op.nn.layer_norm(in_node, gamma, beta, axis, eps)
                return new_node

        out = rewrite(MyCallback(), mod["main"].body)
        res = tvm.IRModule.from_expr(out)

        return res["main"]


@function_pass(opt_level=1)
class TConv1dAddT:
    r"""
      Input
        |
    Transpose           Dense
        |           -->   |
      Conv1D           BiasAdd
        |
     BiasAdd
        |
    Transpose

    """

    def transform_function(self, func, mod, ctx):
        """patten and convert op"""

        class MyCallback(DFPatternCallback):
            def __init__(self):
                super(MyCallback, self).__init__()
                # input
                self.input = wildcard()
                # transpose1
                self.transpose1 = is_op("transpose")(self.input)
                # conv1d
                self.weight_val = is_constant()
                self.bias_val = is_constant()
                self.conv1d = is_op("nn.conv1d")(self.transpose1, self.weight_val).has_attr(
                    {"kernel_size": [1], "groups": 1, "strides": [1], "padding": [0, 0]}
                )
                self.bias_add = is_op("nn.bias_add")(self.conv1d, self.bias_val)
                # transpose2
                self.transpose2 = is_op("transpose")(self.bias_add)
                self.pattern = self.transpose2

            def callback(self, pre, post, node_map):
                """taget op"""
                in_node = node_map[self.input][0]
                in_shape = infer_shape(in_node)
                if len(in_shape) != 2:
                    in_node = relay.op.reshape(in_node, [-1, in_shape[-1]])
                weight = node_map[self.weight_val][0].data.asnumpy().squeeze(2)
                weight_exp = relay.const(weight)
                bias = node_map[self.bias_val][0]
                new_dense = relay.op.nn.dense(in_node, weight_exp)
                new_out = relay.op.nn.bias_add(new_dense, bias, axis=-1)

                return new_out

        out = rewrite(MyCallback(), mod["main"].body)
        res = tvm.IRModule.from_expr(out)

        return res["main"]


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
class Swish:
    r"""fusion pass for qnn

        Input
       /     \
      |   Sigmoid    -->   Swish
       \     /
         Mul

    """

    def transform_function(self, func, mod, ctx):
        """patten and convert op"""

        class MyCallback(DFPatternCallback):
            def __init__(self):
                super(MyCallback, self).__init__()
                # input
                self.input = wildcard()
                # sigmoid
                self.sigmoid = is_op("sigmoid")(self.input)
                # mul
                self.mul = is_op("multiply")(self.input, self.sigmoid) | is_op("multiply")(
                    self.sigmoid, self.input
                )
                self.pattern = self.mul

            def callback(self, pre, post, node_map):
                """taget op"""
                in_node = node_map[self.input][0]
                new_node = relay.op.nn.swish(in_node)
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
                return new_node

        out = rewrite(MyCallback(), mod["main"].body)
        res = tvm.IRModule.from_expr(out)

        return res["main"]


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

            def callback(self, pre: Expr, post: Expr, node_map: tvm.ir.container.Map) -> Expr:
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


def fuse_input_quant_info(mod):
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

    class InterHelper(relay.ExprMutator):
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

    mod["main"] = InterHelper().visit(mod["main"])
    return mod


def fuse_dequantize_op(mod):
    r"""Fuse dequantize into op.

    dequantize
      |
      op         ->   op

    """

    class InterHelper(relay.ExprMutator):
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

    mod["main"] = InterHelper().visit(mod["main"])
    return mod
