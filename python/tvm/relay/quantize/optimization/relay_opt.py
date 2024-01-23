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
"""Custom relay pass."""
import tvm
from tvm import relay
from tvm.relay.dataflow_pattern import (
    DFPatternCallback,
    is_constant,
    is_var,
    wildcard,
    is_op,
    rewrite,
    is_tuple,
)
from tvm.relay.frontend.common import infer_shape
from tvm.relay.transform import function_pass
from tvm.relay import expr as _expr


def conv2python(data):
    return [conv2python(x) if isinstance(x, tvm.ir.container.Array) else int(x) for x in data]


def InsertNOp(mod):
    """insert Nop"""

    class BetweenLekayReLUAndAdd(relay.ExprMutator):
        """insert Nop between leakyrelu and and"""

        def visit_call(self, call):
            op_args = [self.visit(arg) for arg in call.args]
            if call.op.name == "add":
                l_pre_call = op_args[0]
                r_pre_call = op_args[1]

                if isinstance(l_pre_call, _expr.Call) and l_pre_call.op.name == "nn.leaky_relu":
                    mul_call = relay.op.add(l_pre_call, relay.op.const([2.0], "float32"))
                    new_call = relay.op.add(mul_call, r_pre_call)
                    new_call = relay.op.add(new_call, relay.op.const([-2.0], "float32"))
                    new_call = _expr.Call(
                        new_call.op, new_call.args, new_call.attrs, new_call.type_args, call.span
                    )
                    return new_call
            new_call = _expr.Call(call.op, op_args, call.attrs, call.type_args, call.span)
            return new_call

    mod["main"] = BetweenLekayReLUAndAdd().visit(mod["main"])

    return mod


def InsertRelu(mod):
    """insert relu"""

    class BetweenSigmoidAndMul(relay.ExprMutator):
        """insert relu between simoid and mul"""

        def visit_call(self, call):
            op_args = [self.visit(arg) for arg in call.args]
            if call.op.name == "multiply":
                new_pre_list = []
                for pre in op_args:
                    if isinstance(pre, _expr.Call) and pre.op.name == "sigmoid":
                        new_call = relay.op.nn.relu(pre)
                        new_pre_list.append(new_call)
                    else:
                        new_pre_list.append(pre)
                new_call = _expr.Call(call.op, new_pre_list, call.attrs, call.type_args, call.span)
                return new_call
            new_call = _expr.Call(call.op, op_args, call.attrs, call.type_args, call.span)
            return new_call

    mod["main"] = BetweenSigmoidAndMul().visit(mod["main"])

    return mod


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

                # reshape optition
                self.reshape = is_op("reshape")(self.div)

                # mul
                self.mul_val = is_constant()
                self.mul = is_op("multiply")(self.div, self.mul_val) | is_op("multiply")(
                    self.reshape, self.mul_val
                )

                # add2
                self.add2_val = is_constant()
                self.add2 = is_op("add")(self.mul, self.add2_val)

                self.pattern = self.add2

            def callback(self, pre, post, node_map):
                """taget op"""
                in_node = node_map[self.input][0]
                axis = int(node_map[self.mean1][0].attrs.axis[0])
                eps = node_map[self.add1_val][0].data.asnumpy().reshape(-1)[0]
                gamma = node_map[self.mul_val][0]
                beta = node_map[self.add2_val][0]

                new_node = relay.op.nn.layer_norm(in_node, gamma, beta, axis, eps)
                new_shape = infer_shape(new_node)
                old_shape = infer_shape(pre)
                if new_shape == old_shape:
                    return new_node
                else:
                    return relay.op.reshape(new_node, old_shape)

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
