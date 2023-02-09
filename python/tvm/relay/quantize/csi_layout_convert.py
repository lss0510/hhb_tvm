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
# pylint: disable=invalid-name, unused-argument, not-callable
"""Convert csinn model layout."""
import logging
import numpy as np

from tvm import relay, IRModule
from ..frontend.common import infer_shape
from ._convert_to_csi import _qnn_attrs, _get_csi_op
from ..expr import Constant, Tuple
from ..transform import function_pass
from ..dataflow_pattern import wildcard, is_op, DFPatternCallback, rewrite
from .. import function as _function

logger = logging.getLogger("HHB")

NCHW2NHWC_FUNCS = {}


def nchw2nhwc_attrs_changer(attrs):
    """Change layout attributes"""

    attrs = _qnn_attrs(attrs)
    if "data_layout" in attrs:
        attrs["data_layout"] = "NHWC"
    if "out_layout" in attrs:
        attrs["out_layout"] = "NHWC"
    if "kernel_layout" in attrs:
        attrs["kernel_layout"] = "OHWI"
    if "layout" in attrs:
        attrs["layout"] = "NHWC"
    return attrs


def nchw2nhwc_func_register(func_name):
    """Register func in NCHW2NHWC_FUNCS"""

    def decorator(func):
        NCHW2NHWC_FUNCS[func_name] = func.__name__

        def wrapper(self, call, op_args):
            attrs = nchw2nhwc_attrs_changer(call.attrs)
            return func(self, op_args, attrs)

        return wrapper

    return decorator


def nchw_to_nhwc(mod):
    """Convert layout from NCHW to NHWC"""

    class NCHW2NHWCMutaor(relay.ExprMutator):
        """Convert layout from NCHW to NHWC"""

        def list_convert(self, src_list):
            if len(src_list) == 4:
                return [src_list[i] for i in [0, 2, 3, 1]]
            return src_list

        def axis_convert(self, axis):
            convert_axis = [0, 3, 1, 2]
            return convert_axis[axis]

        def constant_convert(self, src_constat, is_depthwise=False):
            """Convert constant value layout"""
            if isinstance(src_constat, Constant):
                np_value = src_constat.data.asnumpy()
                value_rank = len(np_value.shape)
                if value_rank == 4:
                    if is_depthwise:
                        np_value = np_value.transpose([1, 2, 3, 0])
                    else:
                        np_value = np_value.transpose([0, 2, 3, 1])

                return relay.const(np_value, str(np_value.dtype))
            return src_constat

        def visit_call(self, call):
            op_args = [self.visit(arg) for arg in call.args]
            if call.op.name in NCHW2NHWC_FUNCS:
                func = getattr(self, NCHW2NHWC_FUNCS[call.op.name])
                new_call = func(call=call, op_args=op_args)
            else:
                attrs = nchw2nhwc_attrs_changer(call.attrs)
                func = _get_csi_op(call.op.name)
                new_call = func(*op_args, **attrs)

            return new_call

        def visit_var(self, var):
            shape = list(var.checked_type.concrete_shape)
            new_shape = self.list_convert(shape)
            dtype = var.checked_type.dtype
            name = var.name_hint
            return relay.var(name, shape=new_shape, dtype=dtype)

        def diso_convert(self, op_args, attrs, op_name):
            op_args[1] = self.constant_convert(op_args[1])
            func = _get_csi_op("qnn.csi." + op_name)
            return func(*op_args, **attrs)

        @nchw2nhwc_func_register("qnn.csi.conv2d")
        def conv2d(self, op_args, attrs):
            """convert conv2d layout"""
            dshape = infer_shape(op_args[0])
            wshape = infer_shape(op_args[1])
            is_depthwise = False
            if attrs["groups"] != 1 and attrs["groups"] == dshape[3] == wshape[0]:
                is_depthwise = True
            op_args[1] = self.constant_convert(op_args[1], is_depthwise)
            return relay.qnn.op.csi_conv2d(*op_args, **attrs)

        @nchw2nhwc_func_register("qnn.csi.conv2d_relu")
        def conv2d_relu(self, op_args, attrs):
            """convert conv2d_relu layout"""
            dshape = infer_shape(op_args[0])
            wshape = infer_shape(op_args[1])
            is_depthwise = False
            if attrs["groups"] != 1 and attrs["groups"] == dshape[3] == wshape[0]:
                is_depthwise = True
            op_args[1] = self.constant_convert(op_args[1], is_depthwise)
            return relay.qnn.op.csi_conv2d_relu(*op_args, **attrs)

        @nchw2nhwc_func_register("qnn.csi.reshape")
        def reshape(self, op_args, attrs):
            """convert reshape layout"""
            in_shape_rank = len(infer_shape(op_args[0]))
            newshape_rank = len(attrs["newshape"])
            if in_shape_rank == 4 and newshape_rank != 4:
                axes = [0, 3, 1, 2]
                out_dtype = attrs["out_dtype"]
                q_params = attrs["q_params"]
                layer_name = attrs["layer_name"]
                op_args[0] = relay.qnn.op.csi_transpose(
                    op_args[0], axes, out_dtype, q_params, layer_name
                )
            attrs["newshape"] = self.list_convert(attrs["newshape"])
            return relay.qnn.op.csi_reshape(*op_args, **attrs)

        @nchw2nhwc_func_register("qnn.csi.depth_to_space")
        def depth_to_space(self, op_args, attrs):
            """convert depth_to_space layout"""
            attrs["layout"] = "NHWC"
            return relay.qnn.op.csi_depth_to_space(*op_args, **attrs)

        @nchw2nhwc_func_register("qnn.csi.softmax")
        def softmax(self, op_args, attrs):
            """convert softmax layout"""
            in_expr = op_args[0]
            in_shape_rank = len(infer_shape(in_expr))
            if in_shape_rank == 4:
                attrs["axis"] = self.axis_convert(attrs["axis"])
            return relay.qnn.op.csi_softmax(*op_args, **attrs)

        @nchw2nhwc_func_register("qnn.csi.squeeze")
        def squeeze(self, op_args, attrs):
            """convert squeeze layout"""
            in_expr = op_args[0]
            in_shape_rank = len(infer_shape(in_expr))
            if in_shape_rank == 4:
                new_axis = []
                for i in attrs["axis"]:
                    new_axis.append(self.axis_convert(int(i)))
                attrs["axis"] = new_axis
            return relay.qnn.op.csi_squeeze(*op_args, **attrs)

        # DISO
        @nchw2nhwc_func_register("qnn.csi.subtract")
        def subtract(self, op_args, attrs):
            """convert subtract layout"""
            return self.diso_convert(op_args, attrs, "subtract")

        @nchw2nhwc_func_register("qnn.csi.mul")
        def mul(self, op_args, attrs):
            """convert mul layout"""
            return self.diso_convert(op_args, attrs, "mul")

        @nchw2nhwc_func_register("qnn.csi.add")
        def add(self, op_args, attrs):
            """convert add layout"""
            return self.diso_convert(op_args, attrs, "add")

        @nchw2nhwc_func_register("qnn.csi.div")
        def div(self, op_args, attrs):
            """convert div layout"""
            return self.diso_convert(op_args, attrs, "div")

        @nchw2nhwc_func_register("qnn.csi.minimum")
        def minimum(self, op_args, attrs):
            """convert minimum layout"""
            return self.diso_convert(op_args, attrs, "minimum")

        @nchw2nhwc_func_register("qnn.csi.split")
        def split(self, op_args, attrs):
            """convert split layout"""
            in_rank = len(infer_shape(op_args[0]))
            if in_rank == 4:
                attrs["axis"] = self.axis_convert(attrs["axis"])
            return relay.qnn.op.csi_split(op_args[0], **attrs)

        @nchw2nhwc_func_register("qnn.csi.concatenate")
        def concatenate(self, op_args, attrs):
            """convert concatenate layout"""

            in_rank = len(infer_shape(op_args[0].fields[0]))
            new_args = []
            for arg in op_args[0]:
                new_args.append(self.constant_convert(arg))
            if in_rank == 4:
                attrs["axis"] = self.axis_convert(attrs["axis"])
            return relay.qnn.op.csi_concatenate(Tuple(new_args), **attrs)

        def visit_function(self, fn):
            new_params = [self.visit(x) for x in fn.params]
            new_body = self.visit(fn.body)
            return _function.Function(list(new_params), new_body)

    convert = NCHW2NHWCMutaor()
    mod["main"] = convert.visit(mod["main"])
    return mod


ALIGN_FUNCS = {}


def align_func_register(func_name):
    """Register func in ALIGN_FUNCS"""

    def decorator(func):
        ALIGN_FUNCS[func_name] = func.__name__

        def wrapper(self, call, op_args, old_shape):
            attrs = _qnn_attrs(call.attrs)
            return func(self, op_args, attrs, old_shape)

        return wrapper

    return decorator


def align_to_shape(mod, align):
    """weight shape alignment"""

    class ShapeAlignMutaor(relay.ExprMutator):
        """weight shape alignment"""

        def __init__(self, align):
            super(ShapeAlignMutaor, self).__init__()
            self.align = align

        def fill_tensor(self, src_data, shape, axis):
            fill_data = np.zeros(shape).astype(np.float32)
            return np.concatenate([src_data, fill_data], axis=axis)

        def revert_shape(self, data, length, q_param, l_name, dtype, axis=1):
            """revert shape to origin"""
            index_expr = relay.const(list(range(length)))
            index_params = [q_param] * 3
            layer_name = l_name + "_take"
            ret = relay.qnn.op.csi_take(
                data,
                index_expr,
                axis=axis,
                out_dtype=dtype,
                q_params=index_params,
                mode="clip",
                layer_name=layer_name,
            )
            return ret

        def constant_convert(self, weight, bias, is_depthwise=False, need_fill=True):
            """Convert constant value layout"""
            np_weight = weight.data.asnumpy()
            np_bias = bias.data.asnumpy()
            fill_bias = np.prod(np_bias.shape) > 1
            k_o, k_i, k_h, k_w = np_weight.shape

            if is_depthwise:
                if need_fill:
                    np_weight = self.fill_tensor(np_weight, [need_fill, k_i, k_h, k_w], 0)
                    if fill_bias:
                        np_bias = self.fill_tensor(np_bias, [need_fill], 0)
            else:
                o_fill = self.align - k_o % self.align if k_o % self.align != 0 else 0

                if o_fill:
                    np_weight = self.fill_tensor(np_weight, [o_fill, k_i, k_h, k_w], 0)
                    if fill_bias:
                        np_bias = self.fill_tensor(np_bias, [o_fill], 0)

                if need_fill:
                    shape = list(np_weight.shape)
                    shape[1] = need_fill
                    np_weight = self.fill_tensor(np_weight, shape, 1)
                    if fill_bias:
                        np_bias = self.fill_tensor(np_bias, [need_fill], 0)

            new_weight = relay.const(np_weight, str(np_weight.dtype))
            new_bias = relay.const(np_bias, str(np_bias.dtype))
            return new_weight, new_bias

        def visit_call(self, call):
            op_args = [self.visit(arg) for arg in call.args]
            if call.op.name in ALIGN_FUNCS:
                func = getattr(self, ALIGN_FUNCS[call.op.name])
                old_shape = infer_shape(call.args[0])

                new_call = func(call=call, op_args=op_args, old_shape=old_shape)
            else:
                attrs = _qnn_attrs(call.attrs)
                func = _get_csi_op(call.op.name)
                new_call = func(*op_args, **attrs)

            return new_call

        @align_func_register("qnn.csi.conv2d")
        def conv2d(self, op_args, attrs, old_shape):
            """convert conv2d weight layout"""
            dshape = infer_shape(op_args[0])
            wshape = infer_shape(op_args[1])
            is_depthwise = False
            if attrs["groups"] != 1 and attrs["groups"] == old_shape[1] == wshape[0]:
                is_depthwise = True

            need_fill = dshape[1] - old_shape[1]

            if attrs["groups"] > 1 and not is_depthwise:
                if need_fill:
                    op_args[0] = self.revert_shape(
                        op_args[0],
                        old_shape[1],
                        attrs["q_params"][0],
                        attrs["layer_name"],
                        attrs["out_dtype"],
                    )
                logger.debug(
                    "aligned %s shape: in_shape %s, w_shape: %s",
                    attrs["layer_name"],
                    dshape,
                    wshape,
                )

                new_call = relay.qnn.op.csi_conv2d(*op_args, **attrs)
                return new_call

            op_args[1:] = self.constant_convert(op_args[1], op_args[2], is_depthwise, need_fill)
            attrs["channels"] = infer_shape(op_args[1])[0]
            if is_depthwise:
                attrs["groups"] = attrs["channels"]
            new_call = relay.qnn.op.csi_conv2d(*op_args, **attrs)
            logger.debug(
                "aligned %s shape: in_shape %s, w_shape: %s",
                attrs["layer_name"],
                dshape,
                op_args[1].data.asnumpy().shape,
            )

            return new_call

        @align_func_register("qnn.csi.softmax")
        def softmax(self, op_args, attrs, old_shape):
            """convert softmax layout"""
            dshape = infer_shape(op_args[0])
            if dshape != old_shape:
                op_args[0] = self.revert_shape(
                    op_args[0],
                    old_shape[1],
                    attrs["q_params"][0],
                    attrs["layer_name"],
                    attrs["out_dtype"],
                )

            return relay.qnn.op.csi_softmax(*op_args, **attrs)

        def visit_function(self, fn):
            new_params = [self.visit(x) for x in fn.params]
            new_body = self.visit(fn.body)
            return _function.Function(list(new_params), new_body)

    convert = ShapeAlignMutaor(align)
    mod["main"] = convert.visit(mod["main"])
    return mod


@function_pass(opt_level=1)
class FuseTRDense:
    r"""
      Input
        |              Input
    Transpose            |
        |       -->   Reshape
     Reshape             |
        |              Dense
      Dense

    """

    def transform_function(self, func, mod, ctx):
        """patten and convert op"""

        class MyCallback(DFPatternCallback):
            """patten and convert op"""

            def __init__(self):
                super(MyCallback, self).__init__()
                self.input = wildcard()
                # transpose
                self.t = is_op("qnn.csi.transpose")(self.input)
                # reshape
                self.r = is_op("qnn.csi.reshape")(self.t)
                # dense
                self.weight = wildcard()
                self.b = wildcard()
                self.dense = is_op("qnn.csi.dense")(self.r, self.weight, self.b)
                self.pattern = self.dense

            def callback(self, pre, post, node_map):
                """taget op"""

                in_node = node_map[self.input][0]
                weight = node_map[self.weight][0].data.numpy()
                bias = node_map[self.b][0]

                in_shape = infer_shape(node_map[self.t][0])
                reshape_attr = _qnn_attrs(node_map[self.r][0].attrs)
                dense_attr = _qnn_attrs(node_map[self.dense][0].attrs)

                # check
                transpose_attr = _qnn_attrs(node_map[self.t][0].attrs)
                trans_axes = transpose_attr["axes"]
                if trans_axes != [0, 3, 1, 2]:
                    raise Exception("dense fuse error!")

                w_shape = weight.shape
                new_weight = weight.reshape([-1, *in_shape])
                # to nhwc
                new_weight = new_weight.transpose([0, 1, 3, 4, 2])
                # to dense
                new_weight = new_weight.reshape(w_shape)

                new_reshape = relay.qnn.op.csi_reshape(in_node, **reshape_attr)
                new_node = relay.qnn.op.csi_dense(
                    new_reshape, relay.const(new_weight), bias, **dense_attr
                )

                return new_node

        out = rewrite(MyCallback(), mod["main"].body)
        res = IRModule.from_expr(out)

        return res["main"]


def csi_layout_convert(mod, src_layout="NCHW", dest_layout="NHWC", align=1):
    """layout convert"""
    if align > 1:
        mod = align_to_shape(mod, align)

    if src_layout == "NCHW" and dest_layout == "NHWC":
        mod = nchw_to_nhwc(mod)
        mod = FuseTRDense()(mod)

    return mod
