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
# pylint: disable=invalid-name, wildcard-import, unused-wildcard-import
"""base function for qnn."""
import tvm
from tvm import relay
from tvm.relay.expr import Call


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
