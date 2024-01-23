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

import tvm
from tvm import relay
from tvm.relay import function, analysis
from tvm.relay.expr import Var, Call, TupleGetItem, Constant, Tuple
from tvm.relay.frontend.common import infer_shape as _infer_shape

from ..quantization.calibrate import CONST, ACTIVATION, PER_TENSOR, USE_MINMAX, USE_SCALE


LOG = 25
logger = logging.getLogger("HHB")


def get_layer_name(call, layer_index):
    layer_name = call.op.name.split(".")[-1]
    if call.span:
        layer_name = layer_name + "_" + call.span.source_name.name
    layer_name = layer_name + "_" + layer_index
    return layer_name


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
