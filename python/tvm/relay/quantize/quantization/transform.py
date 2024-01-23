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
# pylint: disable=not-callable, import-outside-toplevel, unused-argument
# pylint: disable=unused-argument, invalid-name, too-many-nested-blocks
"""quantization passess for qnn."""
import math
import numpy

import tvm
from tvm import relay
from tvm.relay.transform import function_pass
from tvm.relay.frontend.common import infer_shape, infer_type
from tvm.relay.backend.contrib.csinn_backend import QnnConfig, QuantCalculator
from tvm.relay.expr import RelayExpr as Expr
from tvm.relay.expr import Call, Var, Tuple, Constant, TupleGetItem
from tvm.relay.dataflow_pattern import (
    DFPatternCallback,
    wildcard,
    rewrite,
)

from ..ir.base import _qnn_attrs, csi_op
from .calibrate import USE_MINMAX, _find_abs_minmax


@function_pass(opt_level=1)
class QNNConvertReshapeToFlatten:
    """Convert reshape into flatten.

    .. code-block:: text

        input(n, 3, 2, 2) -> reshape(n, 12) -> output(n, 12)

    Or

    .. code-block:: text

        input(n, 3, 2, 2) -> reshape(n, -1) -> output(n, 12)

    Would become:

    .. code-block:: text

        input(n, 3, 2, 2) -> flatten -> output(n, 12)

    """

    def transform_function(self, func, mod, ctx):
        """Helper function to convert qnn ir."""
        func = relay.Function(func.params, func.body, None, func.type_params, func.attrs)

        class InterHelper(relay.ExprMutator):
            """Helper class"""

            def visit_call(self, call):
                op_args = [self.visit(arg) for arg in call.args]
                call_attrs = _qnn_attrs(call.attrs)

                if call.op.name == "qnn.csi.reshape":
                    in_shape = infer_shape(op_args[0])
                    newshape = call_attrs["newshape"]
                    if len(newshape) == 2 and in_shape[0] == newshape[0]:
                        new_call = relay.qnn.op.csi_flatten(
                            op_args[0],
                            out_dtype=call_attrs["out_dtype"],
                            q_params=call_attrs["q_params"],
                            layer_name=call_attrs["layer_name"],
                        )
                        return new_call
                return csi_op().all_handle[call.op.name](*op_args, **call_attrs)

        return InterHelper().visit(func)


@function_pass(opt_level=1)
class InsertQDQToQNN:
    """Insert QDQ nodes into qnn ir.

    .. code-block:: text

        input -> qnn_layer1 -> qnn_layer2 -> output

    Would become:

    .. code-block

        input -> quantize -> dequantize -> qnn_layer1 -> quantize -> dequantize -> qnn_layer2 ->
        quantize -> dequantize -> output

    """

    def __init__(self) -> None:
        self.multi_output_ops = ["qnn.csi.split"]
        self.dtype_map = {
            "int4_t": "int4",
            "int8_t": "int8",
            "uint8_t": "uint8",
            "int16_t": "int16",
            "float": "float32",
            "int32_t": "int32",
            "float16": "float16",
        }
        self.ignored_ops = [
            "qnn.csi.quantize",
            "qnn.csi.dequantize",
            "qnn.csi.cast",
            "qnn.csi.equal",
            "qnn.csi.left_shift",
            "qnn.csi.right_shift",
            "qnn.csi.where",
            "qnn.csi.less",
        ] + self.multi_output_ops
        self.all_qnn_ops = list(csi_op().all_handle.keys())

    def transform_function(self, func, mod, ctx):
        """Helper function to convert qnn ir."""
        func = relay.Function(func.params, func.body, None, func.type_params, func.attrs)
        class_obj = self
        q_config = QnnConfig()
        qc = QuantCalculator(q_config)

        def _get_scale_zp_value(cal_qinfo):
            scale = []
            zp = []
            for i in range(cal_qinfo.q_size):
                scale.append(float(cal_qinfo.qinfo[i].scale))
                zp.append(cal_qinfo.qinfo[i].zero_point)
            if cal_qinfo.q_size == 1:
                scale = scale[0]
                zp = zp[0]
            return scale, zp

        def _create_scale_zp_node(scale, zp):
            scale_node = relay.const(scale, dtype="float32")
            zp_node = relay.const(zp, dtype="int32")
            return scale_node, zp_node

        def _generate_dequantize_const_node(origin_data, origin_q_params):
            const_quant_info = qc.get_quant_params(origin_q_params, "output")
            const_quantized_data = qc.quantize_weight(origin_data, const_quant_info, False)
            new_const_node = relay.const(const_quantized_data)
            const_scale_value, const_zp_value = _get_scale_zp_value(const_quant_info)
            const_scale_node, const_zp_node = _create_scale_zp_node(
                const_scale_value, const_zp_value
            )
            dequantized_call = relay.qnn.op.csi_dequantize(
                new_const_node,
                const_scale_node,
                const_zp_node,
                axis=1,
                out_dtype="float32",
                q_params=[],
            )
            return dequantized_call

        def _inset_qdq_nodes(input_node, scale_node, zp_node, out_dtype, axis=1):
            quantize_call = relay.qnn.op.csi_quantize(
                input_node,
                scale_node,
                zp_node,
                axis=axis,
                out_dtype=out_dtype,
                q_params=[],
            )
            dequantize_call = relay.qnn.op.csi_dequantize(
                quantize_call,
                scale_node,
                zp_node,
                axis=axis,
                out_dtype="float32",
                q_params=[],
            )
            return dequantize_call

        def _is_depthwise_conv(in_shape, kernel_shape, group, layout):
            res = False
            if layout == "NCHW" and kernel_shape[1] == 1 and group == in_shape[1]:
                res = True
            elif layout == "NHWC" and kernel_shape[0] == 1 and group == in_shape[3]:
                res = True
            return res

        class InsertQDQAfterSingleOutputOp(DFPatternCallback):
            """Insert quantize/dequantize after the op that holds only an output."""

            def __init__(self):
                super(InsertQDQAfterSingleOutputOp, self).__init__()
                self.op = wildcard()(None)
                self.pattern = self.op

            def callback(self, pre: Expr, post: Expr, node_map: tvm.ir.container.Map) -> Expr:
                op_call = node_map[self.op][0]

                if isinstance(op_call, Call):
                    call_attrs = _qnn_attrs(op_call.attrs)
                    # avoid to endless loop
                    if (
                        op_call.op.name not in class_obj.ignored_ops
                        and "quantized_" not in call_attrs["layer_name"]
                    ):
                        call_attrs["layer_name"] = "quantized_" + call_attrs["layer_name"]
                        call_type = infer_type(op_call)
                        if call_type.checked_type.dtype not in ("float32", "float64", "float16"):
                            # can not insert qdq nodes
                            return csi_op().all_handle[op_call.op.name](*op_call.args, **call_attrs)

                        output_q_params = call_attrs["q_params"][-1]

                        if output_q_params[1] == USE_MINMAX:
                            output_q_info = qc.get_quant_params(output_q_params, "output")
                            scale_value, zp_value = _get_scale_zp_value(output_q_info)
                        else:
                            # scale mode: qnn ir has been quantized
                            scale_value = output_q_params[3::2]
                            zp_value = output_q_params[4::2]
                            if len(scale_value) == 1:
                                scale_value = scale_value[0]
                                zp_value = zp_value[0]
                        scale_node, zp_node = _create_scale_zp_node(scale_value, zp_value)

                        new_op_call = csi_op().all_handle[op_call.op.name](
                            *op_call.args, **call_attrs
                        )

                        if len(call_type.checked_type.shape) >= 2:
                            axis = 1
                        else:
                            axis = 0
                        dequantize_call = _inset_qdq_nodes(
                            new_op_call,
                            scale_node,
                            zp_node,
                            class_obj.dtype_map[q_config.dtype_weight],
                            axis=axis,
                        )
                        return dequantize_call
                return csi_op().all_handle[op_call.op.name](
                    *op_call.args, **_qnn_attrs(op_call.attrs)
                )

        class InsertQDQAfterMultiOutputOp(relay.ExprMutator):
            """Insert quantize/dequantize after ops that holds multi-outputs."""

            def visit_call(self, call):
                op_args = [self.visit(arg) for arg in call.args]
                call_attrs = _qnn_attrs(call.attrs)

                new_args = []
                q_param_idx = 0
                for i, arg in enumerate(op_args):
                    if isinstance(arg, TupleGetItem):
                        arg_call = arg.tuple_value
                        if arg_call.op.name in class_obj.multi_output_ops:
                            i_params = call_attrs["q_params"][i]

                            if i_params[1] == USE_MINMAX:
                                i_q_info = qc.get_quant_params(i_params, "output")
                                scale_value, zp_value = _get_scale_zp_value(i_q_info)
                            else:
                                # scale mode: qnn ir has been quantized
                                scale_value = i_params[3::2]
                                zp_value = i_params[4::2]
                                if len(scale_value) == 1:
                                    scale_value = scale_value[0]
                                    zp_value = zp_value[0]
                            scale_node, zp_node = _create_scale_zp_node(scale_value, zp_value)
                            dequantize_call = _inset_qdq_nodes(
                                arg,
                                scale_node,
                                zp_node,
                                class_obj.dtype_map[q_config.dtype_weight],
                                axis=1,
                            )
                            new_args.append(dequantize_call)
                            q_param_idx += 1
                            continue
                    elif isinstance(arg, Tuple):
                        new_arg_tuple = []
                        for j, a in enumerate(arg):
                            if isinstance(a, TupleGetItem):
                                a_call = a.tuple_value
                                if a_call.op.name in class_obj.multi_output_ops:
                                    i_params = call_attrs["q_params"][q_param_idx + j]

                                    if i_params[1] == USE_MINMAX:
                                        i_q_info = qc.get_quant_params(i_params, "output")
                                        scale_value, zp_value = _get_scale_zp_value(i_q_info)
                                    else:
                                        # scale mode: qnn ir has been quantized
                                        scale_value = i_params[3::2]
                                        zp_value = i_params[4::2]
                                        if len(scale_value) == 1:
                                            scale_value = scale_value[0]
                                            zp_value = zp_value[0]
                                    scale_node, zp_node = _create_scale_zp_node(
                                        scale_value, zp_value
                                    )
                                    dequantize_call = _inset_qdq_nodes(
                                        a,
                                        scale_node,
                                        zp_node,
                                        class_obj.dtype_map[q_config.dtype_weight],
                                        axis=1,
                                    )
                                    new_arg_tuple.append(dequantize_call)
                                    continue
                            new_arg_tuple.append(a)
                        new_args.append(Tuple(new_arg_tuple))
                        q_param_idx += len(arg)
                        continue

                    new_args.append(arg)
                    q_param_idx += 1

                return csi_op().all_handle[call.op.name](*new_args, **call_attrs)

        class InsertQDQAfterVar(relay.ExprMutator):
            """Insert quantize/dequantize after the inputs of model."""

            def visit_call(self, call):
                op_args = [self.visit(arg) for arg in call.args]
                call_attrs = _qnn_attrs(call.attrs)

                new_args = []
                for i, arg in enumerate(op_args):
                    if isinstance(arg, Var):
                        i_params = call_attrs["q_params"][i]

                        if arg.type_annotation.dtype not in ("float32", "float64", "float16"):
                            new_args.append(arg)
                            continue

                        if i_params[1] == USE_MINMAX:
                            i_q_info = qc.get_quant_params(i_params, "output")
                            scale_value, zp_value = _get_scale_zp_value(i_q_info)
                        else:
                            # scale mode: qnn ir has been quantized
                            scale_value = i_params[3::2]
                            zp_value = i_params[4::2]
                            if len(scale_value) == 1:
                                scale_value = scale_value[0]
                                zp_value = zp_value[0]
                        scale_node, zp_node = _create_scale_zp_node(scale_value, zp_value)
                        dequantize_call = _inset_qdq_nodes(
                            arg,
                            scale_node,
                            zp_node,
                            class_obj.dtype_map[q_config.dtype_weight],
                            axis=1,
                        )
                        new_args.append(dequantize_call)
                        continue
                    new_args.append(arg)

                return csi_op().all_handle[call.op.name](*new_args, **call_attrs)

        class ConvertConstantToTargetDtype(relay.ExprMutator):
            """Class helper that convert weight/bais/constant into quantized data."""

            def visit_call(self, call):
                op_args = [self.visit(arg) for arg in call.args]
                call_attrs = _qnn_attrs(call.attrs)

                new_op_args = []
                if call.op.name in list(csi_op().conv_handle.keys()) + ["qnn.csi.dense"]:
                    # deal with conv ops
                    in_node, weight_node, bias_node = op_args
                    new_op_args.append(in_node)

                    # deal with weight
                    if call.op.name in csi_op().conv_handle:
                        is_depthwise = _is_depthwise_conv(
                            infer_shape(in_node),
                            infer_shape(weight_node),
                            call_attrs["groups"],
                            call_attrs["kernel_layout"],
                        )
                        w_tensor_type = "depthwise_kernel" if is_depthwise else "conv_kernel"
                    else:
                        is_depthwise = False
                        w_tensor_type = "input1"
                    if call_attrs["q_params"][1][1] == USE_MINMAX:
                        w_quant_info = qc.get_quant_params(call_attrs["q_params"][1], w_tensor_type)
                        w_quantized_data = qc.quantize_weight(
                            weight_node.data.numpy(), w_quant_info, is_depthwise
                        )
                        new_weight_node = relay.const(w_quantized_data)
                        w_scale_value, w_zp_value = _get_scale_zp_value(w_quant_info)
                    else:
                        # scale mode: weight has been quantized
                        new_weight_node = weight_node
                        w_scale_value = call_attrs["q_params"][1][3::2]
                        w_zp_value = call_attrs["q_params"][1][4::2]
                        if len(w_scale_value) == 1:
                            w_scale_value = w_scale_value[0]
                            w_zp_value = w_zp_value[0]
                    w_scale_node, w_zp_node = _create_scale_zp_node(w_scale_value, w_zp_value)
                    w_dequantize_call = relay.qnn.op.csi_dequantize(
                        new_weight_node,
                        w_scale_node,
                        w_zp_node,
                        axis=0,
                        out_dtype="float32",
                        q_params=[],
                    )
                    new_op_args.append(w_dequantize_call)

                    # deal with bias
                    bias_data = bias_node.data.numpy()
                    if isinstance(bias_data, float) and math.isclose(bias_data, 0.0):
                        new_op_args.append(bias_node)
                    else:
                        # bias in not zero
                        in_quant_info = qc.get_quant_params(call_attrs["q_params"][0], "output")
                        if call.op.name in csi_op().conv_handle:
                            b_tensor_type = "depthwise_bias" if is_depthwise else "conv_bias"
                        else:
                            b_tensor_type = "dense_bias"

                        if call_attrs["q_params"][2][1] == USE_MINMAX:
                            b_quant_info = qc.get_quant_params(
                                call_attrs["q_params"][2], b_tensor_type
                            )

                            b_quantized_data = qc.quantize_bias(
                                bias_data, q_config.dtype_activation, in_quant_info, w_quant_info
                            )
                            new_bias_node = relay.const(b_quantized_data)

                            # scale of bias equals to in_scale * w_scale
                            assert (
                                in_quant_info.q_size == 1
                            ), "Activation can not be per-channel quantization."
                            b_scale_value, b_zp_value = [], []
                            for i in range(b_quant_info.q_size):
                                correct_scale = float(in_quant_info.qinfo[0].scale) * float(
                                    w_quant_info.qinfo[0].scale
                                )
                                if (
                                    in_quant_info.dtype == "int16_t"
                                    and w_quant_info.dtype == "int16_t"
                                    and b_quant_info.dtype == "int32_t"
                                    and abs(correct_scale) < 1e-5
                                ):
                                    correct_scale = 1e-5
                                b_scale_value.append(correct_scale)
                                b_zp_value.append(0)
                            if b_quant_info.q_size == 1:
                                b_scale_value = b_scale_value[0]
                                b_zp_value = b_zp_value[0]
                        else:
                            # scale mode: weight has been quantized
                            new_bias_node = bias_node
                            b_scale_value = call_attrs["q_params"][2][3::2]
                            b_zp_value = call_attrs["q_params"][2][4::2]
                            if len(b_scale_value) == 1:
                                b_scale_value = b_scale_value[0]
                                b_zp_value = b_zp_value[0]

                        b_scale_node, b_zp_node = _create_scale_zp_node(b_scale_value, b_zp_value)
                        b_dequantize_call = relay.qnn.op.csi_dequantize(
                            new_bias_node,
                            b_scale_node,
                            b_zp_node,
                            axis=0,
                            out_dtype="float32",
                            q_params=[],
                        )

                        new_op_args.append(b_dequantize_call)
                elif call.op.name in ("qnn.csi.quantize", "qnn.csi.dequantize"):
                    new_op_args = op_args
                else:
                    # deal with other op with constant input
                    start_idx = 0
                    for i, arg in enumerate(op_args):
                        if isinstance(arg, Constant):
                            if arg.data.numpy().dtype != numpy.float32:
                                new_op_args.append(arg)
                            else:
                                dequantized_call = _generate_dequantize_const_node(
                                    arg.data.numpy(), call_attrs["q_params"][i]
                                )
                                new_op_args.append(dequantized_call)
                            start_idx += 1
                        elif isinstance(arg, Tuple):
                            # the inputs of concat may hold constant input.
                            new_i_args = []
                            for j in range(len(arg)):
                                if isinstance(arg.fields[j], Constant):
                                    data = arg.fields[j].data.numpy()
                                    if data.dtype != numpy.float32:
                                        new_i_args.append(arg.fields[j])
                                    else:
                                        q_params = call_attrs["q_params"][start_idx + j]
                                        dequantized_call = _generate_dequantize_const_node(
                                            data, q_params
                                        )
                                        new_i_args.append(dequantized_call)
                                else:
                                    new_i_args.append(arg.fields[j])
                            new_op_args.append(Tuple(new_i_args))
                            start_idx += len(arg)
                        else:
                            new_op_args.append(arg)
                            start_idx += 1
                return csi_op().all_handle[call.op.name](*new_op_args, **call_attrs)

        # ensure that the ops which hold only an output have subsequent quantize/dequantize
        out = rewrite(InsertQDQAfterSingleOutputOp(), mod["main"].body)
        res = tvm.IRModule.from_expr(out)

        # ensure that the ops which hold multi-output have different quantize/dequantize
        res["main"] = InsertQDQAfterMultiOutputOp().visit(res["main"])

        # ensure that the inputs of model have subsequent quantize/dequantize
        res["main"] = InsertQDQAfterVar().visit(res["main"])

        # convert constant node into dequantize op
        res["main"] = ConvertConstantToTargetDtype().visit(res["main"])
        return res["main"]


@function_pass(opt_level=1)
class ConvertQnnToFloat16:
    """Convert Qnn ir into float32 dtype."""

    def __init__(self) -> None:
        self.dtype = "float16"

    def transform_function(self, func, mod, ctx):
        """Helper function to convert qnn ir."""
        func = relay.Function(func.params, func.body, None, func.type_params, func.attrs)
        class_obj = self
        q_config = QnnConfig()
        if q_config.calibrate_mode == "scale":
            raise ValueError(
                "Unsupport to convert qnn ir into onnx while calibrate_mode equals to 'scale'"
            )
        qc = QuantCalculator(q_config)

        class ConvertToFloat16(relay.ExprMutator):
            """Class helper that convert dtype into float16."""

            def visit_var(self, var):
                var_shape = infer_shape(var)
                new_var = relay.expr.var(var.name_hint, shape=var_shape, dtype=class_obj.dtype)
                return new_var

            def visit_constant(self, const):
                data = const.data.numpy()

                if data.dtype != numpy.float32:
                    const_quantized_data = data
                else:
                    fmin, fmax = _find_abs_minmax(data)
                    q_param = [1, 0, 0, float(fmin), float(fmax)]
                    const_quant_info = qc.get_quant_params(q_param, "output")
                    const_quantized_data = qc.quantize_weight(data, const_quant_info, False)
                new_const_node = relay.const(const_quantized_data)
                return new_const_node

        return ConvertToFloat16().visit(func)
