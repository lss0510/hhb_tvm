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
# pylint: disable=unused-argument, invalid-name, too-many-nested-blocks
"""Convert qnn ir into onnx."""
import math
import numpy
import onnx
from onnx import defs
from onnx import TensorProto
import tvm
from tvm import relay
from tvm.relay.expr import RelayExpr as Expr
from tvm.relay.dataflow_pattern import DFPatternCallback
from tvm.relay.dataflow_pattern import wildcard
from tvm.relay.dataflow_pattern import rewrite
from tvm.relay.expr import Call, Var, Tuple, Constant, TupleGetItem
from tvm.relay.backend.contrib.csinn_backend import QnnConfig, QuantCalculator
from tvm.contrib.target.onnx import (
    RelayToONNXConverter,
    get_onnx_version,
    add_input,
    ModelContainer,
    run_onnx_optimizer,
    get_node_shape,
)
from tvm.relay.transform import function_pass
from tvm.relay.frontend.common import infer_shape, infer_type
from ._convert_to_csi import _qnn_attrs, csi_op, _find_abs_minmax
from ._convert_to_csi import USE_MINMAX

ONNX_OPSET_VERSONS_SUPPORTED = [11, 13]


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


class QnnOpConverter(object):
    """A helper class for holding Qnn op converters."""

    @classmethod
    def get_converter(cls, opset):
        """Get converter matches given opset.

        Parameters
        ----------
        opset: int
            opset from model.

        Returns
        -------
        converter, which should be `_impl_vx`. Number x is the biggest
            number smaller than or equal to opset belongs to all support versions.
        """
        versions = [int(d.replace("_impl_v", "")) for d in dir(cls) if "_impl_v" in d]
        versions = sorted(versions + [opset])
        version = versions[max([i for i, v in enumerate(versions) if v == opset]) - 1]
        if hasattr(cls, "_impl_v{}".format(version)):
            return getattr(cls, "_impl_v{}".format(version))
        raise NotImplementedError(
            "opset version {} of {} not implemented".format(version, cls.__name__)
        )

    @classmethod
    def convert_attributes(cls, attrs):
        """convert Qnn attributes to ONNX attributes.
        The derived classes should implement this method
        if attributes are required by the operator
        otherwise by default no attributes are passed
        """
        return {}

    @classmethod
    def _impl_v1(
        cls,
        node_entry,
        model_container,
        node_dict,
    ):
        qnn_attrs = _qnn_attrs(node_entry["relay_node"].attrs)
        onnx_attrs = cls.convert_attributes(qnn_attrs)
        onnx_node = onnx.helper.make_node(
            cls.__name__,
            node_entry["input_names"],
            node_entry["output_names"],
            qnn_attrs["layer_name"],
            **onnx_attrs,
        )
        model_container.add_nodes([onnx_node])


def qnn_rename(op_name):
    """This method creates dynamic operator of name op_name with empty attributes"""
    return type(op_name, (QnnOpConverter,), {})


class Conv(QnnOpConverter):
    """Qnn Operator converter for Conv."""

    @classmethod
    def convert_attributes(cls, attrs):
        return {
            "group": attrs["groups"],
            "pads": attrs["padding"],
            "strides": attrs["strides"],
            "dilations": attrs["dilation"],
            "kernel_shape": attrs["kernel_size"],
        }

    @classmethod
    def _impl_v1(
        cls,
        node_entry,
        model_container,
        node_dict,
    ):
        qnn_attrs = _qnn_attrs(node_entry["relay_node"].attrs)
        onnx_attrs = cls.convert_attributes(qnn_attrs)
        bias = node_entry["relay_node"].args[2]
        if isinstance(bias, Constant):
            bias_value = bias.data.numpy().tolist()
            if isinstance(bias_value, float) and math.isclose(bias_value, 0.0):
                node_entry["input_names"].pop(2)
        elif isinstance(bias, Call) and bias.op.name == "qnn.csi.dequantize":
            if isinstance(bias.args[0], Constant):
                bias_value = bias.args[0].data.numpy().tolist()
                if isinstance(bias_value, (float, int)) and math.isclose(bias_value, 0):
                    model_container.remove_node("DequantizeLinear_" + node_entry["input_names"][2])
                    node_entry["input_names"].pop(2)

        onnx_node = onnx.helper.make_node(
            cls.__name__,
            node_entry["input_names"],
            node_entry["output_names"],
            qnn_attrs["layer_name"],
            **onnx_attrs,
        )
        model_container.add_nodes([onnx_node])


class Conv2dRelu(QnnOpConverter):
    """Qnn Operator converter for Conv2dRelu."""

    @classmethod
    def convert_attributes(cls, attrs):
        return {
            "group": attrs["groups"],
            "pads": attrs["padding"],
            "strides": attrs["strides"],
            "dilations": attrs["dilation"],
            "kernel_shape": attrs["kernel_size"],
        }

    @classmethod
    def _impl_v1(
        cls,
        node_entry,
        model_container,
        node_dict,
    ):
        qnn_attrs = _qnn_attrs(node_entry["relay_node"].attrs)
        onnx_attrs = cls.convert_attributes(qnn_attrs)
        bias = node_entry["relay_node"].args[2]
        if isinstance(bias, Constant):
            bias_value = bias.data.numpy().tolist()
            if isinstance(bias_value, float) and math.isclose(bias_value, 0.0):
                node_entry["input_names"].pop(2)
        elif isinstance(bias, Call) and bias.op.name == "qnn.csi.dequantize":
            if isinstance(bias.args[0], Constant):
                bias_value = bias.args[0].data.numpy().tolist()
                if isinstance(bias_value, (float, int)) and math.isclose(bias_value, 0):
                    model_container.remove_node("DequantizeLinear_" + node_entry["input_names"][2])
                    node_entry["input_names"].pop(2)

        onnx_node = onnx.helper.make_node(
            "Conv",
            node_entry["input_names"],
            ["temp_conv2d_" + qnn_attrs["layer_name"]],
            "Conv" + "_" + qnn_attrs["layer_name"],
            **onnx_attrs,
        )
        model_container.add_nodes([onnx_node])

        onnx_node = onnx.helper.make_node(
            "Relu",
            ["temp_conv2d_" + qnn_attrs["layer_name"]],
            node_entry["output_names"],
            "Relu" + "_" + qnn_attrs["layer_name"],
        )
        model_container.add_nodes([onnx_node])


class Conv2dRelu6(QnnOpConverter):
    """Qnn Operator converter for Conv2dRelu6."""

    @classmethod
    def convert_attributes(cls, attrs):
        return {
            "group": attrs["groups"],
            "pads": attrs["padding"],
            "strides": attrs["strides"],
            "dilations": attrs["dilation"],
            "kernel_shape": attrs["kernel_size"],
        }

    @classmethod
    def _impl_v1(
        cls,
        node_entry,
        model_container,
        node_dict,
    ):
        qnn_attrs = _qnn_attrs(node_entry["relay_node"].attrs)
        onnx_attrs = cls.convert_attributes(qnn_attrs)
        bias = node_entry["relay_node"].args[2]
        if isinstance(bias, Constant):
            bias_value = bias.data.numpy().tolist()
            if isinstance(bias_value, float) and math.isclose(bias_value, 0.0):
                node_entry["input_names"].pop(2)
        elif isinstance(bias, Call) and bias.op.name == "qnn.csi.dequantize":
            if isinstance(bias.args[0], Constant):
                bias_value = bias.args[0].data.numpy().tolist()
                if isinstance(bias_value, (float, int)) and math.isclose(bias_value, 0):
                    model_container.remove_node("DequantizeLinear_" + node_entry["input_names"][2])
                    node_entry["input_names"].pop(2)

        onnx_node = onnx.helper.make_node(
            "Conv",
            node_entry["input_names"],
            ["temp_conv2d_" + qnn_attrs["layer_name"]],
            "Conv" + "_" + qnn_attrs["layer_name"],
            **onnx_attrs,
        )
        model_container.add_nodes([onnx_node])

        if node_entry["types"][0].dtype == "float16":
            min_val = numpy.float16(0)
            max_val = numpy.float16(6)
        else:
            min_val = numpy.float32(0)
            max_val = numpy.float32(6)

        input_names = [
            "temp_conv2d_" + qnn_attrs["layer_name"],
            add_input(min_val, "", "min", model_container),
            add_input(max_val, "", "max", model_container),
        ]

        onnx_node = onnx.helper.make_node(
            "Clip",
            input_names,
            node_entry["output_names"],
            "Clip" + "_" + qnn_attrs["layer_name"],
        )
        model_container.add_nodes([onnx_node])


class Split(QnnOpConverter):
    """Qnn Operator converter for Split."""

    @classmethod
    def convert_attributes(cls, attrs):
        return {
            "axis": attrs["axis"],
        }

    @classmethod
    def _impl_v1(
        cls,
        node_entry,
        model_container,
        node_dict,
    ):
        qnn_attrs = _qnn_attrs(node_entry["relay_node"].attrs)
        onnx_attrs = cls.convert_attributes(qnn_attrs)
        onnx_node = onnx.helper.make_node(
            cls.__name__,
            node_entry["input_names"],
            node_entry["output_names"],
            qnn_attrs["layer_name"],
            **onnx_attrs,
        )
        model_container.add_nodes([onnx_node])


class LeakyRelu(QnnOpConverter):
    """Qnn Operator converter for LeakyRelu."""

    @classmethod
    def convert_attributes(cls, attrs):
        return {
            "alpha": attrs["alpha"],
        }


class Relu6(QnnOpConverter):
    """Qnn Operator converter for Relu6."""

    @classmethod
    def _impl_v1(
        cls,
        node_entry,
        model_container,
        node_dict,
    ):
        qnn_attrs = _qnn_attrs(node_entry["relay_node"].attrs)

        name = node_entry["name"]
        if node_entry["types"][0].dtype == "float16":
            min_val = numpy.float16(0)
            max_val = numpy.float16(6)
        else:
            min_val = numpy.float32(0)
            max_val = numpy.float32(6)
        input_names = [
            add_input(min_val, name, "min", model_container),
            add_input(max_val, name, "max", model_container),
        ]

        input_names = [node_entry["input_names"][0]] + input_names

        node = onnx.helper.make_node(
            "Clip", input_names, node_entry["output_names"], qnn_attrs["layer_name"]
        )
        model_container.add_nodes([node])


class AveragePool(QnnOpConverter):
    """Qnn Operator converter for AveragePool."""

    @classmethod
    def convert_attributes(cls, attrs):
        return {
            "pads": attrs["padding"],
            "strides": attrs["strides"],
            "kernel_shape": attrs["pool_size"],
            "ceil_mode": int(attrs["ceil_mode"]),
            "count_include_pad": int(attrs["count_include_pad"]),
        }


class MaxPool(QnnOpConverter):
    """Qnn Operator converter for MaxPool."""

    @classmethod
    def convert_attributes(cls, attrs):
        return {
            "pads": attrs["padding"],
            "strides": attrs["strides"],
            "kernel_shape": attrs["pool_size"],
            "ceil_mode": int(attrs["ceil_mode"]),
        }


class Reshape(QnnOpConverter):
    """Qnn Operator converter for Reshape."""

    @classmethod
    def convert_attributes(cls, attrs):
        return {}

    @classmethod
    def _impl_v1(
        cls,
        node_entry,
        model_container,
        node_dict,
    ):
        qnn_attrs = _qnn_attrs(node_entry["relay_node"].attrs)
        onnx_attrs = cls.convert_attributes(qnn_attrs)
        name = node_entry["name"]
        shape = numpy.asarray(
            [a.value for a in node_entry["relay_node"].attrs.newshape], dtype=numpy.int64
        )
        input_names = [
            node_entry["input_names"][0],
            add_input(shape, name, "shape", model_container),
        ]
        onnx_node = onnx.helper.make_node(
            cls.__name__,
            input_names,
            node_entry["output_names"],
            qnn_attrs["layer_name"],
            **onnx_attrs,
        )
        model_container.add_nodes([onnx_node])


class Clip(QnnOpConverter):
    """Qnn Operator converter for Clip."""

    @classmethod
    def convert_attributes(cls, attrs):
        return {}

    @classmethod
    def _impl_v1(
        cls,
        node_entry,
        model_container,
        node_dict,
    ):
        qnn_attrs = _qnn_attrs(node_entry["relay_node"].attrs)
        name = node_entry["name"]

        if node_entry["types"][0].dtype == "float16":
            min_val = numpy.float16(node_entry["relay_node"].attrs.a_min)
            max_val = numpy.float16(node_entry["relay_node"].attrs.a_max)
        else:
            min_val = numpy.float32(node_entry["relay_node"].attrs.a_min)
            max_val = numpy.float32(node_entry["relay_node"].attrs.a_max)
        input_names = [
            node_entry["input_names"][0],
            add_input(min_val, name, "min", model_container),
            add_input(max_val, name, "max", model_container),
        ]
        onnx_node = onnx.helper.make_node(
            cls.__name__,
            input_names,
            node_entry["output_names"],
            qnn_attrs["layer_name"],
        )
        model_container.add_nodes([onnx_node])


class DepthToSpace(QnnOpConverter):
    """Qnn Operator converter for DepthToSpace."""

    @classmethod
    def convert_attributes(cls, attrs):
        return {
            "blocksize": attrs["block_size"],
        }


class SpaceToDepth(QnnOpConverter):
    """Qnn Operator converter for SpaceToDepth."""

    @classmethod
    def convert_attributes(cls, attrs):
        return {
            "blocksize": attrs["block_size"],
        }


class Cast(QnnOpConverter):
    """Qnn Operator converter for Cast."""

    @classmethod
    def convert_attributes(cls, attrs):
        dtype = str.upper(attrs["out_dtype"])
        dtype_list = [
            "UNDEFINED",
            "FLOAT",
            "UINT8",
            "INT8",
            "UINT16",
            "INT16",
            "INT32",
            "INT64",
            "STRING",
            "BOOL",
            "FLOAT16",
            "DOUBLE",
            "UINT32",
            "UINT64",
            "COMPLEX64",
            "COMPLEX128",
            "BFLOAT16",
            "FLOAT8E4M3FN",
            "FLOAT8E4M3FNUZ",
            "FLOAT8E5M2",
            "FLOAT8E5M2FNUZ",
        ]
        if dtype in dtype_list:
            return {"to": getattr(TensorProto, dtype)}
        elif dtype == "FLOAT32":
            return {"to": getattr(TensorProto, "FLOAT")}
        else:
            raise NotImplementedError(
                "The out_dtype '{0}' is "
                "not supported.\n".format(attrs["out_dtype"]) + "choices: {0}".format(dtype_list)
            )


class LRN(QnnOpConverter):
    """Qnn Operator converter for LRN."""

    @classmethod
    def convert_attributes(cls, attrs):
        return {
            "size": attrs["size"],
            "beta": attrs["beta"],
            "bias": attrs["bias"],
            "alpha": attrs["alpha"],
        }


class Softmax(QnnOpConverter):
    """Qnn Operator converter for Softmax."""

    @classmethod
    def convert_attributes(cls, attrs):
        return {
            "axis": attrs["axis"],
        }


class Concat(QnnOpConverter):
    """Qnn Operator converter for Concat."""

    @classmethod
    def convert_attributes(cls, attrs):
        return {
            "axis": attrs["axis"],
        }


class QuantizeLinear(QnnOpConverter):
    """Qnn Operator converter for quantize."""

    @classmethod
    def convert_attributes(cls, attrs):
        return {"axis": attrs["axis"]}

    @classmethod
    def _impl_v1(
        cls,
        node_entry,
        model_container,
        node_dict,
    ):
        scale = node_entry["relay_node"].args[1]
        zp = node_entry["relay_node"].args[2]
        if scale.data.numpy().size > 1 or zp.data.numpy().size > 1:
            raise ValueError("Onnx support per-channel quantization since opset 13.")
        return cls._impl_v13(
            node_entry,
            model_container,
            node_dict,
        )

    @classmethod
    def _impl_v13(
        cls,
        node_entry,
        model_container,
        node_dict,
    ):
        qnn_attrs = _qnn_attrs(node_entry["relay_node"].attrs)
        onnx_attrs = cls.convert_attributes(qnn_attrs)
        out_dtype = qnn_attrs["out_dtype"]
        # convert the dtype of zero_point into uint8/int8
        zp_node = node_entry["relay_node"].args[2]
        new_zp_data = zp_node.data.numpy().astype(out_dtype)
        zp_name = node_entry["input_names"][1].strip(model_container._name + "_")
        zp_name = "zp_" + zp_name
        input_names = [
            node_entry["input_names"][0],
            node_entry["input_names"][1],
            add_input(new_zp_data, zp_name, model_container._name, model_container),
        ]
        node = onnx.helper.make_node(
            cls.__name__,
            input_names,
            node_entry["output_names"],
            "QuantizeLinear_" + node_entry["output_names"][0],
            **onnx_attrs,
        )
        model_container.add_nodes([node])
        model_container.remove_input(node_entry["input_names"][2])


class DequantizeLinear(QnnOpConverter):
    """Qnn Operator converter for dequantize."""

    @classmethod
    def convert_attributes(cls, attrs):
        return {"axis": attrs["axis"]}

    @classmethod
    def _impl_v1(
        cls,
        node_entry,
        model_container,
        node_dict,
    ):
        scale = node_entry["relay_node"].args[1]
        zp = node_entry["relay_node"].args[2]
        if scale.data.numpy().size > 1 or zp.data.numpy().size > 1:
            raise ValueError("Onnx support per-channel quantization since opset 13.")
        return cls._impl_v13(
            node_entry,
            model_container,
            node_dict,
        )

    @classmethod
    def _impl_v13(
        cls,
        node_entry,
        model_container,
        node_dict,
    ):
        qnn_attrs = _qnn_attrs(node_entry["relay_node"].attrs)
        onnx_attrs = cls.convert_attributes(qnn_attrs)
        in0_node = node_entry["relay_node"].args[0]
        if isinstance(in0_node, Constant):
            in_dtype = in0_node.checked_type.dtype
        elif isinstance(in0_node, Call):
            in_dtype = in0_node.attrs.out_dtype
        else:
            raise ValueError(f"unsupport node type: {type(in0_node)}")

        # convert the dtype of zero_point into uint8/int8
        zp_node = node_entry["relay_node"].args[2]
        new_zp_data = zp_node.data.numpy().astype(in_dtype)

        zp_name = node_entry["input_names"][1].strip(model_container._name + "_")
        zp_name = "deq_zp_" + zp_name
        input_names = [
            node_entry["input_names"][0],
            node_entry["input_names"][1],
            add_input(new_zp_data, zp_name, model_container._name, model_container),
        ]
        node = onnx.helper.make_node(
            cls.__name__,
            input_names,
            node_entry["output_names"],
            "DequantizeLinear_" + node_entry["output_names"][0],
            **onnx_attrs,
        )
        model_container.add_nodes([node])
        model_container.remove_input(node_entry["input_names"][2])


class Transpose(QnnOpConverter):
    """Qnn Operator converter for Transpose."""

    @classmethod
    def convert_attributes(cls, attrs):
        return {
            "perm": attrs["axes"],
        }


class Dense(QnnOpConverter):
    """Qnn Operator converter for Dense."""

    @classmethod
    def convert_attributes(cls, attrs):
        return {
            "transB": 1,
        }

    @classmethod
    def _impl_v1(
        cls,
        node_entry,
        model_container,
        node_dict,
    ):
        qnn_attrs = _qnn_attrs(node_entry["relay_node"].attrs)
        onnx_attrs = cls.convert_attributes(qnn_attrs)
        input_names = [
            node_entry["input_names"][0],
            node_entry["input_names"][1],
            node_entry["input_names"][2],
        ]
        onnx_node = onnx.helper.make_node(
            "Gemm",
            input_names,
            node_entry["output_names"],
            "Gemm" + "_" + qnn_attrs["layer_name"],
            **onnx_attrs,
        )
        model_container.add_nodes([onnx_node])


class ArgMax(QnnOpConverter):
    """Qnn Operator converter for ArgMax."""

    @classmethod
    def convert_attributes(cls, attrs):
        return {
            "axis": attrs["axis"][0],
            "keepdims": attrs["keepdims"],
            "select_last_index": attrs["exclude"],
        }

    @classmethod
    def _impl_v1(
        cls,
        node_entry,
        model_container,
        node_dict,
    ):
        qnn_attrs = _qnn_attrs(node_entry["relay_node"].attrs)
        onnx_attrs = cls.convert_attributes(qnn_attrs)
        node_entry["types"][0].dtype = "int64"
        onnx_node = onnx.helper.make_node(
            cls.__name__,
            node_entry["input_names"],
            node_entry["output_names"],
            qnn_attrs["layer_name"],
            **onnx_attrs,
        )
        model_container.add_nodes([onnx_node])


class ArgMin(QnnOpConverter):
    """Qnn Operator converter for ArgMin."""

    @classmethod
    def convert_attributes(cls, attrs):
        return {
            "axis": attrs["axis"][0],
            "keepdims": attrs["keepdims"],
            "select_last_index": attrs["exclude"],
        }

    @classmethod
    def _impl_v1(
        cls,
        node_entry,
        model_container,
        node_dict,
    ):
        qnn_attrs = _qnn_attrs(node_entry["relay_node"].attrs)
        onnx_attrs = cls.convert_attributes(qnn_attrs)
        node_entry["types"][0].dtype = "int64"
        onnx_node = onnx.helper.make_node(
            cls.__name__,
            node_entry["input_names"],
            node_entry["output_names"],
            qnn_attrs["layer_name"],
            **onnx_attrs,
        )
        model_container.add_nodes([onnx_node])


class BiasAdd(QnnOpConverter):
    """Qnn Operator converter for BiasAdd."""

    @classmethod
    def _impl_v1(
        cls,
        node_entry,
        model_container,
        node_dict,
    ):
        qnn_attrs = _qnn_attrs(node_entry["relay_node"].attrs)
        onnx_attrs = cls.convert_attributes(qnn_attrs)
        name = node_entry["name"]
        shape = numpy.asarray(node_entry["types"][0].shape[1:], dtype=numpy.int64)
        shape = numpy.flip(shape, 0)
        input_names = [
            node_entry["input_names"][1],
            add_input(shape, name, "shape", model_container),
        ]
        onnx_node = onnx.helper.make_node(
            "Expand",
            input_names,
            ["output_temp_" + qnn_attrs["layer_name"]],
            "Expand" + "_" + qnn_attrs["layer_name"],
            **onnx_attrs,
        )
        model_container.add_nodes([onnx_node])

        onnx_node = onnx.helper.make_node(
            "Transpose",
            ["output_temp_" + qnn_attrs["layer_name"]],
            ["output_temp2_" + qnn_attrs["layer_name"]],
            "Transpose" + "_" + qnn_attrs["layer_name"],
        )
        model_container.add_nodes([onnx_node])

        input_names_add = [
            node_entry["input_names"][0],
            "output_temp2_" + qnn_attrs["layer_name"],
        ]
        onnx_node = onnx.helper.make_node(
            "Add",
            input_names_add,
            node_entry["output_names"],
            "Add" + "_" + qnn_attrs["layer_name"],
        )
        model_container.add_nodes([onnx_node])


class MatMul(QnnOpConverter):
    """Qnn Operator converter for MatMul."""

    @classmethod
    def _impl_v1(
        cls,
        node_entry,
        model_container,
        node_dict,
    ):
        qnn_attrs = _qnn_attrs(node_entry["relay_node"].attrs)

        lhs_name = node_entry["input_names"][0]
        rhs_name = node_entry["input_names"][1]

        lhs_node = node_dict[node_entry["inputs"][0]][0]
        rhs_node = node_dict[node_entry["inputs"][1]][0]

        lhs_shape = get_node_shape(lhs_node["types"][0])
        rhs_shape = get_node_shape(rhs_node["types"][0])
        if qnn_attrs["transpose_a"]:
            trans_output = "transpose_a_" + qnn_attrs["layer_name"]
            perm = list(range(len(lhs_shape)))
            if len(perm) >= 2:
                perm[-2], perm[-1] = perm[-1], perm[-2]
            trans_node = onnx.helper.make_node(
                "Transpose",
                [lhs_name],
                [trans_output],
                perm=perm,
            )
            model_container.add_nodes([trans_node])
            lhs_name = trans_output

        if qnn_attrs["transpose_b"]:
            trans_output = "transpose_b_" + qnn_attrs["layer_name"]
            perm = list(range(len(rhs_shape)))
            if len(perm) >= 2:
                perm[-2], perm[-1] = perm[-1], perm[-2]
            trans_node = onnx.helper.make_node(
                "Transpose",
                [rhs_name],
                [trans_output],
                perm=perm,
            )
            model_container.add_nodes([trans_node])
            rhs_name = trans_output

        is_valid_bias = True
        bias = node_entry["relay_node"].args[2]
        if isinstance(bias, Constant):
            bias_value = bias.data.numpy().tolist()
            if isinstance(bias_value, float) and math.isclose(bias_value, 0.0):
                is_valid_bias = False
        elif isinstance(bias, Call) and bias.op.name == "qnn.csi.dequantize":
            if isinstance(bias.args[0], Constant):
                bias_value = bias.args[0].data.numpy().tolist()
                if isinstance(bias_value, (float, int)) and math.isclose(bias_value, 0):
                    is_valid_bias = False

        if is_valid_bias:
            matmul_node = onnx.helper.make_node(
                cls.__name__,
                [lhs_name, rhs_name],
                ["matmul_out_" + qnn_attrs["layer_name"]],
                "matmul_" + qnn_attrs["layer_name"],
            )
            model_container.add_nodes([matmul_node])
            onnx_node = onnx.helper.make_node(
                "Add",
                ["matmul_out_" + qnn_attrs["layer_name"], node_entry["input_names"][2]],
                node_entry["output_names"],
                qnn_attrs["layer_name"],
            )
            model_container.add_nodes([onnx_node])
        else:
            onnx_node = onnx.helper.make_node(
                cls.__name__,
                [lhs_name, rhs_name],
                node_entry["output_names"],
                qnn_attrs["layer_name"],
            )
            model_container.add_nodes([onnx_node])
            model_container.remove_input(node_entry["input_names"][2])


class ConvTranspose(QnnOpConverter):
    """Qnn Operator converter for ConvTranspose."""

    @classmethod
    def convert_attributes(cls, attrs):
        return {
            "group": attrs["groups"],
            "pads": attrs["padding"],
            "strides": attrs["strides"],
            "dilations": attrs["dilation"],
            "kernel_shape": attrs["kernel_size"],
        }

    @classmethod
    def _impl_v1(
        cls,
        node_entry,
        model_container,
        node_dict,
    ):
        qnn_attrs = _qnn_attrs(node_entry["relay_node"].attrs)
        onnx_attrs = cls.convert_attributes(qnn_attrs)
        bias = node_entry["relay_node"].args[2]
        if isinstance(bias, Constant):
            bias_value = bias.data.numpy().tolist()
            if isinstance(bias_value, float) and math.isclose(bias_value, 0.0):
                node_entry["input_names"].pop(2)
        elif isinstance(bias, Call) and bias.op.name == "qnn.csi.dequantize":
            if isinstance(bias.args[0], Constant):
                bias_value = bias.args[0].data.numpy().tolist()
                if isinstance(bias_value, (float, int)) and math.isclose(bias_value, 0):
                    node_entry["input_names"].pop(2)

        onnx_node = onnx.helper.make_node(
            cls.__name__,
            node_entry["input_names"],
            node_entry["output_names"],
            qnn_attrs["layer_name"],
            **onnx_attrs,
        )
        model_container.add_nodes([onnx_node])


class Pad(QnnOpConverter):
    """Qnn Operator converter for Pad."""

    @classmethod
    def convert_attributes(cls, attrs):
        return {
            "mode": attrs["pad_mode"],
        }

    @classmethod
    def _impl_v1(
        cls,
        node_entry,
        model_container,
        node_dict,
    ):
        qnn_attrs = _qnn_attrs(node_entry["relay_node"].attrs)
        onnx_attrs = cls.convert_attributes(qnn_attrs)
        pads = list(
            numpy.asarray(
                _qnn_attrs(node_entry["relay_node"].attrs)["pad_width"], dtype=numpy.int64
            ).flatten()
        )
        pp = []
        for i in range(0, len(pads), 2):
            pp.append(pads[i])
        for i in range(1, len(pads), 2):
            pp.append(pads[i])
        name = node_entry["name"]
        pads = numpy.asarray(pp, dtype=numpy.int64)
        input_names = [
            node_entry["input_names"][0],
            add_input(pads, name, "pads", model_container),
            node_entry["input_names"][1],
        ]
        onnx_node = onnx.helper.make_node(
            cls.__name__,
            input_names,
            node_entry["output_names"],
            qnn_attrs["layer_name"],
            **onnx_attrs,
        )
        model_container.add_nodes([onnx_node])


class Slice(QnnOpConverter):
    """Qnn Operator converter for Slice."""

    @classmethod
    def _impl_v1(cls, node_entry, model_container, node_dict):
        qnn_attrs = _qnn_attrs(node_entry["relay_node"].attrs)
        name = node_entry["name"]
        begin = numpy.asarray(qnn_attrs["begin"], dtype=numpy.int64)
        end = numpy.asarray(qnn_attrs["end"], dtype=numpy.int64)
        axes = numpy.asarray(range(len(begin)), dtype=numpy.int64)
        steps = numpy.asarray(qnn_attrs["strides"], dtype=numpy.int64)
        input_names = [
            node_entry["input_names"][0],
            add_input(begin, name, "begin", model_container),
            add_input(end, name, "end", model_container),
            add_input(axes, name, "axes", model_container),
            add_input(steps, name, "steps", model_container),
        ]
        onnx_node = onnx.helper.make_node(
            cls.__name__,
            input_names,
            node_entry["output_names"],
            qnn_attrs["layer_name"],
        )
        model_container.add_nodes([onnx_node])


class Take(QnnOpConverter):
    """Qnn Operator converter for Take."""

    @classmethod
    def convert_attributes(cls, attrs):
        return {
            "axis": int(attrs["axis"]),
        }

    @classmethod
    def _impl_v1(
        cls,
        node_entry,
        model_container,
        node_dict,
    ):
        qnn_attrs = _qnn_attrs(node_entry["relay_node"].attrs)
        onnx_attrs = cls.convert_attributes(qnn_attrs)
        onnx_node = onnx.helper.make_node(
            "Gather",
            node_entry["input_names"],
            node_entry["output_names"],
            qnn_attrs["layer_name"],
            **onnx_attrs,
        )
        model_container.add_nodes([onnx_node])


class Resize(QnnOpConverter):
    """Qnn Operator converter for Resize."""

    @classmethod
    def convert_attributes(cls, attrs):
        onnx_attrs = {}
        if attrs["method"] == "nearest_neighbor":
            onnx_attrs["mode"] = "nearest"
        else:
            onnx_attrs["mode"] = attrs["method"]
        if attrs["align_corners"]:
            onnx_attrs["coordinate_transformation_mode"] = "align_corners"
        return onnx_attrs

    @classmethod
    def _impl_v13(
        cls,
        node_entry,
        model_container,
        node_dict,
    ):
        qnn_attrs = _qnn_attrs(node_entry["relay_node"].attrs)
        onnx_attrs = cls.convert_attributes(qnn_attrs)
        x = node_entry["relay_node"].args[0]
        x_shape = infer_shape(x)
        assert len(x_shape) == 4, "Only support 4-dim shape of resize"

        scales = [1.0, 1.0, 1.0, 1.0]
        if qnn_attrs["layout"] == "NCHW":
            scales[2], scales[3] = float(qnn_attrs["scale_h"]), float(qnn_attrs["scale_w"])
        elif qnn_attrs["layout"] == "NHWC":
            scales[1], scales[2] = float(qnn_attrs["scale_h"]), float(qnn_attrs["scale_w"])
        else:
            raise ValueError(f"Unsupport for {qnn_attrs['layout']}")

        scales = numpy.array(scales, dtype=numpy.float32)

        x_name = node_entry["input_names"][0].strip(model_container._name + "_")
        input_names = [
            node_entry["input_names"][0],
            "",
            add_input(scales, "scales_" + x_name, model_container._name, model_container),
            "",
        ]

        onnx_node = onnx.helper.make_node(
            cls.__name__,
            input_names,
            node_entry["output_names"],
            qnn_attrs["layer_name"],
            **onnx_attrs,
        )
        model_container.add_nodes([onnx_node])


class ReduceMean(QnnOpConverter):
    """Operator convertor for ReduceMean"""

    @classmethod
    def _impl_v13(cls, node_entry, model_container, node_dict):
        qnn_attrs = _qnn_attrs(node_entry["relay_node"].attrs)
        input_node = node_dict[node_entry["inputs"][0]]
        assert len(input_node) == 1, "input node can not be a Tuple"
        input_node = input_node[0]
        shape = input_node["types"][0].shape
        axis = qnn_attrs["axis"]
        if not axis:
            axis = list(range(len(shape)))
        exclude = 0 if not bool(qnn_attrs["exclude"]) else 1
        keepdims = 0 if not bool(qnn_attrs["keepdims"]) else 1
        if exclude:
            all_axis = list(range(len(shape)))
            axis = set(all_axis) - set(axis)

        node = onnx.helper.make_node(
            cls.__name__,
            node_entry["input_names"],
            node_entry["output_names"],
            qnn_attrs["layer_name"],
            axes=axis,
            keepdims=keepdims,
        )
        model_container.add_nodes([node])


QNN_TO_ONNX_OP_MAPPING = {
    "qnn.csi.abs": qnn_rename("Abs"),
    "qnn.csi.acos": qnn_rename("Acos"),
    "qnn.csi.acosh": qnn_rename("Acosh"),
    "qnn.csi.asin": qnn_rename("Asin"),
    "qnn.csi.asinh": qnn_rename("Asinh"),
    "qnn.csi.atan": qnn_rename("Atan"),
    "qnn.csi.atanh": qnn_rename("Atanh"),
    "qnn.csi.argmin": ArgMin,
    "qnn.csi.argmax": ArgMax,
    "qnn.csi.add": qnn_rename("Add"),
    "qnn.csi.avgpool2d": AveragePool,
    "qnn.csi.bn": qnn_rename("BatchNormalization"),
    "qnn.csi.bias_add": BiasAdd,
    "qnn.csi.cast": Cast,
    "qnn.csi.clip": Clip,
    "qnn.csi.concatenate": Concat,
    "qnn.csi.conv1d": Conv,
    "qnn.csi.conv2d": Conv,
    "qnn.csi.conv2d_relu": Conv2dRelu,
    "qnn.csi.conv2d_relu6": Conv2dRelu6,
    "qnn.csi.cos": qnn_rename("Cos"),
    "qnn.csi.cosh": qnn_rename("Cosh"),
    "qnn.csi.deconv2d": ConvTranspose,
    "qnn.csi.dense": Dense,
    "qnn.csi.depth_to_space": DepthToSpace,
    "qnn.csi.dequantize": DequantizeLinear,
    "qnn.csi.div": qnn_rename("Div"),
    "qnn.csi.exp": qnn_rename("Exp"),
    "qnn.csi.erf": qnn_rename("Erf"),
    "qnn.csi.flatten": qnn_rename("Flatten"),
    "qnn.csi.global_maxpool2d": qnn_rename("GlobalMaxPool"),
    "qnn.csi.global_avgpool2d": qnn_rename("GlobalAveragePool"),
    "qnn.csi.log_softmax": qnn_rename("LogSoftmax"),
    "qnn.csi.lrn": LRN,
    "qnn.csi.leaky_relu": LeakyRelu,
    "qnn.csi.maxpool2d": MaxPool,
    "qnn.csi.mul": qnn_rename("Mul"),
    "qnn.csi.matmul": MatMul,
    "qnn.csi.pad": Pad,
    "qnn.csi.power": qnn_rename("Pow"),
    "qnn.csi.prelu": qnn_rename("PRelu"),
    "qnn.csi.quantize": QuantizeLinear,
    "qnn.csi.mean": ReduceMean,
    "qnn.csi.relu": qnn_rename("Relu"),
    "qnn.csi.relu6": Relu6,
    "qnn.csi.reshape": Reshape,
    "qnn.csi.upsampling": Resize,
    "qnn.csi.sigmoid": qnn_rename("Sigmoid"),
    "qnn.csi.sin": qnn_rename("Sin"),
    "qnn.csi.sinh": qnn_rename("Sinh"),
    "qnn.csi.softmax": Softmax,
    "qnn.csi.subtract": qnn_rename("Sub"),
    "qnn.csi.squeeze": qnn_rename("Squeeze"),
    "qnn.csi.sqrt": qnn_rename("Sqrt"),
    "qnn.csi.split": Split,
    "qnn.csi.space_to_depth": SpaceToDepth,
    "qnn.csi.strided_slice": Slice,
    "qnn.csi.transpose": Transpose,
    "qnn.csi.take": Take,
    "qnn.csi.tan": qnn_rename("Tan"),
    "qnn.csi.tanh": qnn_rename("Tanh"),
}


class QnnModelContainer(ModelContainer):
    """A container class to hold  different attributes of ONNX model graph"""

    def remove_input(self, name=""):
        """Remove the onnx input from the graph"""
        assert isinstance(name, str), "input var must be a string"
        name_to_input = {}
        for data in self._inputs:
            name_to_input[data.name] = data

        for initializer in self._initializers:
            if initializer.name in name_to_input:
                if name_to_input[initializer.name] in self._inputs:
                    self._inputs.remove(name_to_input[initializer.name])
        if name != "":
            for i in range(len(self._initializers)):
                if self._initializers[i].name == name:
                    self._initializers.pop(i)
                    break

    def remove_node(self, name=""):
        """Remove the onnx node from the graph"""
        assert isinstance(name, str), "node name must be a string"
        for node in self._nodes:
            if node.name == name:
                self._nodes.remove(node)
                break


class QnnToONNXConvert(RelayToONNXConverter):
    """A helper class to traverse the Qnn graph and convert Qnn nodes to ONNX model.

    Parameters
    ----------
    name : str
       name of the model

    params : dict
        dict of the parameter names and NDarray values

    opset_version : int
        target onnx opset version

    """

    def __init__(self, name, params, opset_version):
        super().__init__(name, params, opset_version)
        self._name = name
        self._mc = QnnModelContainer(name, opset_version)
        self._params = params
        self._node_dict = {}
        self._node_count = 0
        self.last_node = None

    def _add_node(self, node_entry, idx):
        """Convert Qnn operator node to ONNX operator and add it to container nodes list"""
        if node_entry["op"].name not in QNN_TO_ONNX_OP_MAPPING:
            raise NotImplementedError(
                "Currently the operator '{0}' is " "not supported.".format(node_entry["op"].name)
            )
        converter = QNN_TO_ONNX_OP_MAPPING[node_entry["op"].name]().get_converter(
            self._mc._opset_version
        )
        return converter(node_entry, self._mc, self._node_dict)

    def convert_to_onnx(self, func):
        """Traverse Relay graph and generate a ONNX model"""
        self.visit(func)
        self._add_output(self._node_dict[self.last_node])
        self._mc.remove_input()
        model = self._mc.make_model()
        return run_onnx_optimizer(model)


def qnn_to_onnx(relay_ir, params, name, opset_version=13, path=None):
    """Convert a Qnn Function Module into an equivalent ONNX and serialize it to the path

    Parameters
    ----------
    relay_ir : tvm.ir.IRModule or tvm.relay.Function
        The relay module object

    params : dict
        dict of the parameter names and NDarray values

    name : str
        name of the output ONNX graph

    opset_version : int
        target onnx opset version

    path : str
        The path where ONNX model will be saved

    Returns
    -------
    onnx_model : onnx.ModelProto
        converted ONNX model as a ModelProto.

    """

    if opset_version not in ONNX_OPSET_VERSONS_SUPPORTED:
        raise NotImplementedError("Currently only opset version 11 is supported.")

    if opset_version > defs.onnx_opset_version():
        raise Exception(
            "The ONNX package installed of version {} does not support the opset "
            "version {}. Upgrade the ONNX package to latest version.".format(
                get_onnx_version(), opset_version
            )
        )

    func = relay_ir["main"] if isinstance(relay_ir, tvm.ir.IRModule) else relay_ir
    converter = QnnToONNXConvert(name, params, opset_version)
    onnx_model = converter.convert_to_onnx(func)
    if path:
        onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
        onnx.save(onnx_model, path)
    return onnx_model
