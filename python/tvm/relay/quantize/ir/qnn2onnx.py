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
from tvm.relay.expr import Call, Constant
from tvm.contrib.target.onnx import (
    RelayToONNXConverter,
    get_onnx_version,
    add_input,
    ModelContainer,
    run_onnx_optimizer,
    get_node_shape,
)
from tvm.relay.frontend.common import infer_shape

from .base import _qnn_attrs


ONNX_OPSET_VERSONS_SUPPORTED = [11, 13]


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
