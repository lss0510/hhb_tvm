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
# pylint: disable=invalid-name,unused-argument, not-context-manager
"""QNN dialect operators."""

from __future__ import absolute_import as _abs

import tvm
import tvm.ir
from tvm import relay
from tvm.relay.expr import Tuple, TupleWrapper
from tvm.relay.op.nn.utils import get_pad_tuple2d, get_pad_tuple1d
from tvm.runtime import Object
from tvm.target import Target
from tvm.topi.nn.qnn import SQNN_DTYPE_TO_CODE
from tvm.topi.x86.utils import target_has_sse41

from ... import op as reg
from ...op import OpPattern
from . import _make, _requantize


@tvm._ffi.register_object("relay.qnn.op.RequantizeConfig")
class RequantizeConfig(Object):
    """Configure the requantization behavior by setting config variables.

    Note
    ----
    This object is backed by node system in C++, with arguments that can be
    exchanged between python and C++.

    Do not construct directly, use requantize_config instead.

    The fields that are backed by the C++ node are immutable once an instance
    is constructed. Use _node_defaults getters to get results for the fields.
    """

    @staticmethod
    def _get_node_default_rounding():
        return "UPWARD"

    @staticmethod
    def _get_node_default_compute_dtype():
        target = Target.current(True)
        if target and str(target.kind) == "llvm" and target_has_sse41(target.mcpu):
            return "float32"

        return "int64"

    _node_defaults = {
        "rounding": _get_node_default_rounding.__func__,
        "compute_dtype": _get_node_default_compute_dtype.__func__,
    }

    # pylint: disable=no-member
    def __init__(self, handle):
        """Initialize the function with handle

        Parameters
        ----------
        handle : SymbolHandle
            the handle to the underlying C++ Symbol
        """
        super(RequantizeConfig, self).__init__(handle)
        self.handle = handle

    def __enter__(self):
        # pylint: disable=protected-access
        _requantize._EnterRequantizeConfigScope(self)
        return self

    def __exit__(self, ptype, value, trace):
        _requantize._ExitRequantizeConfigScope()

    def __setattr__(self, name, value):
        if name in RequantizeConfig._node_defaults:
            raise AttributeError("'%s' object cannot set attribute '%s'" % (str(type(self)), name))
        return super(RequantizeConfig, self).__setattr__(name, value)


def current_requantize_config():
    """Get the current requantization configuration."""
    return _requantize._GetCurrentRequantizeConfig()


def requantize_config(**kwargs):
    """Configure the requantization behavior by setting config variables.

    Parameters
    ---------
    rounding: "UPWARD" or "TONEAREST"
        Rounding direction for fixed point multiplications.
    compute_dtype:
        Specifies the data type used during requantize.
        Supported options: \"int64\", \"float32\", \"float64\"

    Returns
    -------
    config: RequantizeConfig
        The requantization configuration
    """
    node_args = {
        k: v() if k not in kwargs else kwargs[k] for k, v in RequantizeConfig._node_defaults.items()
    }
    return tvm.ir.make_node("relay.qnn.op.RequantizeConfig", **node_args)


def requantize(
    data,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    axis=-1,
    rounding="None",
    compute_dtype="None",
    out_dtype="int8",
):
    r"""Requantized operator.

    The requantize operator converts one quantized tensor representation to
    another quantized tensor representation. For the output tensor, we are
    provided with output scale and zero point. The computation is as follows

    Q_output = zp_output +  (scale_input)/(scale_output) * (Q_input - zp_input)

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    input_scale: tvm.relay.Expr
        The quantization scale for the input tensor.

    input_zero_point: tvm.relay.Expr
        The zero point of the input tensor.

    output_scale: tvm.relay.Expr
        The quantization scale for the output tensor.

    output_zero_point: tvm.relay.Expr
        The zero point of the output tensor.

    axis : int
        The channel axis for quantization. Default value is -1 which corresponds to the last axis.

    rounding : string, optional
        Defines the rounding direction when the value is midway between two
        representable values.
    compute_dtype:
        Specifies the data type used during requantize.
        Supported options: \"int64\", \"float32\", \"float64\"
    out_dtype : str, optional
        Specifies the output data type.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """

    return _make.requantize(
        data,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        axis,
        rounding,
        compute_dtype,
        out_dtype,
    )


def quantize(data, output_scale, output_zero_point, axis=-1, out_dtype="int8"):
    r"""Quantize op
    This operator takes float32 as input and produces quantized int8 or unit8 as output.
    The input tensor can be of any shape. The output shape is the same as input shape.

    Q_output = clamp((round(input_tensor/output_scale) + output_zero_point),
                     out_dtype::min,
                     out_dtype::max)

    Parameters
    ----------
    data : tvm.relay.Expr
        The input tensor to be quantized. Can be of type float32.

    output_scale : tvm.relay.Expr
        The output scale.

    output_zero_point : tvm.relay.Expr
        The output zero_point.

    axis : int
        The channel axis for quantization. Default value is -1 which corresponds to the last axis.
    out_dtype : str, optional
        The data type of the input tensor. Can be [int8, uint8, int32]

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """

    return _make.quantize(data, output_scale, output_zero_point, axis, out_dtype)


def simulated_quantize(data, output_scale, output_zero_point, axis=-1, out_dtype="int8"):
    r"""Simulated Quantize op
    Mimics the quantize op but has more flexibility in valid inputs and always
    outputs the same type as the input. This can be useful for
    calibrating or training a quantized network.

    Parameters
    ----------
    data : tvm.relay.Expr
        The input tensor to be quantized. Can be of type float32.

    out_dtype : string or tvm.relay.Expr
        A string or tensor indicating which datatype to quantize to.

    output_scale : tvm.relay.Expr
        The output scale.

    output_zero_point : tvm.relay.Expr
        The output zero_point.

    axis : int
        The channel axis for quantization. Default value is -1 which corresponds to the last axis.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    # Convert string dtype to a constant if needed.
    if isinstance(out_dtype, str):
        type_code = SQNN_DTYPE_TO_CODE[out_dtype]
        out_dtype = relay.const(type_code, dtype="int32")
    # Wrap reshapes around qnn parameter tensors to guarantee shape compatibility.
    output_scale = relay.op.reshape(output_scale, [-1])
    output_zero_point = relay.op.reshape(output_zero_point, [-1])
    return _make.simulated_quantize(data, out_dtype, output_scale, output_zero_point, axis)


def dequantize(data, input_scale, input_zero_point, axis=-1):
    r"""Dequantize op
    This operator takes quantized int8 and unit8 as input and produces
    dequantized float32 as output. The output shape is the same as input shape. The input
    tensor can be of any shape.

    Parameters
    ----------
    data : tvm.relay.Expr
        The input tensor to be dequantized. Can be of type [int8, uint8, int32].

    input_scale : tvm.relay.Expr
        The input scale.

    input_zero_point : tvm.relay.Expr
        The input zero_point.

    axis : int
        The channel axis for quantization. Default value is -1 which corresponds to the last axis.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """

    return _make.dequantize(data, input_scale, input_zero_point, axis)


def csinn_deinit(data, input_scale, input_zero_point):
    r"""Dequantize op
    This operator takes quantized int8 and unit8 as input and produces
    dequantized float32 as output. The output shape is the same as input shape. The input
    tensor can be of any shape.
        The input tensor to be dequantized. Can be of type [int8, uint8].
    input_zero_point : int
        The output zero_point.
    input_scale : float
        The output scale.
    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """

    return _make.CSINNDeinit(data, input_scale, input_zero_point)


def csinn_init(
    data, output_scale, output_zero_point, out_dtype="uint8", max_values=tuple(), min_values=tuple()
):
    r"""init op
    This operator takes float32 as input and produces quantized int8 or unit8 as output.
    The input tensor can be of any shape. The output shape is the same as input shape.

    Q_output = clamp((round(input_tensor/output_scale) + output_zero_point),
                     out_dtype::min,
                     out_dtype::max)

    Parameters
    ----------
    data : tvm.relay.Expr
        The input tensor to be quantized. Can be of type float32.
    output_zero_point : int
        The output zero_point.
    output_scale : float
        The output scale.
    out_dtype : str, optional
        The data type of the input tensor. Can be [int8, uint8]
    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """

    return _make.CSINNInit(data, output_scale, output_zero_point, out_dtype, max_values, min_values)


def simulated_dequantize(data, input_scale, input_zero_point, axis=-1, in_dtype="int8"):
    r"""Simulated Dequantize op
    Mimics the dequantize op but has more flexibility in valid inputs and always
    outputs the same type as the input. This can be useful for calibrating or
    training a quantized network.
    Parameters
    ----------
    data : tvm.relay.Expr
        The input tensor to be dequantized.

    in_dtype : string or tvm.relay.Expr
        A string or tensor indicating which datatype to dequantize from.

    input_scale : tvm.relay.Expr
        The input scale.

    input_zero_point : tvm.relay.Expr
        The input zero_point.

    axis : int
        The channel axis for quantization. Default value is -1 which corresponds to the last axis.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    # Convert string dtype to a constant if needed.
    if isinstance(in_dtype, str):
        type_code = SQNN_DTYPE_TO_CODE[in_dtype]
        in_dtype = relay.const(type_code, dtype="int32")
    # Wrap reshapes around qnn parameter tensors to guarantee shape compatibility.
    input_scale = relay.op.reshape(input_scale, [-1])
    input_zero_point = relay.op.reshape(input_zero_point, [-1])
    return _make.simulated_dequantize(data, in_dtype, input_scale, input_zero_point, axis)


def concatenate(data, input_scales, input_zero_points, output_scale, output_zero_point, axis):
    """Concatenate the quantized input tensors along the given axis.

    Parameters
    ----------
    data : Union(List[relay.Expr], Tuple[relay.Expr], TupleWrapper[relay.Expr])
        The list of quantized tensors.

    input_scales : List[relay.Expr]
        The list of scales of input quantized tensors.

    input_zero_points : List[relay.Expr]
        The list of zero points of input quantized tensors.

    output_scale : relay.Expr
        The scale of the output quantized tensor.

    output_zero_point : relay.Expr
        The zero point of the output quantized tensor.

    axis : int
        The axis along which the tensors are concatenated.

    Returns
    -------
    result: relay.Expr
        The concatenated quantized tensor.
    """

    if isinstance(data, (list, tuple)):
        data = Tuple(data)
    elif isinstance(data, TupleWrapper):
        data = data.tuple_value
    if not isinstance(axis, int):
        raise ValueError("For now, we only support integer axis")
    input_scales = list(input_scales)
    input_zero_points = list(input_zero_points)

    return _make.concatenate(
        data, Tuple(input_scales), Tuple(input_zero_points), output_scale, output_zero_point, axis
    )


def conv2d(
    data,
    kernel,
    input_zero_point,
    kernel_zero_point,
    input_scale,
    kernel_scale,
    kernel_size,
    channels,
    strides=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    groups=1,
    data_layout="NCHW",
    kernel_layout="OIHW",
    out_layout="",
    out_dtype="int32",
):
    r"""Quantized 2D convolution.

    This operator convolves quantized data with quantized kernel.
    If doing Per-channel quantization, qnn expects the kernel_zero_scale
    and optionally the kernel_zero_point will be 1-D vectors instead of scalars.
    The scale of the output quantized tensor is the product of the kernel_scale and
    input_scale of the input quantized tensors. The zero point of the output
    quantized tensor is 0. By default, the dtype of output is int32. Please also
    refer to Requantize operator to understand how to scale back the int32
    output to (u)int8.

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    kernel : tvm.relay.Expr
        The kernel expressions.

    input_zero_point: tvm.relay.Expr
           The zero point of the data distribution.

    kernel_zero_point: tvm.relay.Expr
           The zero point of the quantized_kernel distribution.

    input_scale: tvm.relay.Expr
           The scale for the input tensor. The scale for the input tensor is
           stored purely for convenience here. See more commentary below.

    kernel_scale: tvm.relay.Expr
           The scale for the weight tensor. The scale for the weight tensor is
           stored for access to this during relay. This information is not
           needed in the pass pipeline after qnn.conv2d is lowered to the
           sequence of steps as in nn.conv2d. See also input_scale in Requantize.

    kernel_size : tuple of int
        The spatial width and height of the convolution kernel.

    channels : int
        Number of output channels of this convolution.

    strides : tuple of int, optional
        The strides of convolution.

    padding : tuple of int, optional
        The padding of convolution on both sides of inputs before convolution.

    dilation : tuple of int, optional
        Specifies the dilation rate to be used for dilated convolution.

    groups : int, optional
        Number of groups for grouped convolution.

    data_layout : str, optional
        Layout of the input.

    kernel_layout : str, optional
        Layout of the kernel.

    out_layout : str, optional
        Layout of the output, by default, out_layout is the same as data_layout

    out_dtype : str, optional
        Specifies the output data type for mixed precision conv2d.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """

    padding = get_pad_tuple2d(padding)
    return _make.conv2d(
        data,
        kernel,
        input_zero_point,
        kernel_zero_point,
        input_scale,
        kernel_scale,
        strides,
        padding,
        dilation,
        groups,
        channels,
        kernel_size,
        data_layout,
        kernel_layout,
        out_layout,
        out_dtype,
    )


def conv2d_transpose(
    data,
    weight,
    input_zero_point,
    kernel_zero_point,
    input_scale,
    kernel_scale,
    strides=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    groups=1,
    channels=None,
    kernel_size=None,
    data_layout="NCHW",
    kernel_layout="IOHW",
    out_layout="",
    output_padding=(0, 0),
    out_dtype="int32",
):
    """This operator deconvolves quantized data with quantized kernel. The scale of
    the output quantized tensor is the product of the kernel_scale and
    input_scale of the input quantized tensors. The zero point of the output
    quantized tensor is 0. By default, the dtype of output is int32. Please also
    refer to Requantize operator to understand how to scale back the int32
    output to (u)int8.

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    weight : tvm.relay.Expr
        The weight expressions.

    input_zero_point: tvm.relay.Expr
           The zero point of the data distribution.

    kernel_zero_point: tvm.relay.Expr
           The zero point of the quantized_kernel distribution.

    input_scale: tvm.relay.Expr
           The scale for the input tensor. The scale for the input tensor is
           stored purely for convenience here. See more commentary below.

    kernel_scale: tvm.relay.Expr
           The scale for the weight tensor. The scale for the weight tensor is
           stored for access to this during relay. This information is not
           needed in the pass pipeline after qnn.conv2d_transpose is lowered to the
           sequence of steps as in nn.conv2d_transpose. See also input_scale in Requantize.

    strides : Tuple[int], optional
        The strides of convolution.

    padding : Tuple[int], optional
        The padding of convolution.

    dilation : Tuple[int], optional
        Specifies the dilation rate to be used for dilated convolution.

    channels : int, optional
        Number of output channels of this convolution.

    kernel_size : tuple of int, optional
        The spatial dimensions of the convolution kernel.

    groups : int, optional
        Number of groups for grouped convolution.

    data_layout : str, optional
        Layout of the input.

    kernel_layout : str, optional
        Layout of the weight.

    out_layout : Optional[str]
        Layout of the output, by default, out_layout is the same as data_layout

    output_padding : Tuple[int], optional
        Used to identify the padding within the output shape
        (only used in training, where transpose_conv represents the gradient of a convolution )

    out_dtype : str, optional
        Specifies the output data type for mixed precision conv2d.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    # convert 2-way padding to 4-way padding
    padding = get_pad_tuple2d(padding)
    return _make.conv2d_transpose(
        data,
        weight,
        input_zero_point,
        kernel_zero_point,
        input_scale,
        kernel_scale,
        strides,
        padding,
        dilation,
        groups,
        channels,
        kernel_size,
        data_layout,
        kernel_layout,
        out_layout,
        output_padding,
        out_dtype,
    )


def add(
    lhs,
    rhs,
    lhs_scale,
    lhs_zero_point,
    rhs_scale,
    rhs_zero_point,
    output_scale,
    output_zero_point,
    lhs_axis=-1,
    rhs_axis=-1,
):
    """Quantized addition with numpy-style broadcasting.

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side quantized input data.

    rhs : relay.Expr
        The right hand side quantized input data.

    lhs_scale: relay.Expr
        The scale of the lhs quantized expr.

    lhs_zero_point: relay.Expr
       The zero point of lhs quantized expr.

    rhs_scale: relay.Expr
        The scale of the rhs quantized expr.

    rhs_zero_point: relay.Expr
       The zero point of rhs quantized expr.

    output_scale: relay.Expr
        The scale of the output quantized expr.

    output_zero_point: relay.Expr
       The zero point of output quantized expr.

    lhs_axis: int
        The channel axis for lhs quantization. Default value is -1 which corresponds
        to the last axis.

    rhs_axis: int
        The channel axis for rhs quantization. Default value is -1 which corresponds
        to the last axis.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.add(
        lhs,
        rhs,
        lhs_scale,
        lhs_zero_point,
        rhs_scale,
        rhs_zero_point,
        output_scale,
        output_zero_point,
        lhs_axis,
        rhs_axis,
    )


def dense(
    data,
    weight,
    input_zero_point,
    kernel_zero_point,
    input_scale,
    kernel_scale,
    units,
    out_dtype="int32",
):
    """Qnn Dense operator.
    Applies a quantized linear transformation

     .. math::

     `Y = X * W`

    If doing Per-channel quantization, qnn expects the kernel_zero_scale
    and optionally the kernel_zero_point will be 1-D vectors instead of scalars.

    Parameters
    ----------
    data : tvm.relay.Expr
        The quantized input data to the operator.
    weight : tvm.relay.Expr
        The quantized weight expressions.
    input_zero_point: tvm.relay.Expr
        The input zero point.
    kernel_zero_point: tvm.relay.Expr
        The kernel zero point.
    input_scale: tvm.relay.Expr
        The scale for the input tensor.
    kernel_scale: tvm.relay.Expr
        The scale for the weight tensor. The scale for the weight tensor is
        stored for access to this during relay. This information is not
        needed in the pass pipeline after qnn.conv2d is lowered to the
        sequence of steps as in nn.conv2d. See also input_scale in Requantize.
    units : int
        Number of hidden units of the dense transformation.
    out_dtype : str, optional
        Specifies the output data type for mixed precision dense can be int32 or int16.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """

    return _make.dense(
        data,
        weight,
        input_zero_point,
        kernel_zero_point,
        input_scale,
        kernel_scale,
        units,
        out_dtype,
    )


def mul(
    lhs,
    rhs,
    lhs_scale,
    lhs_zero_point,
    rhs_scale,
    rhs_zero_point,
    output_scale,
    output_zero_point,
    lhs_axis=-1,
    rhs_axis=-1,
):
    """Quantized multiplication with numpy-style broadcasting.

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side quantized input data.

    rhs : relay.Expr
        The right hand side quantized input data.

    lhs_scale: relay.Expr
        The scale of the lhs quantized expr.

    lhs_zero_point: relay.Expr
       The zero point of lhs quantized expr.

    rhs_scale: relay.Expr
        The scale of the rhs quantized expr.

    rhs_zero_point: relay.Expr
       The zero point of rhs quantized expr.

    output_scale: relay.Expr
        The scale of the output quantized expr.

    output_zero_point: relay.Expr
       The zero point of output quantized expr.

    lhs_axis: int
        The channel axis for lhs quantization. Default value is -1 which corresponds
        to the last axis.

    rhs_axis: int
        The channel axis for rhs quantization. Default value is -1 which corresponds
        to the last axis.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.mul(
        lhs,
        rhs,
        lhs_scale,
        lhs_zero_point,
        rhs_scale,
        rhs_zero_point,
        output_scale,
        output_zero_point,
        lhs_axis,
        rhs_axis,
    )


def tanh(x, scale, zero_point, output_scale, output_zero_point):
    """Quantized tanh.

    Parameters
    ----------
    x : relay.Expr
        The quantized input tensor.

    scale: relay.Expr
        The scale of the quantized expr.

    zero_point: relay.Expr
       The zero point of quantized expr.

    output_scale: relay.Expr
        The scale of the output quantized expr.

    output_zero_point: relay.Expr
       The zero point of output quantized expr.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.tanh(
        x,
        scale,
        zero_point,
        output_scale,
        output_zero_point,
    )


def exp(x, scale, zero_point, output_scale, output_zero_point):
    """Quantized exponential function.

    Parameters
    ----------
    x : relay.Expr
        The quantized input tensor.

    scale: relay.Expr
        The scale of the quantized expr.

    zero_point: relay.Expr
       The zero point of quantized expr.

    output_scale: relay.Expr
        The scale of the output quantized expr.

    output_zero_point: relay.Expr
       The zero point of output quantized expr.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.exp(
        x,
        scale,
        zero_point,
        output_scale,
        output_zero_point,
    )


def sqrt(x, scale, zero_point, output_scale, output_zero_point):
    """Quantized square root.

    Parameters
    ----------
    x : relay.Expr
        The quantized input tensor.

    scale: relay.Expr
        The scale of the quantized expr.

    zero_point: relay.Expr
       The zero point of quantized expr.

    output_scale: relay.Expr
        The scale of the output quantized expr.

    output_zero_point: relay.Expr
       The zero point of output quantized expr.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.sqrt(
        x,
        scale,
        zero_point,
        output_scale,
        output_zero_point,
    )


def rsqrt(x, scale, zero_point, output_scale, output_zero_point):
    """Quantized reciprocal square root.

    Parameters
    ----------
    x : relay.Expr
        The quantized input tensor.

    scale: relay.Expr
        The scale of the quantized expr.

    zero_point: relay.Expr
       The zero point of quantized expr.

    output_scale: relay.Expr
        The scale of the output quantized expr.

    output_zero_point: relay.Expr
       The zero point of output quantized expr.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.rsqrt(
        x,
        scale,
        zero_point,
        output_scale,
        output_zero_point,
    )


def erf(x, scale, zero_point, output_scale, output_zero_point):
    """Quantized error function.

    Parameters
    ----------
    x : relay.Expr
        The quantized input tensor.

    scale: relay.Expr
        The scale of the quantized expr.

    zero_point: relay.Expr
       The zero point of quantized expr.

    output_scale: relay.Expr
        The scale of the output quantized expr.

    output_zero_point: relay.Expr
       The zero point of output quantized expr.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.erf(
        x,
        scale,
        zero_point,
        output_scale,
        output_zero_point,
    )


def sigmoid(x, scale, zero_point, output_scale, output_zero_point):
    """Quantized sigmoid.

    Parameters
    ----------
    x : relay.Expr
        The quantized input tensor.

    scale: relay.Expr
        The scale of the quantized expr.

    zero_point: relay.Expr
       The zero point of quantized expr.

    output_scale: relay.Expr
        The scale of the output quantized expr.

    output_zero_point: relay.Expr
       The zero point of output quantized expr.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.sigmoid(
        x,
        scale,
        zero_point,
        output_scale,
        output_zero_point,
    )


def hardswish(x, scale, zero_point, output_scale, output_zero_point):
    """Quantized hardswish.

    Parameters
    ----------
    x : relay.Expr
        The quantized input tensor.

    scale: relay.Expr
        The scale of the quantized expr.

    zero_point: relay.Expr
       The zero point of quantized expr.

    output_scale: relay.Expr
        The scale of the output quantized expr.

    output_zero_point: relay.Expr
       The zero point of output quantized expr.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.hardswish(
        x,
        scale,
        zero_point,
        output_scale,
        output_zero_point,
    )


def log(x, scale, zero_point, output_scale, output_zero_point):
    """Quantized log.

    Parameters
    ----------
    x : relay.Expr
        The quantized input tensor.

    scale: relay.Expr
        The scale of the quantized expr.

    zero_point: relay.Expr
       The zero point of quantized expr.

    output_scale: relay.Expr
        The scale of the output quantized expr.

    output_zero_point: relay.Expr
       The zero point of output quantized expr.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.log(
        x,
        scale,
        zero_point,
        output_scale,
        output_zero_point,
    )


def subtract(
    lhs,
    rhs,
    lhs_scale,
    lhs_zero_point,
    rhs_scale,
    rhs_zero_point,
    output_scale,
    output_zero_point,
    lhs_axis=-1,
    rhs_axis=-1,
):
    """Quantized subtraction with numpy-style broadcasting.

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side quantized input data.

    rhs : relay.Expr
        The right hand side quantized input data.

    lhs_scale: relay.Expr
        The scale of the lhs quantized expr.

    lhs_zero_point: relay.Expr
       The zero point of lhs quantized expr.

    rhs_scale: relay.Expr
        The scale of the rhs quantized expr.

    rhs_zero_point: relay.Expr
       The zero point of rhs quantized expr.

    output_scale: relay.Expr
        The scale of the output quantized expr.

    output_zero_point: relay.Expr
       The zero point of output quantized expr.

    lhs_axis: int
        The channel axis for lhs quantization. Default value is -1 which corresponds
        to the last axis.

    rhs_axis: int
        The channel axis for rhs quantization. Default value is -1 which corresponds
        to the last axis.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.subtract(
        lhs,
        rhs,
        lhs_scale,
        lhs_zero_point,
        rhs_scale,
        rhs_zero_point,
        output_scale,
        output_zero_point,
        lhs_axis,
        rhs_axis,
    )


def batch_matmul(x, y, x_zero_point, y_zero_point, x_scale, y_scale, out_dtype="int32"):
    r"""
    Computes batch matrix multiplication of `x` and `y` when `x` and `y` are data
    in batch.

    .. math::

        \mbox{batch_matmul}(x, y)[i, :, :] = \mbox{matmul}(x[i, :, :], y[i, :, :]^T)

    Parameters
    ----------
    x : tvm.relay.Expr
        The first quantized input.
        A quantized tensor is represented in following manner
        `A = scale_a x (QA - zp_A)`
        where QA is quantized tensor, scale_a and zp_A are quantization
        params.
    y : tvm.relay.Expr
        The second quantized input.
    x_zero_point: tvm.relay.Expr
        The first input zero point.
    y_zero_point: tvm.relay.Expr
        The second input zero point.
    x_scale: tvm.relay.Expr
        The scale for the first input tensor.
    y_scale: tvm.relay.Expr
        The scale for the second input tensor.
    out_dtype : str, optional
        Specifies the output data type for mixed precision dense can be int32 or int16.

    Returns
    -------
    result: tvm.relay.Expr
        The computed result.
    """
    return _make.batch_matmul(x, y, x_zero_point, y_zero_point, x_scale, y_scale, out_dtype)


def csi_concatenate(data, axis, q_params, layer_name=""):
    """Concatenate the quantized input tensors along the given axis.

    Parameters
    ----------
    data : Union(List[relay.Expr], Tuple[relay.Expr])
        The list of quantized tensors.

    input_scales : List[float32]
        The list of scales of input quantized tensors.

    input_zero_points : List[int32]
        The list of zero points of input quantized tensors.

    output_scale : float32
        The scale of the output quantized tensor.

    output_zero_point : int32
        The zero point of the output quantized tensor.

    axis : int
        The axis along which the tensors are concatenated.

    Returns
    -------
    result: relay.Expr
        The concatenated quantized tensor.
    """

    data = list(data)
    if not data:
        raise ValueError("relay.concatenate requires data to be non-empty.")
    if not isinstance(axis, int):
        raise ValueError("For now, we only support integer axis")

    return _make.CSIConcatenate(Tuple(data), axis, q_params, layer_name)


def csi_conv2d(
    data,
    weight,
    bias,
    strides,
    padding,
    dilation,
    groups,
    channels,
    kernel_size,
    data_layout,
    kernel_layout,
    out_layout,
    out_dtype,
    q_params,
    layer_name="",
):
    r"""Quantized 2D convolution.

    quantized tensor is 0. By default, the dtype of output is int32. Please also
    output to (u)int8.

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    kernel : tvm.relay.Expr
        The kernel expressions.
    strides : tuple of int, optional
        The strides of convolution.

    padding : tuple of int, optional
        The padding of convolution on both sides of inputs before convolution.

    dilation : tuple of int, optional
        Specifies the dilation rate to be used for dilated convolution.

    groups : int, optional
        Number of groups for grouped convolution.

    channels : int, optional
        Number of output channels of this convolution.

    kernel_size : tuple of int, optional
        The spatial of the convolution kernel.

    data_layout : str, optional
        Layout of the input.

    kernel_layout : str, optional
        Layout of the kernel.

    out_layout : str, optional
        Layout of the output, by default, out_layout is the same as data_layout

    out_dtype : str, optional
        Specifies the output data type for mixed precision conv2d.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    return _make.CSIConv2D(
        data,
        weight,
        bias,
        strides,
        padding,
        dilation,
        groups,
        channels,
        kernel_size,
        data_layout,
        kernel_layout,
        out_layout,
        out_dtype,
        q_params,
        layer_name,
    )


def csi_conv1d(
    data,
    weight,
    bias,
    strides=1,
    padding=0,
    dilation=1,
    groups=1,
    channels=None,
    kernel_size=None,
    data_layout="NCW",
    kernel_layout="OIW",
    out_layout="",
    out_dtype="",
    q_params=None,
    layer_name="",
):
    r"""1D convolution.

    This operator takes the weight as the convolution kernel
    and convolves it with data to produce an output.


    In the default case, where the data_layout is `NCW`
    and kernel_layout is `OIW`, conv1d takes in
    a data Tensor with shape `(batch_size, in_channels, width)`,
    and a weight Tensor with shape `(channels, in_channels, kernel_size)`
    to produce an output Tensor with the following rule:

    .. math::

        \mbox{out}[b, c, w] = \sum_{dw, k}
           \mbox{data}[b, k, \mbox{strides}[0] * w + dw] *
           \mbox{weight}[c, k, dw]

    Padding and dilation are applied to data and weight respectively before the computation.
    This operator accepts data layout specification.
    Semantically, the operator will convert the layout to the canonical layout
    (`NCW` for data and `OIW` for weight), perform the computation,
    then convert to the out_layout.


    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    weight : tvm.relay.Expr
        The weight expressions.

    strides : Optional[int, Tuple[int]]
        The strides of convolution.

    padding : Optional[int, Tuple[int]]
        The padding of convolution on both sides of the input before convolution.

    dilation : Optional[int, Tuple[int]]
        Specifies the dilation rate to be used for dilated convolution.

    groups : Optional[int]
        Currently unused for 1D convolution.

    channels : Optional[int]
        Number of output channels of this convolution.

    kernel_size : Optional[int, Tuple[int]]
        The spatial dimension of the convolution kernel.

    data_layout : Optional[str]
        Layout of the input.

    kernel_layout : Optional[str]
        Layout of the weight.

    out_layout : Optional[str]
        Layout of the output, by default, out_layout is the same as data_layout

    out_dtype : Optional[str]
        Specifies the output data type for mixed precision conv2d.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,)
    if isinstance(strides, int):
        strides = (strides,)
    if isinstance(dilation, int):
        dilation = (dilation,)
    padding = get_pad_tuple1d(padding)

    return _make.CSIConv1D(
        data,
        weight,
        bias,
        strides,
        padding,
        dilation,
        groups,
        channels,
        kernel_size,
        data_layout,
        kernel_layout,
        out_layout,
        out_dtype,
        q_params,
        layer_name,
    )


def csi_conv2d_channel(
    data,
    weight,
    bias,
    strides,
    padding,
    dilation,
    groups,
    channels,
    kernel_size,
    data_layout,
    kernel_layout,
    out_layout,
    out_dtype,
    q_params,
    layer_name="",
):
    r"""Quantized 2D convolution.

    quantized tensor is 0. By default, the dtype of output is int32. Please also
    output to (u)int8.

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    kernel : tvm.relay.Expr
        The kernel expressions.
    strides : tuple of int, optional
        The strides of convolution.

    padding : tuple of int, optional
        The padding of convolution on both sides of inputs before convolution.

    dilation : tuple of int, optional
        Specifies the dilation rate to be used for dilated convolution.

    groups : int, optional
        Number of groups for grouped convolution.

    channels : int, optional
        Number of output channels of this convolution.

    kernel_size : tuple of int, optional
        The spatial of the convolution kernel.

    data_layout : str, optional
        Layout of the input.

    kernel_layout : str, optional
        Layout of the kernel.

    out_layout : str, optional
        Layout of the output, by default, out_layout is the same as data_layout

    out_dtype : str, optional
        Specifies the output data type for mixed precision conv2d.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    return _make.CSIConv2DChannel(
        data,
        weight,
        bias,
        strides,
        padding,
        dilation,
        groups,
        channels,
        kernel_size,
        data_layout,
        kernel_layout,
        out_layout,
        out_dtype,
        q_params,
        layer_name,
    )


def csi_conv2d_relu_channel(
    data,
    weight,
    bias,
    strides,
    padding,
    dilation,
    groups,
    channels,
    kernel_size,
    data_layout,
    kernel_layout,
    out_layout,
    out_dtype,
    q_params,
    layer_name="",
):
    r"""Quantized 2D convolution.

    quantized tensor is 0. By default, the dtype of output is int32. Please also
    output to (u)int8.

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    kernel : tvm.relay.Expr
        The kernel expressions.
    strides : tuple of int, optional
        The strides of convolution.

    padding : tuple of int, optional
        The padding of convolution on both sides of inputs before convolution.

    dilation : tuple of int, optional
        Specifies the dilation rate to be used for dilated convolution.

    groups : int, optional
        Number of groups for grouped convolution.

    channels : int, optional
        Number of output channels of this convolution.

    kernel_size : tuple of int, optional
        The spatial of the convolution kernel.

    data_layout : str, optional
        Layout of the input.

    kernel_layout : str, optional
        Layout of the kernel.

    out_layout : str, optional
        Layout of the output, by default, out_layout is the same as data_layout

    out_dtype : str, optional
        Specifies the output data type for mixed precision conv2d.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    return _make.CSIConv2DReluChannel(
        data,
        weight,
        bias,
        strides,
        padding,
        dilation,
        groups,
        channels,
        kernel_size,
        data_layout,
        kernel_layout,
        out_layout,
        out_dtype,
        q_params,
        layer_name,
    )


def csi_conv2d_relu6_channel(
    data,
    weight,
    bias,
    strides,
    padding,
    dilation,
    groups,
    channels,
    kernel_size,
    data_layout,
    kernel_layout,
    out_layout,
    out_dtype,
    q_params,
    layer_name="",
):
    r"""Quantized 2D convolution.

    quantized tensor is 0. By default, the dtype of output is int32. Please also
    output to (u)int8.

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    kernel : tvm.relay.Expr
        The kernel expressions.
    strides : tuple of int, optional
        The strides of convolution.

    padding : tuple of int, optional
        The padding of convolution on both sides of inputs before convolution.

    dilation : tuple of int, optional
        Specifies the dilation rate to be used for dilated convolution.

    groups : int, optional
        Number of groups for grouped convolution.

    channels : int, optional
        Number of output channels of this convolution.

    kernel_size : tuple of int, optional
        The spatial of the convolution kernel.

    data_layout : str, optional
        Layout of the input.

    kernel_layout : str, optional
        Layout of the kernel.

    out_layout : str, optional
        Layout of the output, by default, out_layout is the same as data_layout

    out_dtype : str, optional
        Specifies the output data type for mixed precision conv2d.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    return _make.CSIConv2DRelu6Channel(
        data,
        weight,
        bias,
        strides,
        padding,
        dilation,
        groups,
        channels,
        kernel_size,
        data_layout,
        kernel_layout,
        out_layout,
        out_dtype,
        q_params,
        layer_name,
    )


def csi_conv3d(
    data,
    weight,
    bias,
    strides,
    padding,
    dilation,
    groups,
    channels,
    kernel_size,
    data_layout,
    kernel_layout,
    out_layout,
    out_dtype,
    q_params,
    layer_name="",
):
    r"""Quantized 3D convolution.

    quantized tensor is 0. By default, the dtype of output is int32. Please also
    output to (u)int8.

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    kernel : tvm.relay.Expr
        The kernel expressions.
    strides : tuple of int, optional
        The strides of convolution.

    padding : tuple of int, optional
        The padding of convolution on both sides of inputs before convolution.

    dilation : tuple of int, optional
        Specifies the dilation rate to be used for dilated convolution.

    groups : int, optional
        Number of groups for grouped convolution.

    channels : int, optional
        Number of output channels of this convolution.

    kernel_size : tuple of int, optional
        The spatial of the convolution kernel.

    data_layout : str, optional
        Layout of the input.

    kernel_layout : str, optional
        Layout of the kernel.

    out_layout : str, optional
        Layout of the output, by default, out_layout is the same as data_layout

    out_dtype : str, optional
        Specifies the output data type for mixed precision conv2d.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    return _make.CSIConv3D(
        data,
        weight,
        bias,
        strides,
        padding,
        dilation,
        groups,
        channels,
        kernel_size,
        data_layout,
        kernel_layout,
        out_layout,
        out_dtype,
        q_params,
        layer_name,
    )


def csi_dilation2d(
    data,
    weight,
    strides,
    padding,
    dilations,
    data_layout,
    kernel_layout,
    out_dtype,
    q_params,
    layer_name="",
):
    r"""Quantized dilation2d.
    - **data**: This depends on the `layout` parameter. Input is 4D array of shape
                (batch_size, in_channels, height, width) if `layout` is `NCHW`.
    - **weight**: (in_channels, height, width)
    - **out**:  This depends on the `layout` parameter. Output is 4D array of shape
                (batch_size, channels, out_height, out_width) if `layout` is `NCHW`.
    """
    return _make.CSIDilation2D(
        data,
        weight,
        strides,
        padding,
        dilations,
        data_layout,
        kernel_layout,
        out_dtype,
        q_params,
        layer_name,
    )


def csi_conv2d_relu(
    data,
    weight,
    bias,
    strides,
    padding,
    dilation,
    groups,
    channels,
    kernel_size,
    data_layout,
    kernel_layout,
    out_layout,
    out_dtype,
    q_params,
    layer_name="",
):
    r"""Quantized 2D convolution.

    quantized tensor is 0. By default, the dtype of output is int32. Please also
    output to (u)int8.

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    kernel : tvm.relay.Expr
        The kernel expressions.
    strides : tuple of int, optional
        The strides of convolution.

    padding : tuple of int, optional
        The padding of convolution on both sides of inputs before convolution.

    dilation : tuple of int, optional
        Specifies the dilation rate to be used for dilated convolution.

    groups : int, optional
        Number of groups for grouped convolution.

    channels : int, optional
        Number of output channels of this convolution.

    kernel_size : tuple of int, optional
        The spatial of the convolution kernel.

    data_layout : str, optional
        Layout of the input.

    kernel_layout : str, optional
        Layout of the kernel.

    out_layout : str, optional
        Layout of the output, by default, out_layout is the same as data_layout

    out_dtype : str, optional
        Specifies the output data type for mixed precision conv2d.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    return _make.CSIConv2DRelu(
        data,
        weight,
        bias,
        strides,
        padding,
        dilation,
        groups,
        channels,
        kernel_size,
        data_layout,
        kernel_layout,
        out_layout,
        out_dtype,
        q_params,
        layer_name,
    )


def csi_conv2d_relu6(
    data,
    weight,
    bias,
    strides,
    padding,
    dilation,
    groups,
    channels,
    kernel_size,
    data_layout,
    kernel_layout,
    out_layout,
    out_dtype,
    q_params,
    layer_name="",
):
    r"""Quantized 2D convolution.

    quantized tensor is 0. By default, the dtype of output is int32. Please also
    output to (u)int8.

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    kernel : tvm.relay.Expr
        The kernel expressions.
    strides : tuple of int, optional
        The strides of convolution.

    padding : tuple of int, optional
        The padding of convolution on both sides of inputs before convolution.

    dilation : tuple of int, optional
        Specifies the dilation rate to be used for dilated convolution.

    groups : int, optional
        Number of groups for grouped convolution.

    channels : int, optional
        Number of output channels of this convolution.

    kernel_size : tuple of int, optional
        The spatial of the convolution kernel.

    data_layout : str, optional
        Layout of the input.

    kernel_layout : str, optional
        Layout of the kernel.

    out_layout : str, optional
        Layout of the output, by default, out_layout is the same as data_layout

    out_dtype : str, optional
        Specifies the output data type for mixed precision conv2d.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    return _make.CSIConv2DRelu6(
        data,
        weight,
        bias,
        strides,
        padding,
        dilation,
        groups,
        channels,
        kernel_size,
        data_layout,
        kernel_layout,
        out_layout,
        out_dtype,
        q_params,
        layer_name,
    )


def csi_deconv2d(
    data,
    weight,
    bias,
    strides,
    padding,
    dilation,
    groups,
    channels,
    kernel_size,
    data_layout,
    kernel_layout,
    out_layout,
    output_padding,
    out_dtype,
    q_params,
    layer_name="",
):
    r"""Quantized 2D deconvolution.

    quantized tensor is 0. By default, the dtype of output is int32. Please also
    output to (u)int8.

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    kernel : tvm.relay.Expr
        The kernel expressions.
    strides : tuple of int, optional
        The strides of convolution.

    padding : tuple of int, optional
        The padding of convolution on both sides of inputs before convolution.

    dilation : tuple of int, optional
        Specifies the dilation rate to be used for dilated convolution.

    groups : int, optional
        Number of groups for grouped convolution.

    channels : int, optional
        Number of output channels of this convolution.

    kernel_size : tuple of int, optional
        The spatial of the convolution kernel.

    data_layout : str, optional
        Layout of the input.

    kernel_layout : str, optional
        Layout of the kernel.

    out_layout : str, optional
        Layout of the output, by default, out_layout is the same as data_layout

    out_dtype : str, optional
        Specifies the output data type for mixed precision conv2d.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    return _make.CSIDeConv2D(
        data,
        weight,
        bias,
        strides,
        padding,
        dilation,
        groups,
        channels,
        kernel_size,
        data_layout,
        kernel_layout,
        out_layout,
        output_padding,
        out_dtype,
        q_params,
        layer_name,
    )


def csi_deconv3d(
    data,
    weight,
    bias,
    strides,
    padding,
    dilation,
    groups,
    channels,
    kernel_size,
    data_layout,
    kernel_layout,
    out_layout,
    output_padding,
    out_dtype,
    q_params,
    layer_name="",
):
    r"""Quantized 3D deconvolution.

    quantized tensor is 0. By default, the dtype of output is int32. Please also
    output to (u)int8.

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    kernel : tvm.relay.Expr
        The kernel expressions.
    strides : tuple of int, optional
        The strides of convolution.

    padding : tuple of int, optional
        The padding of convolution on both sides of inputs before convolution.

    dilation : tuple of int, optional
        Specifies the dilation rate to be used for dilated convolution.

    groups : int, optional
        Number of groups for grouped convolution.

    channels : int, optional
        Number of output channels of this convolution.

    kernel_size : tuple of int, optional
        The spatial of the convolution kernel.

    data_layout : str, optional
        Layout of the input.

    kernel_layout : str, optional
        Layout of the kernel.

    out_layout : str, optional
        Layout of the output, by default, out_layout is the same as data_layout

    out_dtype : str, optional
        Specifies the output data type for mixed precision conv2d.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    return _make.CSIDeConv3D(
        data,
        weight,
        bias,
        strides,
        padding,
        dilation,
        groups,
        channels,
        kernel_size,
        data_layout,
        kernel_layout,
        out_layout,
        output_padding,
        out_dtype,
        q_params,
        layer_name,
    )


def csi_add(lhs, rhs, q_params, layer_name=""):
    """Quantized addition with numpy-style broadcasting.

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side quantized input data.

    rhs : relay.Expr
        The right hand side quantized input data.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIAdd(lhs, rhs, q_params, layer_name)


def csi_subtract(lhs, rhs, q_params, layer_name=""):
    """Quantized subtract with numpy-style broadcasting.

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side quantized input data.

    rhs : relay.Expr
        The right hand side quantized input data.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSISubtract(lhs, rhs, q_params, layer_name)


def csi_bias_add(lhs, rhs, q_params, layer_name=""):
    """Quantized addition with numpy-style broadcasting.

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side quantized input data.

    rhs : relay.Expr
        The right hand side quantized input data.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIBiasAdd(lhs, rhs, q_params, layer_name)


def csi_mul(lhs, rhs, q_params, layer_name=""):
    """Quantized multiplication with numpy-style broadcasting.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    weight : relay.Expr
        The quantized weight data.

    bias : relay.Expr
        The quantized bias data.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIMul(lhs, rhs, q_params, layer_name)


def csi_div(lhs, rhs, q_params, layer_name=""):
    """Quantized divide with numpy-style broadcasting.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    weight : relay.Expr
        The quantized weight data.

    bias : relay.Expr
        The quantized bias data.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIDiv(lhs, rhs, q_params, layer_name)


def csi_power(lhs, rhs, q_params, layer_name=""):
    """Quantized power with numpy-style broadcasting.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    weight : relay.Expr
        The quantized weight data.

    bias : relay.Expr
        The quantized bias data.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIPower(lhs, rhs, q_params, layer_name)


def csi_mod(lhs, rhs, q_params, layer_name=""):
    """Quantized power with numpy-style broadcasting.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    weight : relay.Expr
        The quantized weight data.

    bias : relay.Expr
        The quantized bias data.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIMod(lhs, rhs, q_params, layer_name)


def csi_dense(data, weight, bias, units, out_dtype, q_params, layer_name=""):
    """Qnn Dense operator.
    Applies a quantized linear transformation

     .. math::

     `Y = X * W`

    Parameters
    ----------
    data : tvm.relay.Expr
        The quantized input data to the operator.
    weight : tvm.relay.Expr
        The quantized weight expressions.
    units : int, optional
        Number of hidden units of the dense transformation.
    out_dtype : str, optional
        Specifies the output data type for mixed precision dense can be int32 or int16.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    return _make.CSIDense(data, weight, bias, units, out_dtype, q_params, layer_name)


def csi_sin(data, out_dtype, q_params, layer_name=""):
    """Quantized activation sin.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.
    out_dtype : str
        Specifies the output data type for mixed precision dense can be uint8.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSISin(data, out_dtype, q_params, layer_name)


def csi_cos(data, out_dtype, q_params, layer_name=""):
    """Quantized activation cos.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.
    out_dtype : str
        Specifies the output data type for mixed precision dense can be uint8.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSICos(data, out_dtype, q_params, layer_name)


def csi_tan(data, out_dtype, q_params, layer_name=""):
    """Quantized activation tan.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.
    out_dtype : str
        Specifies the output data type for mixed precision dense can be uint8.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSITan(data, out_dtype, q_params, layer_name)


def csi_asin(data, out_dtype, q_params, layer_name=""):
    """Quantized activation asin.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.
    out_dtype : str
        Specifies the output data type for mixed precision dense can be uint8.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIAsin(data, out_dtype, q_params, layer_name)


def csi_acos(data, out_dtype, q_params, layer_name=""):
    """Quantized activation acos.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.
    out_dtype : str
        Specifies the output data type for mixed precision dense can be uint8.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIAcos(data, out_dtype, q_params, layer_name)


def csi_atan(data, out_dtype, q_params, layer_name=""):
    """Quantized activation atan.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.
    out_dtype : str
        Specifies the output data type for mixed precision dense can be uint8.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIAtan(data, out_dtype, q_params, layer_name)


def csi_sinh(data, out_dtype, q_params, layer_name=""):
    """Quantized activation sinh.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.
    out_dtype : str
        Specifies the output data type for mixed precision dense can be uint8.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSISinh(data, out_dtype, q_params, layer_name)


def csi_cosh(data, out_dtype, q_params, layer_name=""):
    """Quantized activation cosh.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.
    out_dtype : str
        Specifies the output data type for mixed precision dense can be uint8.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSICosh(data, out_dtype, q_params, layer_name)


def csi_tanh(data, out_dtype, q_params, layer_name=""):
    """Quantized activation tanh.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.
    out_dtype : str
        Specifies the output data type for mixed precision dense can be uint8.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSITanh(data, out_dtype, q_params, layer_name)


def csi_asinh(data, out_dtype, q_params, layer_name=""):
    """Quantized activation asinh.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.
    out_dtype : str
        Specifies the output data type for mixed precision dense can be uint8.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIAsinh(data, out_dtype, q_params, layer_name)


def csi_acosh(data, out_dtype, q_params, layer_name=""):
    """Quantized activation acosh.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.
    out_dtype : str
        Specifies the output data type for mixed precision dense can be uint8.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIAcosh(data, out_dtype, q_params, layer_name)


def csi_atanh(data, out_dtype, q_params, layer_name=""):
    """Quantized activation atanh.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.
    out_dtype : str
        Specifies the output data type for mixed precision dense can be uint8.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIAtanh(data, out_dtype, q_params, layer_name)


def csi_relu(data, out_dtype, q_params, layer_name=""):
    """Quantized activation relu.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.
    out_dtype : str
        Specifies the output data type for mixed precision dense can be uint8.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIRelu(data, out_dtype, q_params, layer_name)


def csi_leaky_relu(data, alpha, out_dtype, q_params, layer_name=""):
    """Quantized activation relu.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.alpha
    out_dtype : str
        Specifies the output data type for mixed precision dense can be uint8.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSILeakyRelu(data, alpha, out_dtype, q_params, layer_name)


def csi_prelu(data, alpha, axis, out_dtype, q_params, layer_name=""):
    """Quantized activation relu.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    alpha : relay.Expr
        The quantized alpha.
    out_dtype : str
        Specifies the output data type for mixed precision dense can be uint8.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIPRelu(data, alpha, axis, out_dtype, q_params, layer_name)


def csi_relu6(data, out_dtype, q_params, layer_name=""):
    """Quantized activation relu.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.
    out_dtype : str
        Specifies the output data type for mixed precision dense can be uint8.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIRelu6(data, out_dtype, q_params, layer_name)


def csi_maxpool2d(
    data,
    out_dtype,
    strides,
    padding,
    dilation,
    pool_size,
    ceil_mode,
    layout,
    q_params,
    layer_name="",
):
    """Quantized activation max pooling.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.
    out_dtype : str
        Specifies the output data type for mixed precision dense can be uint8.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIMaxPool(
        data,
        out_dtype,
        list(strides),
        list(padding),
        list(dilation),
        list(pool_size),
        ceil_mode,
        str(layout),
        q_params,
        layer_name,
    )


def csi_maxpool2d_with_argmax(
    data,
    out_dtype,
    strides,
    padding,
    dilation,
    pool_size,
    ceil_mode,
    layout,
    q_params,
    layer_name="",
):
    """Quantized activation max pooling.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.
    out_dtype : str
        Specifies the output data type for mixed precision dense can be uint8.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIMaxPool2dWithArgmax(
        data,
        out_dtype,
        strides,
        padding,
        dilation,
        pool_size,
        ceil_mode,
        str(layout),
        q_params,
        layer_name,
    )


def csi_maxpool2d_locat(
    data, strides, padding, pool_size, ceil_mode, out_dtype, layout, q_params, layer_name=""
):
    """Quantized activation max pooling.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.
    out_dtype : str
        Specifies the output data type for mixed precision dense can be uint8.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIMaxPool2DLocat(
        data, strides, padding, pool_size, ceil_mode, out_dtype, layout, q_params, layer_name
    )


def csi_avgpool2d(
    data,
    out_dtype,
    strides,
    padding,
    dilation,
    pool_size,
    ceil_mode,
    count_include_pad,
    layout,
    q_params,
    layer_name="",
):
    """Quantized activation average pooling.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.
    out_dtype : str
        Specifies the output data type for mixed precision dense can be uint8.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIAvgPool2d(
        data,
        out_dtype,
        strides,
        padding,
        dilation,
        pool_size,
        ceil_mode,
        count_include_pad,
        layout,
        q_params,
        layer_name,
    )


def csi_avgpool3d(
    data,
    out_dtype,
    strides,
    padding,
    pool_size,
    ceil_mode,
    count_include_pad,
    layout,
    q_params,
    layer_name="",
):
    """Quantized activation average pooling.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.
    out_dtype : str
        Specifies the output data type for mixed precision dense can be uint8.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIAvgPool3D(
        data,
        out_dtype,
        strides,
        padding,
        pool_size,
        ceil_mode,
        count_include_pad,
        layout,
        q_params,
        layer_name,
    )


def csi_maxpool3d(
    data, out_dtype, strides, padding, pool_size, ceil_mode, layout, q_params, layer_name=""
):
    """Quantized activation max pooling 3D.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.
    out_dtype : str
        Specifies the output data type for mixed precision dense can be uint8.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIMaxPool3D(
        data, out_dtype, strides, padding, pool_size, ceil_mode, layout, q_params, layer_name
    )


def csi_reshepe(data, newshape, reverse, out_dtype, q_params, layer_name=""):
    """Quantized activation reshepe.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    newshape : tuple of int, optional
        The shape of output tensor.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIReshape(data, newshape, reverse, out_dtype, q_params, layer_name)


def csi_proposal(
    cls_prob,
    bbox_pred,
    im_info,
    scales,
    ratios,
    feature_stride,
    threshold,
    rpn_pre_nms_top_n,
    rpn_post_nms_top_n,
    rpn_min_size,
    iou_loss,
    out_dtype,
    q_params,
    layer_name="",
):
    """Quantized proposal.

    Parameters
    ----------
    cls_prob: 4-D with shape [batch, 2 * num_anchors, height, width].
    bbox_pred: 4-D with shape [batch, 4 * num_anchors, height, width].
    im_info: 2-D with shape [batch, 3].

    Returns
    -------
    result : relay.Expr
        2-D with shape [batch * rpn_post_nms_top_n, 5].

    """
    return _make.CSIProposal(
        cls_prob,
        bbox_pred,
        im_info,
        scales,
        ratios,
        feature_stride,
        threshold,
        rpn_pre_nms_top_n,
        rpn_post_nms_top_n,
        rpn_min_size,
        iou_loss,
        out_dtype,
        q_params,
        layer_name,
    )


def csi_psroipooling(
    cls_prob, roi, spatial_scale, output_dim, group_size, out_dtype, q_params, layer_name=""
):
    """Quantized psroipooling.

    Parameters
    ----------
    Returns
    -------
    result : relay.Expr
    """
    return _make.CSIPSROIPooling(
        cls_prob, roi, spatial_scale, output_dim, group_size, out_dtype, q_params, layer_name
    )


def csi_roipooling(data, roi, pooled_size, spatial_scale, out_dtype, q_params, layer_name=""):
    """Quantized roipooling.

    Parameters
    ----------


    Returns
    -------
    result : relay.Expr


    """
    return _make.CSIROIPooling(
        data, roi, pooled_size, spatial_scale, out_dtype, q_params, layer_name
    )


def csi_unpooling(data, mask, scales, out_padding, out_dtype, layout, q_params, layer_name=""):
    """Quantized unpooling.

    Parameters
    ----------
    Returns
    -------
    result : relay.Expr
    """
    return _make.CSIUnPooling(
        data, mask, scales, out_padding, out_dtype, layout, q_params, layer_name
    )


def csi_upsampling(
    data, scale_h, scale_w, align_corners, method, out_dtype, layout, q_params, layer_name=""
):
    """Quantized upsampling.

    Parameters
    ----------
    Returns
    -------
    result : relay.Expr
    """

    return _make.CSIUpSampling(
        data, scale_h, scale_w, align_corners, method, out_dtype, layout, q_params, layer_name
    )


def csi_flatten(data, out_dtype, q_params, layer_name=""):
    """Quantized activation flatten.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    newshape : tuple of int, optional
        The shape of output tensor.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIFlatten(data, out_dtype, q_params, layer_name)


def csi_sigmoid(data, out_dtype, q_params, layer_name=""):
    """Quantized activation flatten.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    newshape : tuple of int, optional
        The shape of output tensor.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSISigmoid(data, out_dtype, q_params, layer_name)


def csi_transpose(data, axes, out_dtype, q_params, layer_name=""):
    """Quantized activation transpose.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    newshape : tuple of int, optional
        The shape of output tensor.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSITranspose(data, axes, out_dtype, q_params, layer_name)


def csi_softmax(data, axis, out_dtype, q_params, layer_name=""):
    """Quantized activation softmax.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.
    out_dtype : str
        Specifies the output data type for mixed precision dense can be uint8.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSISoftMax(data, axis, out_dtype, q_params, layer_name)


def csi_reverse(data, axis, out_dtype, q_params, layer_name=""):
    """Quantized reverse.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.
    out_dtype : str
        Specifies the output data type for mixed precision dense can be uint8.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIReverse(data, axis, out_dtype, q_params, layer_name)


def csi_log_softmax(data, axis, out_dtype, q_params, layer_name=""):
    """Quantized activation softmax.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.
    out_dtype : str
        Specifies the output data type for mixed precision dense can be uint8.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSILogSoftMax(data, axis, out_dtype, q_params, layer_name)


def csi_lrn(data, size, axis, alpha, beta, bias, norm_region, out_dtype, q_params, layer_name=""):
    """Quantized activation lrn.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.
    out_dtype : str
        Specifies the output data type for mixed precision dense can be uint8.

    size : int
        The size of the local region to be considered for normalization.

    bias : int
        The offset parameter to avoid division by 0.

    alpha : float
        The scaling parameter.

    beta : float
        The exponent parameter.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSILRN(
        data, size, axis, alpha, beta, bias, norm_region, out_dtype, q_params, layer_name
    )


def csi_global_avgpool2d(data, layout, out_dtype, q_params, layer_name=""):
    """Quantized global average pooling layer.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.
    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIGlobalAvgPool(data, layout, out_dtype, q_params, layer_name)


def csi_global_maxpool2d(data, layout, out_dtype, q_params, layer_name=""):
    """Quantized global max pooling layer.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.
    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIGlobalMaxPool(data, layout, out_dtype, q_params, layer_name)


def csi_mean(data, axis, keepdims, exclude, out_dtype, q_params, layer_name=""):
    """Quantized activation lrn.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.
    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIMean(data, axis, keepdims, exclude, out_dtype, q_params, layer_name)


def csi_prod(data, axis, keepdims, exclude, out_dtype, q_params, layer_name=""):
    """Quantized prod.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.
    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIProd(data, axis, keepdims, exclude, out_dtype, q_params, layer_name)


def csi_max(data, axis, keepdims, exclude, out_dtype, q_params, layer_name=""):
    """Quantized Max.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.
    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIMax(data, axis, keepdims, exclude, out_dtype, q_params, layer_name)


def csi_min(data, axis, keepdims, exclude, out_dtype, q_params, layer_name=""):
    """Quantized Max.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.
    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIMin(data, axis, keepdims, exclude, out_dtype, q_params, layer_name)


def csi_sum(data, axis, keepdims, exclude, out_dtype, q_params, layer_name=""):
    """Quantized activation sum.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.
    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSISum(data, axis, keepdims, exclude, out_dtype, q_params, layer_name)


def csi_pad(data, pad_value, pad_width, pad_mode, out_dtype, q_params, layer_name=""):
    """Quantized pad.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.
    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIPad(data, pad_value, pad_width, pad_mode, out_dtype, q_params, layer_name)


def csi_squeeze(data, axis, out_dtype, q_params, layer_name=""):
    """Quantized activation lrn.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.
    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSISqueeze(data, axis, out_dtype, q_params, layer_name)


def csi_reshape(data, newshape, out_dtype, q_params, layer_name=""):
    """Quantized activation lrn.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.
    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIReshape(data, newshape, out_dtype, q_params, layer_name)


def csi_batch_norm(
    data,
    gamma,
    beta,
    moving_mean,
    moving_var,
    axis,
    epsilon,
    center,
    scale,
    q_params,
    layer_name="",
):
    r"""
    Batch normalization layer (Ioffe and Szegedy, 2014).
    Normalizes the input at each batch, i.e. applies a transformation
    that maintains the mean activation close to 0 and the activation
    standard deviation close to 1.

    .. math::

        data\_mean[i] = mean(data[:,i,:,...]) \\
        data\_var[i] = var(data[:,i,:,...])

    Then compute the normalized output, which has the same shape as input, as following:

    .. math::

        out[:,i,:,...] = \frac{data[:,i,:,...] - data\_mean[i]}{\sqrt{data\_var[i]+\epsilon}}
            * gamma[i] + beta[i]

    Both *mean* and *var* returns a scalar by treating the input as a vector.

    Assume the input has size *k* on axis 1, then both ``gamma`` and ``beta``
    have shape *(k,)*.

    Besides the inputs and the outputs, this operator accepts two auxiliary
    states, ``moving_mean`` and ``moving_var``, which are *k*-length
    vectors. They are global statistics for the whole dataset, which are updated by

    .. code:: python

        moving_mean = moving_mean * momentum + data_mean * (1 - momentum)
        moving_var = moving_var * momentum + data_var * (1 - momentum)

    The parameter ``axis`` specifies which axis of the input shape denotes
    the 'channel' (separately normalized groups).  The default is 1.
    Specifying -1 sets the channel axis to be the last item in the input shape.

    .. note::

        This operator can be optimized away for inference.

    Parameters
    ----------
    data : tvm.relay.Expr
        Input to which batch_norm will be applied.

    gamma : tvm.relay.Expr
        The gamma scale factor.

    beta : tvm.relay.Expr
        The beta offset factor.

    moving_mean : tvm.relay.Expr
        Running mean of input,

    moving_var : tvm.relay.Expr
        Running variance of input.

    axis : int, optional, default=1
        Specify along which shape axis the channel is specified.

    epsilon : double, optional, default=1e-5
        Small float added to variance to avoid dividing by zero.

    center : boolean, optional, default=True
        If True, add offset of beta to normalized tensor, If False,
        beta is ignored.

    scale : boolean, optional, default=True
        If true, multiply by gamma. If False, gamma is not used.
        When the next layer is piecewise linear (also e.g. nn.relu),
        this can be disabled since the scaling will be done by the next layer.

    Returns
    -------
    result : relay.Tuple([tvm.relay.Expr, tvm.relay.Expr, tvm.relay.Expr])
        Tuple of normed data (same shape as input),
        new running mean (k-length vector),
        and new running variance (k-length vector)
    """
    return _make.csi_batch_norm(
        data,
        gamma,
        beta,
        moving_mean,
        moving_var,
        axis,
        epsilon,
        center,
        scale,
        q_params,
        layer_name,
    )


def csi_strided_slice(data, begin, end, strides, out_dtype, q_params, layer_name=""):
    """Strided slice of an array.

    Parameters
    ----------
    data : relay.Expr
        The source array to be sliced.

    begin: list of int
        The indices to begin with in the slicing.

    end: list of int
        Indices indicating end of the slice.

    strides: list of int, optional
        Specifies the stride values, it can be negative in that case,
        the input tensor will be reversed in that particular axis.

    Returns
    -------
    ret : relay.Expr
        The computed result.
    """
    return _make.CSIStridedSlice(
        data, list(begin), list(end), list(strides), out_dtype, q_params, layer_name
    )


def csi_split(data, indices_or_sections, axis, out_dtype, q_params, layer_name=""):
    """Split input tensor along axis by sections or indices.

    If indices_or_sections is an integer, the input will be divided equally
    along given axis. If such a split is not possible, an error is raised.

    If indices_or_sections is a tuple of sorted integers,
    the entries indicate where along axis the array is split.

    Parameters
    ----------
    data : relay.Expr
        The source array.

    indices_or_sections : int or tuple of int
        Indices or sections to split into. Accepts an int or a tuple

    axis : int, optional
        The axis over which to split.

    Returns
    -------
    ret : relay.Tuple([relay.Expr, relay.Expr])
        The computed result.
    """
    return _make.CSISplit(data, indices_or_sections, axis, out_dtype, q_params, layer_name)


def csi_variance(data, axis, keepdims, exclude, out_dtype, q_params, layer_name=""):
    """Computes the variance of data over given axes.

    Parameters
    ----------
    data : relay.Expr
        The input data

    axis : None or int or tuple of int
        Axis or axes along which a variance operation is performed.
        The default, axis=None, will compute the variance of all elements in the input array.
        If axis is negative it counts from the last to the first axis.

    keepdims : bool
        If this is set to True, the axes which are reduced are left in the result as dimensions
        with size one.
        With this option, the result will broadcast correctly against the input array.

    exclude : bool
        If `exclude` is true, reduction will be performed on the axes that are
        NOT in axis instead.

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    axis = [axis] if isinstance(axis, int) else axis
    return _make.CSIVariance(data, axis, keepdims, exclude, out_dtype, q_params, layer_name)


def csi_exp(data, out_dtype, q_params, layer_name=""):
    """Take exponetial of input x.

    Parameters
    ----------
    data : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _make.CSIExp(data, out_dtype, q_params, layer_name)


def csi_equal(lhs, rhs, q_params, layer_name=""):
    """Quantized equal.

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side quantized input data.

    rhs : relay.Expr
        The right hand side quantized input data.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIEqual(lhs, rhs, q_params, layer_name)


def csi_scatter_nd(data, indices, updates, out_dtype, q_params, layer_name=""):
    """Quantized scatter nd.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    indices : relay.Expr
        The index locations to update.

    updates : relay.Expr
        The values to replace.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIScatterND(data, indices, updates, out_dtype, q_params, layer_name)


def csi_segment_max(data, ids, num_segments, out_dtype, q_params, layer_name=""):
    """Quantized segment max.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    ids : relay.Expr
        The index.
    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSISegmentMax(data, ids, num_segments, out_dtype, q_params, layer_name)


def csi_segment_min(data, ids, num_segments, out_dtype, q_params, layer_name=""):
    """Quantized segment max.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    ids : relay.Expr
        The index.
    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSISegmentMin(data, ids, num_segments, out_dtype, q_params, layer_name)


def csi_segment_mean(data, ids, num_segments, out_dtype, q_params, layer_name=""):
    """Quantized segment max.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    ids : relay.Expr
        The index.
    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSISegmentMean(data, ids, num_segments, out_dtype, q_params, layer_name)


def csi_segment_prod(data, ids, num_segments, out_dtype, q_params, layer_name=""):
    """Quantized segment max.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    ids : relay.Expr
        The index.
    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSISegmentProd(data, ids, num_segments, out_dtype, q_params, layer_name)


def csi_segment_sum(data, ids, num_segments, out_dtype, q_params, layer_name=""):
    """Quantized segment max.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    ids : relay.Expr
        The index.
    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSISegmentSum(data, ids, num_segments, out_dtype, q_params, layer_name)


def csi_log(data, out_dtype, q_params, layer_name=""):
    """Take log of input x.

    Parameters
    ----------
    data : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _make.CSILog(data, out_dtype, q_params, layer_name)


def csi_negative(data, out_dtype, q_params, layer_name=""):
    """Take negative of input x.

    Parameters
    ----------
    data : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _make.CSINegative(data, out_dtype, q_params, layer_name)


def csi_abs(data, out_dtype, q_params, layer_name=""):
    """Take abs of input x.

    Parameters
    ----------
    data : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _make.CSIAbs(data, out_dtype, q_params, layer_name)


def csi_expand_dims(data, axis, num_newaxis, out_dtype, q_params, layer_name=""):
    """Take abs of input x.

    Parameters
    ----------
    data : PrimExpr
        Input argument.
    axis : int

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _make.CSIExpandDims(data, axis, num_newaxis, out_dtype, q_params, layer_name)


def csi_argmax(data, axis, keepdims, exclude, out_dtype, q_params, layer_name=""):
    """Quantized argmax.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.
    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIArgmax(data, axis, keepdims, exclude, out_dtype, q_params, layer_name)


def csi_argmin(data, axis, keepdims, exclude, out_dtype, q_params, layer_name=""):
    """Quantized argmax.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.
    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIArgmin(data, axis, keepdims, exclude, out_dtype, q_params, layer_name)


def csi_broadcast_to(data, shape, out_dtype, q_params, layer_name=""):
    """Quantized broadcast_to.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.
    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIBroadCastTo(data, shape, out_dtype, q_params, layer_name)


def csi_cast(data, out_dtype, q_params, layer_name=""):
    """Quantized cast.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.
    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSICast(data, out_dtype, q_params, layer_name)


def csi_ceil(data, out_dtype, q_params, layer_name=""):
    """Quantized ceil.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.
    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSICeil(data, out_dtype, q_params, layer_name)


def csi_floor(data, out_dtype, q_params, layer_name=""):
    """Quantized floor.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.
    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIFloor(data, out_dtype, q_params, layer_name)


def csi_clip(data, a_min, a_max, out_dtype, q_params, layer_name=""):
    """Quantized clip.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.
    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIClip(data, a_min, a_max, out_dtype, q_params, layer_name)


def csi_erf(data, out_dtype, q_params, layer_name=""):
    """Quantized Erf.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.
    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIErf(data, out_dtype, q_params, layer_name)


def csi_round(data, out_dtype, q_params, layer_name=""):
    """Quantized round.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.
    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIRound(data, out_dtype, q_params, layer_name)


def csi_maximum(lhs, rhs, q_params, layer_name=""):
    """Quantized maximun with numpy-style broadcasting.

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side quantized input data.

    rhs : relay.Expr
        The right hand side quantized input data.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIMaximum(lhs, rhs, q_params, layer_name)


def csi_floor_div(lhs, rhs, q_params, layer_name=""):
    """Quantized floor_div with numpy-style broadcasting.

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side quantized input data.

    rhs : relay.Expr
        The right hand side quantized input data.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIFloorDiv(lhs, rhs, q_params, layer_name)


def csi_floor_mod(lhs, rhs, q_params, layer_name=""):
    """Quantized floor_mod with numpy-style broadcasting.

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side quantized input data.

    rhs : relay.Expr
        The right hand side quantized input data.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIFloorMod(lhs, rhs, q_params, layer_name)


def csi_left_shift(lhs, rhs, q_params, layer_name=""):
    """Quantized left_shift with numpy-style broadcasting.

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side quantized input data.

    rhs : relay.Expr
        The right hand side quantized input data.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSILeftShift(lhs, rhs, q_params, layer_name)


def csi_right_shift(lhs, rhs, q_params, layer_name=""):
    """Quantized right_shift with numpy-style broadcasting.

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side quantized input data.

    rhs : relay.Expr
        The right hand side quantized input data.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIRightShift(lhs, rhs, q_params, layer_name)


def csi_minimum(lhs, rhs, q_params, layer_name=""):
    """Quantized minimun with numpy-style broadcasting.

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side quantized input data.

    rhs : relay.Expr
        The right hand side quantized input data.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIMinimum(lhs, rhs, q_params, layer_name)


def csi_crop_resize(
    data,
    boxes,
    box_indices,
    crop_size,
    layout,
    method,
    extrapolation_value,
    out_dtype,
    q_params,
    layer_name="",
):
    """Quantized crop_resize with numpy-style broadcasting.

    Parameters
    ----------

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSICropResize(
        data,
        boxes,
        box_indices,
        crop_size,
        layout,
        method,
        extrapolation_value,
        out_dtype,
        q_params,
        layer_name,
    )


def csi_depth_to_space(data, block_size, layout, mode, out_dtype, q_params, layer_name=""):
    """Quantized crop_resize with numpy-style broadcasting.

    Parameters
    ----------

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIDepthToSpace(data, block_size, layout, mode, out_dtype, q_params, layer_name)


def csi_space_to_depth(data, block_size, layout, mode, out_dtype, q_params, layer_name=""):
    """Quantized crop_resize with numpy-style broadcasting.

    Parameters
    ----------

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSISpaceToDepth(data, block_size, layout, mode, out_dtype, q_params, layer_name)


def csi_sqrt(data, out_dtype, q_params, layer_name=""):
    """Quantized sqrt.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.
    out_dtype : str
        Specifies the output data type for mixed precision dense can be uint8.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSISqrt(data, out_dtype, q_params, layer_name)


def csi_rsqrt(data, out_dtype, q_params, layer_name=""):
    """Quantized rsqrt.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.
    out_dtype : str
        Specifies the output data type for mixed precision dense can be uint8.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIRsqrt(data, out_dtype, q_params, layer_name)


def csi_sign(data, out_dtype, q_params, layer_name=""):
    """Quantized sign.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.
    out_dtype : str
        Specifies the output data type for mixed precision dense can be uint8.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSISign(data, out_dtype, q_params, layer_name)


def csi_full(data, shape, out_dtype, q_params, layer_name=""):
    """Quantized fill.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.
    out_dtype : str
        Specifies the output data type for mixed precision dense can be uint8.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIFull(data, shape, out_dtype, q_params, layer_name)


def csi_take(data, indices, axis, mode, out_dtype, q_params, layer_name=""):
    """Quantized Take.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.
    out_dtype : str
        Specifies the output data type for mixed precision dense can be uint8.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSITake(data, indices, axis, mode, out_dtype, q_params, layer_name)


def csi_tile(data, reps, out_dtype, q_params, layer_name=""):
    """Quantized tile.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.
    out_dtype : str
        Specifies the output data type for mixed precision dense can be uint8.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSITile(data, reps, out_dtype, q_params, layer_name)


def csi_unravel_index(data, shape, out_dtype, q_params, layer_name=""):
    """Quantized unravel_index.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.
    out_dtype : str
        Specifies the output data type for mixed precision dense can be uint8.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIUnRavelIndex(data, shape, out_dtype, q_params, layer_name)


def csi_topk(data, k, axis, ret_type, is_ascend, dtype, out_dtype, q_params, layer_name=""):
    """Quantized Take.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.
    out_dtype : str
        Specifies the output data type for mixed precision dense can be uint8.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSITopK(data, k, axis, ret_type, is_ascend, dtype, out_dtype, q_params, layer_name)


def csi_fsmn(
    frame,
    l_filter,
    r_filter,
    frame_sequence,
    frame_counter,
    l_order,
    r_order,
    l_stride,
    r_stride,
    unavailable_frames,
    out_dtype,
    q_params,
    layer_name="",
):
    """Quantized fsmn operator.

    Parameters
    ----------
    Input : tvm.te.Tensor
        2-D with shape [1, in_width]

    l_filter : tvm.te.Tensor
        2-D with shape [l_order, in_width]

    r_filter : tvm.te.Tensor
        2-D with shape [r_order, in_width]

    frame_sequence: tvm.te.Tensor
        2-D with shape [(l_order -1) * l_stride + r_order * r_stride, in_width ]

    l_stride : int
        Left stride size

    r_stride : int
        Right stride size

    Returns
    -------
    Output : tvm.te.Tensor
        2-D with shape [1, in_width]
    """
    return _make.CSIFsmn(
        frame,
        l_filter,
        r_filter,
        frame_sequence,
        frame_counter,
        l_order,
        r_order,
        l_stride,
        r_stride,
        unavailable_frames,
        q_params,
        out_dtype,
        layer_name,
    )


def csi_space_to_batch_nd(
    data, block_shape, paddings, pad_value, out_dtype, q_params, layer_name=""
):
    """Quantized space_to_batch_nd operator.

    Parameters
    ----------
    data : tvm.te.tensor
        (batch, spatial_shapes, remaining_shapes) for NHWC

    block_shape : list
        1-D of size [M] where M is number of spatial dims, specifies block size for each
        spatial dimension.

    paddings : list
         2-D of shape [M, 2] where M is number of spatial dims, specifies [before, after]
         paddings for each spatial dimension.

    pad_value : float
        The value used for padding, default is 0

    out_dtype : str
        Specifies the output data type.

    q_params : list
        Quantization parameters

    layer_name : str
        The layer name

    Returns
    -------
    Output : tvm.te.Tensor
        N-D Tensor with shape [in_batch * prod(block_shape), padded_data[1] / block_shape[0], …,
        padded_data[M] / block_shape[M-1], remaining_shape]
    """
    return _make.CSISpaceToBatchND(
        data, block_shape, paddings, pad_value, out_dtype, q_params, layer_name
    )


def csi_batch_to_space_nd(data, block_shape, crops, out_dtype, q_params, layer_name=""):
    """Quantized batch_to_space_nd operator.

    Parameters
    ----------
    data : tvm.te.tensor
        (batch, spatial_shapes, remaining_shapes) for NHWC

    block_shape : list
        1-D of size [M] where M is number of spatial dims, specifies block size for each
        spatial dimension.

    crops : list
         2-D of shape [M, 2] where M is number of spatial dims, specifies [begin, end]
         crop size for each spatial dimension.

    out_dtype : str
        Specifies the output data type.

    q_params : list
        Quantization parameters

    layer_name : str
        The layer name

    Returns
    -------
    Output : tvm.te.Tensor
        N-D Tensor with shape [batch / prod(block_shape), in_shape[1] * block_shape[0] -
        crops[0,0] - crops[0,1], …, in_shape[M] * block_shape[M-1] - crops[M-1, 0] -
        crops[M-1, 1], remaining_shape]
    """
    return _make.CSIBatchToSpaceND(data, block_shape, crops, out_dtype, q_params, layer_name)


def csi_matmul(data_a, data_b, bias, transpose_a, transpose_b, out_dtype, q_params, layer_name=""):
    r"""Qnn MatMul operator.
    Compute batch matrix multiplication of `tensor_a` and `tensor_b`.

    Both `tensor_a` and `tensor_b` can be transposed. For legacy reason, we use NT format
    (transpose_a=False, transpose_b=True) by default.

    .. math::

        \mbox{batch_matmul}(A, B)[i, :, :] = \mbox{matmul}(A[i, :, :], B[i, :, :])

    Parameters
    ----------
    tensor_a : tvm.relay.Expr
        The first input.

    tensor_b : tvm.relay.Expr
        The second input.

    out_dtype : Optional[str]
        Specifies the output data type for mixed precision batch matmul.

    transpose_a : Optional[bool] = False
        Whether the first tensor is in transposed format.

    transpose_b : Optional[bool] = True
        Whether the second tensor is in transposed format.

    Returns
    -------
    result: tvm.relay.Expr
        The computed result.
    """
    return _make.CSIMatMul(
        data_a, data_b, bias, transpose_a, transpose_b, out_dtype, q_params, layer_name
    )


def csi_cache_matmul(
    data, weight, bias, cache_shape, shape, axes, out_dtype, q_params, layer_name=""
):
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
    return _make.CSICacheMatMul(
        data, weight, bias, cache_shape, shape, axes, out_dtype, q_params, layer_name
    )


def csi_layer_norm(
    data, gamma, beta, axis, epsilon, center, scale, out_dtype, q_params, layer_name=""
):
    r"""
    Layer normalization (Lei Ba and et al., 2016).
    Applies layer normalization to the n-dimensional input array.
    This operator takes an n-dimensional input array and normalizes
    the input using the given axis:

    .. math::

        out = \frac{data - mean(data, axis)}{\sqrt{var(data, axis)+\epsilon}}
            * gamma + beta

    Unlike batch normalization, the mean and var are computed along the channel dimension.

    Assume the input has size k on axis 1, then both gamma and beta have shape (k,).

    .. note::

        This operator can be optimized away for inference.

    Parameters
    ----------
    data : tvm.relay.Expr
        Input to which layer_norm will be applied.

    gamma : tvm.relay.Expr
        The gamma scale factor.

    beta : tvm.relay.Expr
        The beta offset factor.

    axis : int, optional, default=-1
        The axis that should be normalized, typically the axis of the channels.

    epsilon : double, optional, default=1e-5
        Small float added to variance to avoid dividing by zero.

    center : boolean, optional, default=True
        If True, add offset of beta to normalized tensor, If False,
        beta is ignored.

    scale : boolean, optional, default=True
        If True, multiply by gamma. If False, gamma is not used.

    Returns
    -------
    result : tvm.relay.Expr
        The normalized data.
    """
    return _make.CSILayerNorm(
        data, gamma, beta, axis, epsilon, center, scale, out_dtype, q_params, layer_name
    )


def csi_cache_conv1d(
    data,
    weight,
    bias,
    cache_shape,
    strides=1,
    padding=0,
    dilation=1,
    groups=1,
    channels=None,
    kernel_size=None,
    data_layout="NCW",
    kernel_layout="OIW",
    out_layout="",
    out_dtype="",
    q_params=None,
    layer_name="",
):
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
    return _make.CSICacheConv1D(
        data,
        weight,
        bias,
        cache_shape,
        strides,
        padding,
        dilation,
        groups,
        channels,
        kernel_size,
        data_layout,
        kernel_layout,
        out_layout,
        out_dtype,
        q_params,
        layer_name,
    )


def csi_custom_op(
    data,
    op_type,
    custom_attr,
    layer_name="",
):
    r"""
    custom op
    """

    attr = []
    if isinstance(custom_attr, dict):
        for x, y in custom_attr.items():
            attr.append(str(x))
            attr.append(str(y))
    elif isinstance(custom_attr, list):
        attr = [str(x) for x in custom_attr]
    else:
        ValueError("Unsupported custom attr type: " + type(custom_attr))

    return _make.CSICustomOp(
        data,
        op_type,
        custom_attr,
        layer_name,
    )


def csi_less(
    lhs,
    rhs,
    q_params,
    layer_name="",
):
    """Quantized less.

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side quantized input data.

    rhs : relay.Expr
        The right hand side quantized input data.

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.CSILess(lhs, rhs, q_params, layer_name)


def csi_one_hot(
    indices,
    depth,
    axis,
    out_dtype,
    q_params,
    layer_name="",
):
    """
    Returns a one-hot tensor where the locations repsented by indices take 1,
    other locations take value 0.
    Final dimension is <indices outer dimensions> x depth x <indices inner dimensions>.

    Parameters
    ----------
    indices : relay.Expr
        Locations to set to on_value.

    depth : int or relay.Expr
        Depth of the one-hot dimension.

    axis : int
        Axis to fill.

    out_dtype : str
        Data type of the output tensor.

    Returns
    -------
    ret : relay.Expr
        The one-hot tensor.

    Examples
    --------
    .. code-block:: python

        indices = [0, 1, 2]

        relay.one_hot(indices, 3) =
            [[1, 0, 0],
             [0, 1, 0],
             [0, 0, 1]]
    """
    return _make.CSIOneHot(indices, depth, axis, out_dtype, q_params, layer_name)


def csi_where(
    condition,
    x,
    y,
    out_dtype,
    q_params,
    layer_name="",
):
    """Selecting elements from either x or y depending on the value of the
    condition.

    .. note::
        Shapes of condition, x, and y must be broadcastable to a common shape.
        Semantics follow numpy where function
        https://numpy.org/doc/stable/reference/generated/numpy.where.html

    Parameters
    ----------
    condition : relay.Expr
        Where True, yield x, otherwise yield y

    x : relay.Expr
        The first array or scalar to be selected.

    y : relay.Expr
        The second array or scalar to be selected.

    Returns
    -------
    result : relay.Expr
        The selected array. The output shape is the broadcasted shape from
        condition, x, and y.

    Examples
    --------
    .. code-block:: python

        x = [[1, 2], [3, 4]]
        y = [[5, 6], [7, 8]]
        condition = [[0, 1], [-1, 0]]
        relay.where(conditon, x, y) = [[5, 2], [3, 8]]

        condition = [[1], [0]]
        relay.where(conditon, x, y) = [[1, 2], [7, 8]]
    """
    return _make.CSIWhere(condition, x, y, out_dtype, q_params, layer_name)


# register fuse pattern for qnn ops
reg.register_pattern("qnn.quantize", OpPattern.OPAQUE)
reg.register_pattern("qnn.dequantize", OpPattern.OPAQUE)


def leaky_relu(x, alpha, scale, zero_point):
    """Quantized leaky relu.

    Parameters
    ----------
    x : relay.Expr
        The quantized input tensor.
    alpha: double
        The alpha value.
    scale: relay.Expr
        The scale of the quantized expr.
    zero_point: relay.Expr
       The zero point of quantized expr.

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    return _make.leaky_relu(
        x,
        alpha,
        scale,
        zero_point,
    )


def csi_where_softmax(
    condition,
    x,
    y,
    minus_inf,
    axis,
    out_dtype,
    q_params,
    layer_name="",
):
    """Selecting elements from either x or y depending on the value of the
    condition, then compute softmax at axis.

    .. note::
        Shapes of condition, x, and y must be broadcastable to a common shape.
        Semantics follow numpy where function
        https://numpy.org/doc/stable/reference/generated/numpy.where.html

    Parameters
    ----------
    condition : relay.Expr
        Where True, yield x, otherwise yield y

    x : relay.Expr
        The first array or scalar to be selected.

    y : relay.Expr
        The second array or scalar to be selected.

    axis : int
        The axis to sum over when computing softmax.

    Returns
    -------
    result : relay.Expr
        The computed result.

    Examples
    --------
    .. code-block:: python

        x = [[1, 2], [3, 4]]
        y = [[5, 6], [7, 8]]
        condition = [[0, 1], [-1, 0]]
        axis = -1
        where_softmax(conditon, x, y, axis) = softmax(where(conditon, x, y), axis)
    """
    return _make.CSIWhereSoftmax(condition, x, y, minus_inf, axis, out_dtype, q_params, layer_name)


def csi_quantize(
    data,
    output_scale,
    output_zero_point,
    axis,
    out_dtype,
    q_params,
    layer_name="",
):
    r"""Quantize op
    This operator takes float32 as input and produces quantized int8 or unit8 as output.
    The input tensor can be of any shape. The output shape is the same as input shape.

    Q_output = clamp((round(input_tensor/output_scale) + output_zero_point),
                     out_dtype::min,
                     out_dtype::max)

    Parameters
    ----------
    data : tvm.relay.Expr
        The input tensor to be quantized. Can be of type float32.

    output_scale : tvm.relay.Expr
        The output scale.

    output_zero_point : tvm.relay.Expr
        The output zero_point.

    axis : int
        The channel axis for quantization. Default value is -1 which corresponds to the last axis.
    out_dtype : str, optional
        The data type of the input tensor. Can be [int8, uint8, int32]

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    return _make.CSIQuantize(
        data, output_scale, output_zero_point, axis, out_dtype, q_params, layer_name
    )


def csi_dequantize(
    data,
    input_scale,
    input_zero_point,
    axis,
    out_dtype,
    q_params,
    layer_name="",
):
    r"""Dequantize op
    This operator takes quantized int8 and unit8 as input and produces
    dequantized float32 as output. The output shape is the same as input shape. The input
    tensor can be of any shape.

    Parameters
    ----------
    data : tvm.relay.Expr
        The input tensor to be dequantized. Can be of type [int8, uint8, int32].

    input_scale : tvm.relay.Expr
        The input scale.

    input_zero_point : tvm.relay.Expr
        The input zero_point.

    axis : int
        The channel axis for quantization. Default value is -1 which corresponds to the last axis.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    return _make.CSIDequantize(
        data, input_scale, input_zero_point, axis, out_dtype, q_params, layer_name
    )


def csi_data_convert(
    data,
    out_dtype,
    q_params,
    layer_name="",
):
    r"""DataConvert op
    This operator convert tensor with one dtype into tensor with another dtype.

    Parameters
    ----------
    data : tvm.relay.Expr
        The input tensor to be dequantized. Can be of type [int8, uint8, int32].

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    return _make.CSIDataConvert(data, out_dtype, q_params, layer_name)
