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
"""Definition of riscv operator strategy."""
# pylint: disable=invalid-name,unused-argument,wildcard-import,unused-wildcard-import
import logging
import re

from tvm import tir, topi
from tvm.auto_scheduler import is_auto_scheduler_enabled
from tvm.meta_schedule import is_meta_schedule_enabled
from tvm.relay.ty import is_dynamic
from tvm.target import Target
from tvm.te import SpecializedCondition
from tvm.topi.riscv.utils import target_has_vnni

from .. import op as _op
from .generic import *

logger = logging.getLogger("strategy")

_NCHWc_matcher = re.compile("^NCHW[0-9]+c$")
_OIHWio_matcher = re.compile("^OIHW[0-9]+i[0-9]+o$")


@schedule_injective.register("riscv")
def schedule_injective_cpu(attrs, outs, target):
    """schedule injective ops for riscv"""
    with target:
        return topi.riscv.schedule_injective(outs)


@schedule_reduce.register("riscv")
def schedule_reduce_cpu(attrs, outs, target):
    """schedule reduction ops for riscv"""
    with target:
        return topi.riscv.schedule_reduce(outs)


@schedule_pool.register("riscv")
def schedule_pool_cpu(attrs, outs, target):
    """schedule pooling ops for riscv"""
    with target:
        return topi.riscv.schedule_pool(outs, attrs, attrs.layout)


@schedule_adaptive_pool.register("riscv")
def schedule_adaptive_pool_cpu(attrs, outs, target):
    """schedule adaptive pooling ops for riscv"""
    with target:
        return topi.riscv.schedule_adaptive_pool(outs)


@softmax_strategy.register("riscv")
def softmax_strategy_cpu(attrs, inputs, out_type, target):
    """softmax riscv strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_softmax(topi.nn.softmax_rvv),
        wrap_topi_schedule(topi.riscv.schedule_softmax),
        name="softmax.riscv",
    )
    return strategy


@reshape_strategy.register("riscv")
def reshape_strategy_cpu(attrs, inputs, out_type, target):
    """reshape riscv strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_reshape(topi.reshape),
        wrap_topi_schedule(topi.riscv.schedule_reshape),
        name="reshape.riscv",
    )
    return strategy


@layout_transform_strategy.register("riscv")
def layout_transform_strategy_cpu(attrs, inputs, out_type, target):
    """layout_transform riscv strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_layout_transform(topi.layout_transform),
        wrap_topi_schedule_with_attrs(topi.riscv.schedule_layout_transform),
        name="layout_transform.riscv",
    )
    return strategy


@schedule_transpose.register("riscv")
def schedule_transpose_cpu(attrs, outs, target):
    """schedule transpose ops for riscv"""
    with target:
        return topi.riscv.schedule_transpose(outs, attrs)


@fast_softmax_strategy.register("riscv")
def fast_softmax_strategy_cpu(attrs, inputs, out_type, target):
    """fast_softmax riscv strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_softmax(topi.nn.fast_softmax),
        wrap_topi_schedule(topi.riscv.schedule_softmax),
        name="fast_softmax.riscv",
    )
    return strategy


@log_softmax_strategy.register("riscv")
def log_softmax_strategy_cpu(attrs, inputs, out_type, target):
    """log_softmax riscv strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_softmax(topi.nn.log_softmax),
        wrap_topi_schedule(topi.riscv.schedule_softmax),
        name="log_softmax.riscv",
    )
    return strategy


def wrap_add_schedule(topi_schedule, broadcast_flag):
    """Wrap add schedule"""

    def wrapper(attrs, outs, target):
        with target:
            return topi_schedule(outs, broadcast_flag)

    return wrapper


@add_strategy.register("riscv")
def add_strategy_cpu(attrs, inputs, out_type, target):
    """add riscv strategy"""
    strategy = _op.OpStrategy()
    if inputs[0].shape[-1] > inputs[1].shape[-1]:
        broadcast_flag = 1
    elif inputs[0].shape[-1] < inputs[1].shape[-1]:
        broadcast_flag = 2
    else:
        broadcast_flag = 0
    strategy.add_implementation(
        wrap_topi_compute(topi.add),
        wrap_add_schedule(topi.riscv.schedule_add, broadcast_flag),
        name="add.riscv",
    )
    return strategy


def wrap_subtract_schedule(topi_schedule, broadcast_flag):
    """Wrap subtract schedule"""

    def wrapper(attrs, outs, target):
        with target:
            return topi_schedule(outs, broadcast_flag)

    return wrapper


@subtract_strategy.register("riscv")
def subtract_strategy_cpu(attrs, inputs, out_type, target):
    """subtract riscv strategy"""
    strategy = _op.OpStrategy()
    if inputs[0].shape[-1] > inputs[1].shape[-1]:
        broadcast_flag = 1
    elif inputs[0].shape[-1] < inputs[1].shape[-1]:
        broadcast_flag = 2
    else:
        broadcast_flag = 0
    strategy.add_implementation(
        wrap_topi_compute(topi.subtract),
        wrap_subtract_schedule(topi.riscv.schedule_subtract, broadcast_flag),
        name="subtract.riscv",
    )
    return strategy


def wrap_multiply_schedule(topi_schedule, broadcast_flag):
    """Wrap multiply schedule"""

    def wrapper(attrs, outs, target):
        with target:
            return topi_schedule(outs, broadcast_flag)

    return wrapper


@multiply_strategy.register("riscv")
def multiply_strategy_cpu(attrs, inputs, out_type, target):
    """multiply riscv strategy"""
    strategy = _op.OpStrategy()
    if inputs[0].shape[-1] > inputs[1].shape[-1]:
        broadcast_flag = 1
    elif inputs[0].shape[-1] < inputs[1].shape[-1]:
        broadcast_flag = 2
    else:
        broadcast_flag = 0
    strategy.add_implementation(
        wrap_topi_compute(topi.multiply),
        wrap_multiply_schedule(topi.riscv.schedule_multiply, broadcast_flag),
        name="multiply.riscv",
    )
    return strategy


def wrap_divide_schedule(topi_schedule, broadcast_flag):
    """Wrap divide schedule"""

    def wrapper(attrs, outs, target):
        with target:
            return topi_schedule(outs, broadcast_flag)

    return wrapper


@divide_strategy.register("riscv")
def divide_strategy_cpu(attrs, inputs, out_type, target):
    """divide riscv strategy"""
    strategy = _op.OpStrategy()
    if inputs[0].shape[-1] > inputs[1].shape[-1]:
        broadcast_flag = 1
    elif inputs[0].shape[-1] < inputs[1].shape[-1]:
        broadcast_flag = 2
    else:
        broadcast_flag = 0
    strategy.add_implementation(
        wrap_topi_compute(topi.divide),
        wrap_divide_schedule(topi.riscv.schedule_divide, broadcast_flag),
        name="divide.riscv",
    )
    return strategy


@conv2d_strategy.register("riscv")
def conv2d_strategy_cpu(attrs, inputs, out_type, target):
    """conv2d riscv strategy"""
    strategy = _op.OpStrategy()
    data, kernel = inputs
    stride_h, stride_w = get_const_tuple(attrs.strides)
    dilation_h, dilation_w = get_const_tuple(attrs.dilation)
    groups = attrs.groups
    layout = attrs.data_layout
    kernel_layout = attrs.kernel_layout
    if dilation_h < 1 or dilation_w < 1:
        raise ValueError("dilation should be positive value")

    need_auto_scheduler_layout = is_auto_scheduler_enabled()
    need_meta_schedule_layout = is_meta_schedule_enabled()

    if groups == 1:
        if layout == "NCHW":
            assert kernel_layout == "OIHW"
            if topi.riscv.is_int8_hw_support(data.dtype, kernel.dtype):
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.riscv.conv2d_nchw_int8),
                    wrap_topi_schedule(topi.riscv.schedule_conv2d_nchw_int8),
                    name="conv2d_nchw_int8.riscv",
                )
            else:
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.riscv.conv2d_nchw),
                    wrap_topi_schedule(topi.riscv.schedule_conv2d_nchw),
                    name="conv2d_nchw.riscv",
                )
        elif _NCHWc_matcher.match(layout):  # check if layout is NCHWxc
            assert _OIHWio_matcher.match(kernel_layout)  # check if kernel is OIHWio
            return conv2d_NCHWc_strategy_cpu(attrs, inputs, out_type, target)
        elif layout == "NHWC":
            assert kernel_layout == "HWIO"
            if (not need_auto_scheduler_layout) and (not need_meta_schedule_layout):
                logger.info("conv2d NHWC layout is not optimized for riscv with autotvm.")
            else:
                strategy.add_implementation(
                    wrap_compute_conv2d(
                        topi.nn.conv2d_nhwc,
                        need_auto_scheduler_layout=need_auto_scheduler_layout,
                        need_meta_schedule_layout=need_meta_schedule_layout,
                    ),
                    wrap_topi_schedule(topi.riscv.schedule_conv2d_nhwc),
                    name="conv2d_nhwc.riscv",
                )

            judge_winograd_auto_scheduler = False
            if len(kernel.shape) == 4:
                kernel_h, kernel_w, _, co = get_const_tuple(kernel.shape)
                judge_winograd_auto_scheduler = (
                    "float" in data.dtype
                    and "float" in kernel.dtype
                    and kernel_h == 3
                    and kernel_w == 3
                    and stride_h == 1
                    and stride_w == 1
                    and dilation_h == 1
                    and dilation_w == 1
                    and 64 < co < 512
                    # The last condition of co is based on our profiling of resnet workloads
                    # on skylake avx512 machines. We found winograd is faster than direct
                    # only when co is within this range
                )

            # register auto-scheduler implementations
            if (
                need_auto_scheduler_layout or need_meta_schedule_layout
            ) and judge_winograd_auto_scheduler:
                strategy.add_implementation(
                    wrap_compute_conv2d(
                        topi.nn.conv2d_winograd_nhwc,
                        need_auto_scheduler_layout=need_auto_scheduler_layout,
                        need_meta_schedule_layout=need_meta_schedule_layout,
                    ),
                    naive_schedule,  # this implementation should never be picked by autotvm
                    name="conv2d_nhwc.winograd",
                    plevel=15,
                )
        elif layout == "HWCN":
            assert kernel_layout == "HWIO"
            if (not need_auto_scheduler_layout) or (not need_meta_schedule_layout):
                logger.info("conv2d HWCN layout is not optimized for riscv with autotvm.")
            strategy.add_implementation(
                wrap_compute_conv2d(topi.nn.conv2d_hwcn),
                wrap_topi_schedule(topi.generic.schedule_conv2d_hwcn),
                name="conv2d_hwcn.generic",
            )
        else:
            raise RuntimeError("Unsupported conv2d layout {} for riscv".format(layout))
    elif is_depthwise_conv2d(data.shape, layout, kernel.shape, kernel_layout, groups):
        if layout == "NCHW":
            assert kernel_layout == "OIHW"
            channel_multiplier = get_const_tuple(inputs[1].shape)[1]
            if channel_multiplier == 1 and dilation_h == 1 and dilation_w == 1:
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.riscv.depthwise_conv2d_nchw),
                    wrap_topi_schedule(topi.riscv.schedule_depthwise_conv2d_nchw),
                    name="depthwise_conv2d_nchw.riscv",
                )
            else:
                logger.warning(
                    "For riscv target, depthwise_conv2d with channel "
                    "multiplier greater than 1 is not optimized"
                )
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.nn.depthwise_conv2d_nchw),
                    wrap_topi_schedule(topi.generic.schedule_depthwise_conv2d_nchw),
                    name="depthwise_conv2d_nchw.generic",
                )
        elif _NCHWc_matcher.match(layout):  # check if layout is NCHWxc
            assert _OIHWio_matcher.match(kernel_layout)  # check if kernel is OIHWio
            return depthwise_conv2d_NCHWc_strategy_cpu(attrs, inputs, out_type, target)
        elif layout == "NHWC":
            assert kernel_layout == "HWOI"
            if (not need_auto_scheduler_layout) and (not need_meta_schedule_layout):
                logger.info("depthwise_conv2d NHWC layout is not optimized for riscv with autotvm.")
            strategy.add_implementation(
                wrap_compute_conv2d(topi.nn.depthwise_conv2d_nhwc),
                wrap_topi_schedule(topi.generic.schedule_depthwise_conv2d_nhwc),
                name="depthwise_conv2d_nhwc.generic",
            )
        else:
            raise RuntimeError("Unsupported depthwise_conv2d layout {}".format(layout))
    else:  # group_conv2d
        if layout == "NCHW":
            assert kernel_layout == "OIHW"
            # logger.warning("group_conv2d is not optimized for riscv.")
            strategy.add_implementation(
                wrap_compute_conv2d(topi.riscv.group_conv2d_nchw, has_groups=True),
                wrap_topi_schedule(topi.riscv.schedule_group_conv2d_nchw),
                name="group_conv2d_nchw.riscv",
            )
        elif layout == "NHWC":
            assert kernel_layout == "HWIO"
            if (not need_auto_scheduler_layout) and (not need_meta_schedule_layout):
                logger.warning("group_conv2d is not optimized for riscv with autotvm.")
            strategy.add_implementation(
                wrap_compute_conv2d(topi.nn.group_conv2d_nhwc, has_groups=True),
                wrap_topi_schedule(topi.generic.schedule_group_conv2d_nhwc),
                name="group_conv2d_nhwc.generic",
            )
        else:
            raise RuntimeError("Unsupported group_conv2d layout {}".format(layout))
    return strategy


@conv2d_NCHWc_strategy.register("riscv")
def conv2d_NCHWc_strategy_cpu(attrs, inputs, out_type, target):
    """conv2d_NCHWc riscv strategy"""
    strategy = _op.OpStrategy()
    data, kernel = inputs
    if topi.riscv.is_int8_hw_support(data.dtype, kernel.dtype):
        strategy.add_implementation(
            wrap_compute_conv2d(topi.riscv.conv2d_NCHWc_int8, True, True),
            wrap_topi_schedule(topi.riscv.schedule_conv2d_NCHWc_int8),
            name="conv2d_NCHWc_int8.riscv",
        )
    else:
        strategy.add_implementation(
            wrap_compute_conv2d(topi.riscv.conv2d_NCHWc, True, True),
            wrap_topi_schedule(topi.riscv.schedule_conv2d_NCHWc),
            name="conv2d_NCHWc.riscv",
        )
    return strategy


@depthwise_conv2d_NCHWc_strategy.register("riscv")
def depthwise_conv2d_NCHWc_strategy_cpu(attrs, inputs, out_type, target):
    """depthwise_conv2d riscv strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_conv2d(topi.riscv.depthwise_conv2d_NCHWc, True, True),
        wrap_topi_schedule(topi.riscv.schedule_depthwise_conv2d_NCHWc),
        name="depthwise_conv2d_NCHWc.riscv",
    )
    return strategy


@conv2d_transpose_strategy.register("riscv")
def conv2d_transpose_strategy_cpu(attrs, inputs, out_type, target):
    """conv2d_transpose riscv strategy"""
    layout = attrs.data_layout
    dilation = get_const_tuple(attrs.dilation)
    groups = attrs.groups
    assert layout == "NCHW", "only support nchw for now"
    assert dilation == (1, 1), "not support dilate now"
    strategy = _op.OpStrategy()
    if groups == 1:
        strategy.add_implementation(
            wrap_compute_conv2d_transpose(topi.riscv.conv2d_transpose_nchw),
            wrap_topi_schedule(topi.riscv.schedule_conv2d_transpose_nchw),
            name="conv2d_transpose_nchw.riscv",
        )
    else:
        strategy.add_implementation(
            wrap_compute_conv2d_transpose(topi.nn.group_conv2d_transpose_nchw, has_groups=True),
            wrap_topi_schedule(topi.generic.schedule_group_conv2d_transpose_nchw),
            name="group_conv2d_transpose_nchw.riscv",
        )
    return strategy


@conv3d_transpose_strategy.register("riscv")
def conv3d_transpose_strategy_cpu(attrs, inputs, out_type, target):
    """conv3d_transpose riscv strategy"""
    layout = attrs.data_layout
    dilation = get_const_tuple(attrs.dilation)
    groups = attrs.groups
    assert layout == "NCDHW", "only support ncdhw for now"
    assert dilation == (1, 1, 1), "not support dilate now"
    assert groups == 1, "only support groups == 1 for now"
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_conv3d_transpose(topi.riscv.conv3d_transpose_ncdhw),
        wrap_topi_schedule(topi.riscv.schedule_conv3d_transpose_ncdhw),
        name="conv3d_transpose_ncdhw.riscv",
    )
    return strategy


@conv3d_strategy.register("riscv")
def conv3d_strategy_cpu(attrs, inputs, out_type, target):
    """conv3d generic strategy"""
    strategy = _op.OpStrategy()
    layout = attrs.data_layout
    need_auto_scheduler_layout = is_auto_scheduler_enabled()
    need_meta_schedule_layout = is_meta_schedule_enabled()
    if need_auto_scheduler_layout or need_meta_schedule_layout:
        # Use auto-scheduler. We should provide clear compute definition without autotvm templates
        # or packed layouts.
        if layout == "NCDHW":
            strategy.add_implementation(
                wrap_compute_conv3d(topi.nn.conv3d_ncdhw),
                naive_schedule,
                name="conv3d_ncdhw.riscv",
            )
        elif layout == "NDHWC":
            strategy.add_implementation(
                wrap_compute_conv3d(
                    topi.nn.conv3d_ndhwc,
                    need_auto_scheduler_layout=need_auto_scheduler_layout,
                    need_meta_schedule_layout=need_meta_schedule_layout,
                ),
                naive_schedule,
                name="conv3d_ndhwc.riscv",
            )
        else:
            raise ValueError("Not support this layout {} yet".format(layout))
    else:
        # Use autotvm templates
        if layout == "NCDHW":
            strategy.add_implementation(
                wrap_compute_conv3d(topi.riscv.conv3d_ncdhw),
                wrap_topi_schedule(topi.riscv.schedule_conv3d_ncdhw),
                name="conv3d_ncdhw.riscv",
            )
        elif layout == "NDHWC":
            strategy.add_implementation(
                wrap_compute_conv3d(topi.riscv.conv3d_ndhwc),
                wrap_topi_schedule(topi.riscv.schedule_conv3d_ndhwc),
                name="conv3d_ndhwc.riscv",
            )
        else:
            raise ValueError("Not support this layout {} yet".format(layout))
    return strategy


@conv1d_strategy.register("riscv")
def conv1d_strategy_cpu(attrs, inputs, out_type, target):
    """conv1d riscv strategy"""
    layout = attrs.data_layout
    groups = attrs.groups
    dilation = get_const_tuple(attrs.dilation)
    if dilation[0] < 1:
        raise ValueError("dilation should be a positive value")
    strategy = _op.OpStrategy()
    if groups == 1:
        if layout == "NCW":
            strategy.add_implementation(
                wrap_compute_conv1d(topi.nn.conv1d_ncw),
                wrap_topi_schedule(topi.riscv.schedule_conv1d_ncw),
                name="conv1d_ncw.riscv",
            )
        elif layout == "NWC":
            strategy.add_implementation(
                wrap_compute_conv1d(topi.nn.conv1d_nwc),
                wrap_topi_schedule(topi.riscv.schedule_conv1d_nwc),
                name="conv1d_nwc.riscv",
            )
        else:
            raise ValueError("Unsupported conv1d layout {}".format(layout))
    else:
        if layout == "NCW":
            strategy.add_implementation(
                wrap_compute_group_conv1d(topi.nn.group_conv1d_ncw),
                wrap_topi_schedule(topi.riscv.schedule_group_conv1d_ncw),
                name="group_conv1d_ncw.riscv",
            )
        elif layout == "NWC":
            strategy.add_implementation(
                wrap_compute_group_conv1d(topi.nn.group_conv1d_nwc),
                wrap_topi_schedule(topi.riscv.schedule_group_conv1d_nwc),
                name="group_conv1d_nwc.riscv",
            )
        else:
            raise ValueError("Unsupported conv1d layout {}".format(layout))
    return strategy


@matmul_strategy.register("riscv")
def matmul_strategy_cpu(attrs, inputs, out_type, target):
    """matmul riscv strategy"""
    strategy = _op.OpStrategy()

    same_type = inputs[0].dtype == inputs[1].dtype == out_type.dtype
    dtype = inputs[0].dtype
    u8s8s32 = dtype == "uint8" and inputs[1].dtype == "int8" and out_type.dtype == "int32"
    if "cblas" in target.libs:
        length_before = len(strategy.specializations) if strategy.specializations else 0
        with SpecializedCondition(same_type and dtype in ["float32", "float64"]):
            strategy.add_implementation(
                wrap_compute_matmul(topi.riscv.matmul_cblas),
                wrap_topi_schedule(topi.riscv.schedule_matmul_cblas),
                name="matmul_cblas.riscv",
                plevel=13,
            )
        length_after = len(strategy.specializations) if strategy.specializations else 0
        if length_before == length_after:
            logger.warning(
                "Currently cblas only support the data type to be float32 or float64. Skip."
            )
    if "mkl" in target.libs:
        length_before = len(strategy.specializations) if strategy.specializations else 0
        with SpecializedCondition(same_type and dtype in ["float32", "float64"] or u8s8s32):
            strategy.add_implementation(
                wrap_compute_matmul(topi.riscv.matmul_mkl),
                wrap_topi_schedule(topi.riscv.schedule_matmul_mkl),
                name="matmul_mkl.riscv",
                plevel=14,
            )
        length_after = len(strategy.specializations) if strategy.specializations else 0
        if length_before == length_after:
            logger.warning(
                "Currently mkl only support the data type to be float32, float64 or input with "
                "uint8 and int8 while output wiht int32. Skip."
            )

    need_auto_scheduler_layout = is_auto_scheduler_enabled()
    need_meta_schedule_layout = is_meta_schedule_enabled()
    if need_auto_scheduler_layout or need_meta_schedule_layout:
        strategy.add_implementation(
            wrap_compute_matmul(
                topi.nn.matmul,
                need_auto_scheduler_layout=need_auto_scheduler_layout,
                need_meta_schedule_layout=need_meta_schedule_layout,
            ),
            naive_schedule,
            name="matmul.generic",
            plevel=11,
        )
    else:
        # If no cblas/mkl/dnnl strategy choosed
        if not strategy.specializations:
            logger.warning(
                "Matmul is not optimized for riscv. "
                "Recommend to use cblas/mkl/dnnl for better performance."
            )
        strategy.add_implementation(
            wrap_compute_matmul(topi.nn.matmul),
            naive_schedule,
            name="matmul.generic",
        )
    return strategy


@dense_strategy.register("riscv")
def dense_strategy_cpu(attrs, inputs, out_type, target):
    """dense riscv strategy"""
    strategy = _op.OpStrategy()
    same_type = inputs[0].dtype == inputs[1].dtype == out_type.dtype
    dtype = inputs[0].dtype
    u8s8s32 = dtype == "uint8" and inputs[1].dtype == "int8" and out_type.dtype == "int32"
    strategy.add_implementation(
        wrap_compute_dense(topi.riscv.dense_nopack),
        wrap_topi_schedule(topi.riscv.schedule_dense_nopack),
        name="dense_nopack.riscv",
        plevel=5,
    )

    strategy.add_implementation(
        wrap_compute_dense(topi.riscv.dense_pack),
        wrap_topi_schedule(topi.riscv.schedule_dense_pack),
        name="dense_pack.riscv",
        plevel=10,
    )

    need_auto_scheduler_layout = is_auto_scheduler_enabled()
    need_meta_schedule_layout = is_meta_schedule_enabled()

    if need_auto_scheduler_layout or need_meta_schedule_layout:
        strategy.add_implementation(
            wrap_compute_dense(
                topi.nn.dense,
                need_auto_scheduler_layout=need_auto_scheduler_layout,
                need_meta_schedule_layout=need_meta_schedule_layout,
            ),
            naive_schedule,
            name="dense.generic",
            plevel=11,
        )

    if "cblas" in target.libs:
        with SpecializedCondition(same_type and dtype in ["float32", "float64"]):
            strategy.add_implementation(
                wrap_compute_dense(topi.riscv.dense_cblas),
                wrap_topi_schedule(topi.riscv.schedule_dense_cblas),
                name="dense_cblas.riscv",
                plevel=13,
            )
    if "mkl" in target.libs:
        with SpecializedCondition(same_type and dtype in ["float32", "float64"] or u8s8s32):
            strategy.add_implementation(
                wrap_compute_dense(topi.riscv.dense_mkl),
                wrap_topi_schedule(topi.riscv.schedule_dense_mkl),
                name="dense_mkl.riscv",
                plevel=14,
            )
    return strategy


@dense_pack_strategy.register("riscv")
def dense_pack_strategy_cpu(attrs, inputs, out_type, target):
    """dense_pack riscv strategy"""
    strategy = _op.OpStrategy()

    if (
        inputs[0].dtype == "uint8"
        and inputs[1].dtype == "int8"
        and out_type.dtype == "int32"
        and attrs["weight_layout"] == "NC16n4c"
    ):
        strategy.add_implementation(
            wrap_compute_dense(topi.riscv.dense_vnni),
            wrap_topi_schedule(topi.riscv.schedule_dense_vnni),
            name="dense_vnni.riscv",
            plevel=12,
        )
    else:
        strategy.add_implementation(
            wrap_compute_dense(topi.riscv.dense_pack),
            wrap_topi_schedule(topi.riscv.schedule_dense_pack),
            name="dense_pack.riscv",
            plevel=10,
        )

    return strategy


@batch_matmul_strategy.register("riscv")
def batch_matmul_strategy_cpu(attrs, inputs, out_type, target):
    """batch_matmul riscv strategy"""
    strategy = _op.OpStrategy()
    mcpu = Target.current().mcpu

    need_auto_scheduler_layout = is_auto_scheduler_enabled()
    need_meta_schedule_layout = is_meta_schedule_enabled()

    if (
        not attrs.transpose_a
        and attrs.transpose_b
        and target_has_vnni(mcpu)
        and inputs[0].dtype == "uint8"
        and inputs[1].dtype == "int8"
        and inputs[1].shape[-2] % 16 == 0
        and inputs[1].shape[-1] % 4 == 0
    ):
        strategy.add_implementation(
            wrap_compute_batch_matmul(topi.riscv.batch_matmul_vnni_compute, need_out_dtype=True),
            wrap_topi_schedule(topi.riscv.schedule_batch_matmul_vnni),
            name="batch_matmul_vnni.riscv",
            plevel=10,
        )
    elif is_dynamic(out_type) or need_auto_scheduler_layout or need_meta_schedule_layout:
        strategy.add_implementation(
            wrap_compute_batch_matmul(
                topi.nn.batch_matmul,
                need_out_dtype=True,
                need_auto_scheduler_layout=need_auto_scheduler_layout,
                need_meta_schedule_layout=need_meta_schedule_layout,
            ),
            wrap_topi_schedule(topi.generic.nn.schedule_batch_matmul),
            name="batch_matmul.generic",
            plevel=10,
        )
    else:
        strategy.add_implementation(
            wrap_compute_batch_matmul(topi.riscv.batch_matmul, need_out_dtype=True),
            wrap_topi_schedule(topi.riscv.schedule_batch_matmul),
            name="batch_matmul.riscv",
            plevel=10,
        )
    if "cblas" in target.libs:
        strategy.add_implementation(
            wrap_compute_batch_matmul(topi.riscv.batch_matmul_cblas),
            wrap_topi_schedule(topi.riscv.schedule_batch_matmul_cblas),
            name="batch_matmul_cblas.riscv",
            plevel=15,
        )
    if "mkl" in target.libs:
        strategy.add_implementation(
            wrap_compute_batch_matmul(topi.riscv.batch_matmul_mkl),
            wrap_topi_schedule(topi.riscv.schedule_batch_matmul_mkl),
            name="batch_matmul_mkl.riscv",
            plevel=15,
        )
    return strategy


@sparse_dense_strategy.register("riscv")
def sparse_dense_strategy_cpu(attrs, inputs, out_type, target):
    """sparse dense riscv strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_sparse_dense(topi.nn.sparse_dense),
        wrap_topi_schedule(topi.riscv.schedule_sparse_dense),
        name="sparse_dense.riscv",
        plevel=10,
    )
    return strategy


@sparse_conv2d_strategy.register("riscv")
def sparse_conv2d_strategy_cpu(attrs, inputs, out_type, target):
    """sparse conv2d riscv strategy"""
    strategy = _op.OpStrategy()
    if attrs["kernel_size"][0] == 1:
        strategy.add_implementation(
            wrap_compute_sparse_conv2d(topi.nn.sparse_conv2d),
            wrap_topi_schedule(topi.generic.schedule_sparse_conv2d),
            name="sparse_conv2d.generic",
        )
    elif attrs["kernel_size"][0] == 3:
        if attrs["layout"] == "NHWC":
            strategy.add_implementation(
                wrap_compute_sparse_conv2d(topi.riscv.spconv2d_3x3_nhwc),
                wrap_topi_schedule(topi.riscv.schedule_spconv2d_3x3_nhwc),
                name="conv3x3_spNHWC.riscv",
            )
        elif attrs["layout"] == "NCHW":
            strategy.add_implementation(
                wrap_compute_sparse_conv2d(topi.riscv.spconv2d_3x3_nchw),
                wrap_topi_schedule(topi.riscv.schedule_spconv2d_3x3_nchw),
            )
    return strategy


@roi_align_strategy.register("riscv")
def roi_align_strategy_cpu(attrs, inputs, out_type, target):
    """roi_align riscv strategy"""
    strategy = _op.OpStrategy()
    layout = attrs.layout
    if layout == "NCHW":
        strategy.add_implementation(
            wrap_compute_roi_align(topi.riscv.roi_align_nchw),
            wrap_topi_schedule(topi.generic.schedule_roi_align),
            name="roi_align.riscv",
        )
    else:
        assert layout == "NHWC", "layout must be NCHW or NHWC."
        strategy.add_implementation(
            wrap_compute_roi_align(topi.vision.rcnn.roi_align_nhwc),
            wrap_topi_schedule(topi.generic.schedule_roi_align),
            name="roi_align.riscv",
        )
    return strategy


@bitserial_conv2d_strategy.register("riscv")
def bitserial_conv2d_strategy_cpu(attrs, inputs, out_type, target):
    """bitserial_conv2d riscv strategy"""
    strategy = _op.OpStrategy()
    layout = attrs.data_layout
    if layout == "NCHW":
        strategy.add_implementation(
            wrap_compute_bitserial_conv2d(topi.riscv.bitserial_conv2d_nchw),
            wrap_topi_schedule(topi.riscv.schedule_bitserial_conv2d_nchw),
            name="bitserial_conv2d_nchw.riscv",
        )
    elif layout == "NHWC":
        strategy.add_implementation(
            wrap_compute_bitserial_conv2d(topi.riscv.bitserial_conv2d_nhwc),
            wrap_topi_schedule(topi.riscv.schedule_bitserial_conv2d_nhwc),
            name="bitserial_conv2d_nhwc.riscv",
        )
    else:
        raise ValueError("Data layout {} not supported.".format(layout))
    return strategy


@bitserial_dense_strategy.register("riscv")
def bitserial_dense_strategy_cpu(attrs, inputs, out_type, target):
    """bitserial_dense riscv strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_bitserial_dense(topi.riscv.bitserial_dense),
        wrap_topi_schedule(topi.riscv.schedule_bitserial_dense),
        name="bitserial_dense.riscv",
    )
    return strategy


@scatter_nd_strategy.register("riscv")
def scatter_nd_strategy_cpu(attrs, inputs, out_type, target):
    """scatter_nd riscv strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_scatter_nd(topi.riscv.scatter_nd),
        wrap_topi_schedule(topi.generic.schedule_extern),
        name="scatter_nd.riscv",
        plevel=10,
    )
    return strategy


@conv2d_winograd_without_weight_transfrom_strategy.register("riscv")
def conv2d_winograd_without_weight_transfrom_strategy_cpu(attrs, inputs, out_type, target):
    """conv2d_winograd_without_weight_transfrom cpu strategy"""
    dilation = attrs.get_int_tuple("dilation")
    groups = attrs.get_int("groups")
    layout = attrs.data_layout
    strides = attrs.get_int_tuple("strides")
    assert dilation == (1, 1), "Do not support dilate now"
    assert strides == (1, 1), "Do not support strides now"
    assert groups == 1, "Do not supoort arbitrary group number"
    strategy = _op.OpStrategy()
    need_auto_scheduler_layout = is_auto_scheduler_enabled()
    need_meta_schedule_layout = is_meta_schedule_enabled()
    if layout == "NHWC":
        if need_meta_schedule_layout:
            strategy.add_implementation(
                wrap_compute_conv2d(
                    topi.nn.conv2d_winograd_nhwc_without_weight_transform,
                    need_auto_scheduler_layout=False,
                    need_meta_schedule_layout=True,
                ),
                naive_schedule,
                name="ansor.winograd",
            )
        elif need_auto_scheduler_layout:
            strategy.add_implementation(
                wrap_compute_conv2d(
                    topi.nn.conv2d_winograd_nhwc_without_weight_transform,
                    need_auto_scheduler_layout=True,
                    need_meta_schedule_layout=False,
                ),
                naive_schedule,
                name="ansor.winograd",
            )
        else:
            raise RuntimeError("Both AutoScheduler and MetaSchedule are not enabled")
    else:
        raise RuntimeError(
            "Unsupported conv2d_winograd_without_weight_transfrom layout {}".format(layout)
        )
    return strategy


@concatenate_strategy.register(["cpu"])
def concatenate_strategy_cpu(attrs, inputs, out_type, target):
    """concatenate riscv strategy"""
    strategy = _op.OpStrategy()
    use_only_old_concat = False
    for inpt in inputs:
        shape = inpt.shape
        for i in shape:
            if not isinstance(i, tir.expr.IntImm):
                use_only_old_concat = True
                break
    if use_only_old_concat:
        strategy.add_implementation(
            wrap_compute_concat(topi.transform.concatenate),
            wrap_topi_schedule(topi.riscv.injective.schedule_concatenate),
            name="concatenate.generic",
        )
    else:
        strategy.add_implementation(
            wrap_compute_concat(topi.riscv.concatenate),
            wrap_topi_schedule(topi.riscv.schedule_concatenate_cpu),
            name="concatenate.cpu",
        )
        strategy.add_implementation(
            wrap_compute_concat(topi.transform.concatenate),
            wrap_topi_schedule(topi.riscv.injective.schedule_concatenate),
            name="concatenate.generic",
        )
    return strategy


@schedule_lrn.register(["riscv"])
def schedule_lrn_riscv(attrs, outs, target):
    """schedule LRN for riscv"""
    with target:
        return topi.riscv.schedule_lrn(outs)
