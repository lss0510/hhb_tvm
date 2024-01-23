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
# pylint: disable=invalid-name,too-many-locals,unused-variable
"""riscv nn operators"""
from tvm import te
from ..utils import traverse_inline
from .utils import (
    get_simd_32bit_lanes,
    get_simd_16bit_lanes,
    intrin_div,
    intrin_sub,
    intrin_sum,
    intrin_max,
)


def _schedule_softmax(softmax_op, s, outs):
    op_tag = softmax_op.tag
    if op_tag == "softmax_output":
        exp = softmax_op.input_tensors[0]
        expsum = softmax_op.input_tensors[1]
        sub = s[exp].op.input_tensors[0]
        max_elem = s[sub].op.input_tensors[1]
        delta = None
        axis = int(softmax_op.attrs["axis"])
    elif op_tag == "fast_softmax_output":
        exp = softmax_op.input_tensors[0]
        expsum = softmax_op.input_tensors[1]
        delta = s[exp].op.input_tensors[0]
        max_elem = s[delta].op.input_tensors[1]
        axis = int(softmax_op.attrs["axis"])
    elif op_tag == "log_softmax_output":
        exp = None
        delta = None
        max_elem = softmax_op.input_tensors[1]
        expsum = softmax_op.input_tensors[2]
        axis = 1
    else:
        raise ValueError(
            "Tag is expected to be softmax_output or log_softmax_output. \
                         Got {0}".format(
                op_tag
            )
        )

    # only parallelize outer dimensions up to axis
    outer_axes = [s[softmax_op].op.axis[i] for i in range(0, axis)]
    inner_axes = s[softmax_op].op.axis[len(outer_axes) :]
    fused_outer_axes = s[softmax_op].fuse(*outer_axes)

    s[softmax_op].parallel(fused_outer_axes)
    #
    ## move computations with the same outer dimensions under the same root
    s[max_elem].compute_at(s[softmax_op], fused_outer_axes)
    s[expsum].compute_at(s[softmax_op], fused_outer_axes)
    s[exp].compute_at(s[softmax_op], fused_outer_axes)
    s[sub].compute_at(s[softmax_op], fused_outer_axes)

    if op_tag == "softmax_output":
        dtype = softmax_op.input_tensors[0].dtype
        if dtype == "float32":
            simd_width = get_simd_32bit_lanes()
        else:
            simd_width = get_simd_16bit_lanes()

        factor = 1
        for tmp in range(simd_width, 0, -1):
            if exp.shape[-1] % tmp == 0:
                factor = tmp
                break
        flag = axis == len(s[softmax_op].op.axis) - 1
        inner_axes = s[sub].op.axis[-1]
        outer, inner = s[sub].split(inner_axes, factor)
        s[sub].parallel(outer)
        my_sub = intrin_sub(factor, dtype, flag)
        s[sub].tensorize(inner, my_sub)

        inner_axes = s[softmax_op].op.axis[-1]
        outer, inner = s[softmax_op].split(inner_axes, factor)
        s[softmax_op].parallel(outer)
        my_div = intrin_div(factor, dtype, flag)
        s[softmax_op].tensorize(inner, my_div)

        if flag:
            inner_axes = s[expsum].op.axis[-1]
            outer, inner = s[expsum].split(inner_axes, 1)
            s[expsum].parallel(outer)
            my_sum = intrin_sum(simd_width, exp.shape[axis], dtype)
            s[expsum].tensorize(inner, my_sum)

            inner_axes = s[max_elem].op.axis[-1]
            outer, inner = s[max_elem].split(inner_axes, 1)
            s[max_elem].parallel(outer)
            my_max = intrin_max(simd_width, exp.shape[axis], dtype)
            s[max_elem].tensorize(inner, my_max)

        else:
            if axis == 0:
                red_axes = s[expsum].op.axis[len(outer_axes)]
                inner_axes = s[expsum].op.axis[len(outer_axes) + 1 :]
                outer, inner = s[expsum].split(red_axes, expsum.shape[0])
                s[expsum].reorder(inner, *inner_axes, outer)
                s[expsum].parallel(outer)
                stride = 1
                num = len(s[softmax_op].op.axis) - axis
                for i in range(axis + 1, len(s[softmax_op].op.axis)):
                    stride *= exp.shape[i]
                my_sum = intrin_sum(simd_width, exp.shape[axis], dtype, num, stride)
                s[expsum].tensorize(outer, my_sum)

                red_axes = s[max_elem].op.axis[len(outer_axes)]
                inner_axes = s[max_elem].op.axis[len(outer_axes) + 1 :]
                outer, inner = s[max_elem].split(red_axes, max_elem.shape[0])
                s[max_elem].reorder(inner, *inner_axes, outer)
                s[max_elem].parallel(outer)
                stride = 1
                num = len(s[softmax_op].op.axis) - axis
                for i in range(axis + 1, len(s[softmax_op].op.axis)):
                    stride *= exp.shape[i]
                my_max = intrin_max(simd_width, exp.shape[axis], dtype, num, stride)
                s[max_elem].tensorize(outer, my_max)
            else:
                red_axes = s[expsum].op.axis[len(outer_axes) - 1]
                inner_axes = s[expsum].op.axis[len(outer_axes) :]
                outer, inner = s[expsum].split(red_axes, 1)
                s[expsum].reorder(*inner_axes, inner)
                s[expsum].parallel(outer)
                stride = 1
                num = len(s[softmax_op].op.axis) - axis
                for i in range(axis + 1, len(s[softmax_op].op.axis)):
                    stride *= exp.shape[i]
                my_sum = intrin_sum(simd_width, exp.shape[axis], dtype, num, stride)
                s[expsum].tensorize(inner, my_sum)

                red_axes = s[max_elem].op.axis[len(outer_axes) - 1]
                inner_axes = s[max_elem].op.axis[len(outer_axes) :]
                outer, inner = s[max_elem].split(red_axes, 1)
                s[max_elem].reorder(*inner_axes, inner)
                s[max_elem].parallel(outer)
                stride = 1
                num = len(s[softmax_op].op.axis) - axis
                for i in range(axis + 1, len(s[softmax_op].op.axis)):
                    stride *= exp.shape[i]
                my_max = intrin_max(simd_width, exp.shape[axis], dtype, num, stride)
                s[max_elem].tensorize(inner, my_max)

    if delta is not None:
        s[exp].compute_inline()
        s[delta].compute_inline()
    if exp is not None:
        s[exp].compute_at(s[softmax_op], fused_outer_axes)

    if softmax_op != outs[0].op:
        # fuse softmax output with following elemwise ops.
        output = outs[0]
        outer_axes = [s[output].op.axis[i] for i in range(0, axis)]
        fused_outer_axes = s[output].fuse(*outer_axes)
        s[output].parallel(fused_outer_axes)
        s[softmax_op].compute_at(s[output], fused_outer_axes)


def schedule_softmax(outs):
    """Schedule for softmax

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of softmax
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if "softmax" in op.tag:
            _schedule_softmax(op, s, outs)

    traverse_inline(s, outs[0].op, _callback)
    return s


def schedule_lrn(outs):
    """Schedule for LRN

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of LRN
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    if outs[0].dtype == "float32":
        max_threads = get_simd_32bit_lanes()
    else:
        max_threads = get_simd_16bit_lanes()

    def _callback(op):
        if "sqr_sum" in op.tag:
            pad = op.input_tensors[0]
            s[pad].compute_inline()
            fused_axis = s[outs[0]].fuse(*s[outs[0]].op.axis)

            bx, tx = s[outs[0]].split(fused_axis, factor=max_threads)
            s[op].compute_at(s[outs[0]], tx)

    traverse_inline(s, outs[0].op, _callback)
    return s
