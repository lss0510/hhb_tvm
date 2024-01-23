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
"""riscv layout_transform operator"""
from tvm import te
from .utils import get_simd_32bit_lanes, get_simd_16bit_lanes
from .utils import intrin_layout_transform


def _schedule_layout_transform(s, out, src_layout, dst_layout):
    """Schedule for layout_transform.
    Parameters
    ----------
    s: Schedule
         The schedule to update.
    out: Tensor
         The tensor representing the layout_transform op.
    Returns
    -------
    s: Schedule
         The updated schedule.
    """

    def my_prod(shape):
        result = 1
        for num in shape:
            result *= num
        return result

    length = len(s[out].op.axis)
    input_shape = s[out].op.input_tensors[0].shape
    dtype = s[out].op.input_tensors[0].dtype
    fused = s[out].fuse(*s[out].op.axis[0:length])

    if dtype == "float32":
        simd_width = get_simd_32bit_lanes()
    else:
        simd_width = get_simd_16bit_lanes()
    factor = 1
    for tmp in range(simd_width, 0, -1):
        if out.shape[-1] % tmp == 0:
            factor = tmp
            break

    index = -1
    digit_num = 0

    while src_layout[index].upper() != dst_layout[-1].upper():
        if src_layout[index].isdigit():
            digit_num += 1
        index -= 1
    num = index + 1 + digit_num
    stride = 1 if index == -1 else my_prod(input_shape[num:])

    lo, li = s[out].split(fused, factor)
    s[out].parallel(lo)
    load_intrin = intrin_layout_transform(factor, dtype, stride, -num)
    s[out].tensorize(li, load_intrin)

    return s


def schedule_layout_transform(outs, attrs):
    """Schedule for layout_transform.

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of layout_transform
          in the format of an array of tensors.
    attrs: Attrs of the layout_transform.

    Returns
    -------
    s: Schedule
        The computation schedule for layout_transform.
    """
    src_layout = attrs.src_layout
    dst_layout = attrs.dst_layout
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    _schedule_layout_transform(s, outs[0], src_layout, dst_layout)

    return s
