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
"""riscv reshape operator"""
from tvm import te
from .utils import get_simd_32bit_lanes, get_simd_16bit_lanes
from .utils import intrin_reshape


def _schedule_reshape(s, out):
    """Schedule for reshape.
    Parameters
    ----------
    s: Schedule
         The schedule to update.
    out: Tensor
         The tensor representing the reshape op.
    Returns
    -------
    s: Schedule
         The updated sedule.
    """

    def my_prod(shape):
        result = 1
        for num in shape:
            result *= num
        return result

    length = len(s[out].op.axis)
    input_shape = s[out].op.input_tensors[0].shape
    last_axis = s[out].op.axis[-1]
    dtype = s[out].op.input_tensors[0].dtype

    if dtype == "float32":
        simd_width = get_simd_32bit_lanes()
    else:
        simd_width = get_simd_16bit_lanes()
    factor = 1
    for tmp in range(simd_width, 0, -1):
        if out.shape[-1] % tmp == 0 and input_shape[-1] % tmp == 0:
            factor = tmp
            break

    flag = 0
    # [1, 96, 56, 56] - > [1, 96, -1]
    # [96, 48, 3, 24, 24, 24] -> [48, 96, 3, -1] ...
    for n in range(2, len(input_shape) + 1):
        if len(input_shape) >= n and last_axis.dom.extent == my_prod(input_shape[-n:]):
            fused = s[out].fuse(*s[out].op.axis[0 : length - 1])
            s[out].parallel(fused)
            lo, li = s[out].split(last_axis, factor)
            s[out].parallel(lo)
            flag = 1
            break

    # other conditions
    if flag == 0:
        fused = s[out].fuse(*s[out].op.axis[0:length])
        lo, li = s[out].split(fused, factor)
        s[out].parallel(lo)

    load_intrin = intrin_reshape(input_shape, lo, factor, dtype, flag)
    s[out].tensorize(li, load_intrin)

    return s


def schedule_reshape(outs):
    """Schedule for reshape

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of reshape
          in the format of an array of tensors.
    Returns
    -------
    s: Schedule
        The computation schedule for reshape.
    """
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    _schedule_reshape(s, outs[0])

    return s
