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
# pylint: disable=invalid-name, unused-variable
"""Schedule for pooling operators"""
from tvm import te, tir
from .. import tag
from .utils import get_simd_32bit_lanes, load_vec, intrin_mul_vf, rv_intrin_dtype


def intrin_sum(l, dtype):
    """match math a[i] + b[i]"""
    a = te.placeholder((l,), name="a", dtype=dtype)
    c = te.compute((l,), lambda i: a[i], name="c")

    Ab = tir.decl_buffer(a.shape, a.dtype, name="A", offset_factor=1, strides=[1])
    Cb = tir.decl_buffer(c.shape, c.dtype, name="C", offset_factor=1, strides=[1])

    vl = get_simd_32bit_lanes()
    m = vl // 4
    rv_dtype = rv_intrin_dtype(dtype)

    def intrin_func(ins, outs):
        aa = ins[0]
        cc = outs[0]

        vec_c = cc.vload([0], dtype)

        def _body():
            ib = tir.ir_builder.create()
            vec_a = aa.access_ptr("r")
            vec_a = load_vec(vec_a, l, ib, dtype, "va")
            vsum = tir.call_extern(dtype, f"vfadd_vv_{rv_dtype}m{m}", vec_c, vec_a, l)
            ib.emit(cc.vstore([0], vsum))
            return ib.get()

        def _reduce_reset():
            ib = tir.ir_builder.create()

            init_value = te.const(0, dtype)

            ib.emit(tir.call_extern(dtype, f"vfmv_v_f_{rv_dtype}m{m}", init_value, l, vec_c))

            return ib.get()

        def _reduce_update():
            return _body()

        return _body(), _reduce_reset(), _reduce_update()

    return te.decl_tensor_intrin(c.op, intrin_func, binds={a: Ab, c: Cb})


def _parallel_sch(sch_pool, oshape, do_vectorize=False):
    vl = get_simd_32bit_lanes()

    def vectorize(fused_axis, num_parallel_axis, vectorize_limit=64):
        """Internal vectorization utility function."""
        reorder_axis = [fused_axis]
        for i in range(num_parallel_axis, len(sch_pool.op.axis) - 1):
            reorder_axis.append(sch_pool.op.axis[i])
        k = sch_pool.op.reduce_axis
        fuse_k = sch_pool.fuse(*k)
        c = sch_pool.op.axis[-1]
        reorder_axis += [fuse_k, c]
        sch_pool.reorder(*reorder_axis)
        inner_length = oshape[-1].value
        if len(oshape) != 5:
            return
        if inner_length <= vectorize_limit:
            vsum = intrin_sum(inner_length, sch_pool.op.input_tensors[0].dtype)
            sch_pool.tensorize(c, vsum)
        else:
            split_factor = 1
            for i in range(vectorize_limit, 1, -1):
                if inner_length % i == 0:
                    split_factor = i
                    break
            if split_factor > 1:
                _, c_i = sch_pool.split(c, split_factor)
                raise Exception("This part has not been handled yet.")

    if len(sch_pool.op.axis) >= 5:
        fused = sch_pool.fuse(sch_pool.op.axis[0], sch_pool.op.axis[1], sch_pool.op.axis[2])
        if do_vectorize:
            vectorize(fused, 3, vl)

    elif len(sch_pool.op.axis) >= 3:
        fused = sch_pool.fuse(sch_pool.op.axis[0], sch_pool.op.axis[1])
        if do_vectorize:
            vectorize(fused, 2, vl)
    else:
        sch_pool.parallel(sch_pool.op.axis[0])
        return


def schedule_pool(outs, attrs, layout):
    """Schedule for pool

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of pool
          in the format of an array of tensors.

    layout: str
        Data layout.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    scheduled_ops = []

    def _schedule(PaddedInput, Pool, do_vectorize):
        if isinstance(PaddedInput.op, te.tensor.ComputeOp):
            s[PaddedInput].compute_inline()
        do_vectorize &= layout != "DHWdhw"
        _parallel_sch(s[Pool], outs[0].shape, do_vectorize)

    def traverse(OP):
        """Internal traverse function"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_injective(OP.tag):
            if OP not in s.outputs:
                s[OP].compute_inline()
            for tensor in OP.input_tensors:
                if isinstance(tensor.op, te.tensor.ComputeOp) and tensor.op not in scheduled_ops:
                    traverse(tensor.op)
        # schedule pool
        elif OP.tag.startswith("pool"):
            # Average pool accumulation and division happens in different for loops (#3607).
            # To ensure good parallel support, apply multi-threading on the second loop.
            if OP != outs[0].op:
                output = outs[0]
                output_fused = s[output].fuse(output.op.axis[0], output.op.axis[1])
                s[output].parallel(output_fused)

            PaddedInput = OP.input_tensors[0]
            Pool = OP.output(0)
            do_vectorize = "max" not in s[Pool].op.tag
            do_vectorize &= all(attrs.padding) == 0
            do_vectorize &= len(outs[0].shape) == 5
            last_axis_length = s[OP.output(0)].op.axis[-1].dom.extent.value
            vl = get_simd_32bit_lanes()
            do_vectorize &= vl >= last_axis_length
            _schedule(PaddedInput, Pool, do_vectorize)
            if OP != outs[0].op:
                s[Pool].compute_at(s[outs[0]], outs[0].op.axis[3])
                s[outs[0]].unroll(outs[0].op.axis[-1])
                reduce_axes = s[Pool].op.reduce_axis
                if "sum" in s[Pool].op.tag:
                    if do_vectorize and last_axis_length != 1:
                        avg_value_h = reduce_axes[0].dom.extent.value
                        avg_value_w = reduce_axes[1].dom.extent.value
                        f_value = avg_value_h * avg_value_w
                        add_vf = intrin_mul_vf(
                            last_axis_length, 1 / f_value, outs[0].dtype, not do_vectorize
                        )
                        s[outs[0]].tensorize(outs[0].op.axis[-1], add_vf)
        else:
            raise RuntimeError("Unsupported operator: %s" % OP.tag)

        scheduled_ops.append(OP)

    traverse(outs[0].op)
    return s


def schedule_adaptive_pool(outs):
    """Schedule for adaptive pool

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of adaptive pool
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    scheduled_ops = []

    def traverse(OP):
        """Internal traverse function"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_injective(OP.tag):
            if OP not in s.outputs:
                s[OP].compute_inline()
            for tensor in OP.input_tensors:
                if isinstance(tensor.op, te.tensor.ComputeOp) and tensor.op not in scheduled_ops:
                    traverse(tensor.op)
        # schedule pool
        elif OP.tag.startswith("adaptive_pool"):
            if OP != outs[0].op:
                output = outs[0]
                output_fused = s[output].fuse(output.op.axis[0], output.op.axis[1])

            Pool = OP.output(0)
            _parallel_sch(s[Pool], outs[0].shape, True)
            if OP != outs[0].op:
                s[Pool].compute_at(s[outs[0]], outs[0].op.axis[3])
                s[outs[0]].unroll(outs[0].op.axis[-1])
                reduce_axes = s[Pool].op.reduce_axis
                if "sum" in s[Pool].op.tag:
                    inner_length = s[Pool].op.axis[-1].dom.extent.value
                    vl = get_simd_32bit_lanes()
                    if vl == inner_length:
                        avg_value_h = reduce_axes[0].dom.extent.b.b.value
                        avg_value_w = reduce_axes[1].dom.extent.b.b.value
                        f_value = avg_value_h * avg_value_w
                        # TODO: support better vectorize
                        add_vf = intrin_mul_vf(vl, 1 / f_value, outs[0].dtype)
                        s[outs[0]].tensorize(outs[0].op.axis[-1], add_vf)
        else:
            raise RuntimeError("Unsupported operator: %s" % OP.tag)

        scheduled_ops.append(OP)

    traverse(outs[0].op)
    return s
