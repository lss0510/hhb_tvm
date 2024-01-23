/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/relay/qnn/op/matmul.cc
 * \brief Property def of qnn matmul operator.
 */

#include <tvm/relay/base.h>
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/qnn/attrs.h>

#include "../../op/nn/nn.h"
#include "../utils.h"

namespace tvm {
namespace relay {
namespace qnn {

// relay.op.qnn.matmul
TVM_REGISTER_NODE_TYPE(QnnCSIMatMulAttrs);

bool QnnCSIMatMulRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                     const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 4);
  const auto* x = types[0].as<TensorTypeNode>();
  const auto* y = types[1].as<TensorTypeNode>();
  if (x == nullptr || y == nullptr) return false;

  auto param = attrs.as<QnnCSIMatMulAttrs>();
  ICHECK(param != nullptr);
  bool transpose_a = param->transpose_a;
  bool transpose_b = param->transpose_b;
  const Array<PrimExpr>& y_shape = y->shape;
  if (x->shape.size() == 3 && y_shape.size() == 3) {
    const PrimExpr& xb = x->shape[0];
    const PrimExpr& xi = x->shape[transpose_a ? 2 : 1];
    const PrimExpr& xk = x->shape[transpose_a ? 1 : 2];
    const PrimExpr& yb = y_shape[0];
    const PrimExpr& yk = y_shape[transpose_b ? 2 : 1];
    const PrimExpr& yj = y_shape[transpose_b ? 1 : 2];

    bool is_dyn = false;
    for (size_t i = 0; i < 3; ++i) {
      if (x->shape[i].as<tir::AnyNode>() != nullptr || y_shape[i].as<tir::AnyNode>() != nullptr) {
        is_dyn = true;
        break;
      }
    }
    if (!is_dyn) {
      ICHECK(reporter->AssertEQ(xb, yb) || reporter->AssertEQ(xb, 1) || reporter->AssertEQ(yb, 1))
          << "BatchDot: batch dimensions don't match, "
          << " x shape=" << x->shape << ", y shape=" << y_shape;
      ICHECK(reporter->AssertEQ(xk, yk)) << "BatchDot: shapes of x and y is inconsistent, "
                                         << " x shape=" << x->shape << ", y shape=" << y_shape;
    }
    const auto& out_b =
        xb->IsInstance<tir::AnyNode>() || yb->IsInstance<tir::AnyNode>() ? tir::Any() : max(xb, yb);
    reporter->Assign(types[3], TensorType(Array<tvm::PrimExpr>({out_b, xi, yj}), x->dtype));
  } else if (x->shape.size() == 4 && y_shape.size() == 4) {
    const PrimExpr& xb = x->shape[0];
    const PrimExpr& xc = x->shape[1];
    const PrimExpr& xi = x->shape[transpose_a ? 3 : 2];
    const PrimExpr& xk = x->shape[transpose_a ? 2 : 3];
    const PrimExpr& yb = y_shape[0];
    const PrimExpr& yc = y_shape[1];
    const PrimExpr& yk = y_shape[transpose_b ? 3 : 2];
    const PrimExpr& yj = y_shape[transpose_b ? 2 : 3];
    const auto& out_b =
        xb->IsInstance<tir::AnyNode>() || yb->IsInstance<tir::AnyNode>() ? tir::Any() : max(xb, yb);
    const auto& out_c =
        xb->IsInstance<tir::AnyNode>() || yb->IsInstance<tir::AnyNode>() ? tir::Any() : max(xc, yc);
    reporter->Assign(types[3], TensorType(Array<tvm::PrimExpr>({out_b, out_c, xi, yj}), x->dtype));
  } else {
    ICHECK(0) << "shape size is not 3 or 4";
  }
  return true;
}

Expr MakeQnnCSIMatMul(Expr data_a, Expr data_b, Expr bias, bool transpose_a, bool transpose_b,
                      DataType out_dtype, Array<Array<IndexExpr>> q_params, String layer_name) {
  auto attrs = make_object<QnnCSIMatMulAttrs>();
  attrs->transpose_a = std::move(transpose_a);
  attrs->transpose_b = std::move(transpose_b);

  attrs->out_dtype = std::move(out_dtype);
  attrs->q_params = std::move(q_params);
  attrs->layer_name = std::move(layer_name);
  static const Op& op = Op::Get("qnn.csi.matmul");
  return Call(op, {data_a, data_b, bias}, Attrs(attrs), {});
}

RELAY_REGISTER_OP("qnn.csi.matmul")
    .describe(R"code(Compute batch matrix multiplication of `tensor_a` and `tensor_b`.

Both `tensor_a` and `tensor_b` can be transposed. For legacy reason, we use NT format
(transpose_a=False, transpose_b=True) by default.

.. math::

  batch\_matmul(A, B)[i, :, :] = matmul(A[i, :, :], B[i, :, :]^T)

- **tensor_a**: `(b, m, k)` or `(b, k, m)`
- **tensor_b**: `(b, k, n)` or `(b, n, k)`
- **out**: `(b, m, n)`.

)code" TVM_ADD_FILELINE)
    .set_attrs_type<QnnCSIMatMulAttrs>()
    .set_num_inputs(3)
    .add_argument("tensor_a", "3D Tensor", "The first input.")
    .add_argument("tensor_b", "3D Tensor", "The second input.")
    .set_support_level(11)
    .add_type_rel("QnnCSIMatMulRel", QnnCSIMatMulRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);

TVM_REGISTER_GLOBAL("relay.qnn.op._make.CSIMatMul").set_body_typed(MakeQnnCSIMatMul);

}  // namespace qnn
}  // namespace relay
}  // namespace tvm
