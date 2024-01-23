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
 * \file src/relay/qnn/op/dense.cc
 * \brief Property def of qnn dense operator.
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

// relay.op.qnn.dense
TVM_REGISTER_NODE_TYPE(QnnCSIDenseAttrs);

bool QnnCSIDenseRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                    const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 4);
  const auto* tensor_a = types[0].as<TensorTypeNode>();
  const auto* tensor_b = types[1].as<TensorTypeNode>();
  if (tensor_a == nullptr) return false;
  ICHECK(static_cast<int>(tensor_a->shape.size()) != 0);

  const QnnCSIDenseAttrs* param = attrs.as<QnnCSIDenseAttrs>();
  ICHECK(param != nullptr);
  // Default set to dense layout
  bool transpose_a = false;
  bool transpose_b = true;
  const auto& mattrs = attrs.as<MatmulAttrs>();
  if (mattrs != nullptr) {
    transpose_a = mattrs->transpose_a;
    transpose_b = mattrs->transpose_b;
  }

  const Array<tvm::PrimExpr>& dshape = tensor_a->shape;
  Array<tvm::PrimExpr> oshape = dshape;
  tvm::PrimExpr reduce = dshape[dshape.size() - 1];
  if (transpose_a) {
    reduce = dshape[dshape.size() - 2];
    oshape.Set((oshape.size() - 2), dshape[oshape.size() - 1]);
  }
  if (param->units.defined()) {
    // validate the tensor_b shape is proper if defined
    // Assign tensor_b type
    const Array<IndexExpr>& wshape = transpose_b ? Array<IndexExpr>({param->units, reduce})
                                                 : Array<IndexExpr>({reduce, param->units});
    // It is possible for tensor_b to be nullptr in which case we will use
    // data dtype as the tensor_b dtype. However if tensor_b dtype is explicitly
    // present we will use that.
    oshape.Set((oshape.size() - 1), param->units);
  } else {
    if (tensor_b == nullptr) return false;
    const Array<tvm::PrimExpr>& wshape = tensor_b->shape;
    // When tensor_b's layout has been rewritten, figure it out based on the
    // total number of elements and input dimensions.
    ICHECK(static_cast<int>(tensor_b->shape.size()) == 2);
    if (!tensor_a->shape.back().as<tir::AnyNode>()) {
      ICHECK((transpose_b && reporter->AssertEQ(reduce, tensor_b->shape[1])) ||
             (!transpose_b && reporter->AssertEQ(reduce, tensor_b->shape[0])))
          << "MatmulRel: input dimension doesn't match,"
          << " tensor_a shape=" << tensor_a->shape << ", tensor_b shape=" << tensor_b->shape;
    }
    oshape.Set((oshape.size() - 1), transpose_b ? wshape[0] : wshape[1]);
  }

  // assign output type
  reporter->Assign(types[3], TensorType(oshape, tensor_a->dtype));
  return true;
}

Expr MakeQnnCSIDense(Expr data, Expr weight, Expr bias, IndexExpr units, DataType out_dtype,
                     Array<Array<IndexExpr>> q_params, String layer_name) {
  auto attrs = make_object<QnnCSIDenseAttrs>();
  attrs->units = std::move(units);
  attrs->out_dtype = std::move(out_dtype);

  attrs->q_params = std::move(q_params);
  attrs->layer_name = std::move(layer_name);
  static const Op& op = Op::Get("qnn.csi.dense");
  return Call(op, {data, weight, bias}, Attrs(attrs), {});
}

RELAY_REGISTER_OP("qnn.csi.dense")
    .describe(R"code(Applies a linear transformation: :math:`Y = XW^T`.
- **data**: quantized(int8, unit8) `(x1, x2, ..., xn, input_dim)`
- **weight**: quantized(int8, unit8) `(units, input_dim)`
- **bias**: quantized(int32) `(units, input_dim)`
- **out**: quantized(int32) `(x1, x2, ..., xn, units)`.
)code" TVM_ADD_FILELINE)
    .set_attrs_type<QnnCSIDenseAttrs>()
    .set_num_inputs(3)
    .add_argument("data", "quantized nD Tensor", "Input data.")
    .add_argument("weight", "quantized 2D Tensor", "Weight matrix.")
    .add_argument("bias", "quantized 2D Tensor", "bias matrix.")
    .set_support_level(11)
    .add_type_rel("QnnCSIDenseRel", QnnCSIDenseRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);

TVM_REGISTER_GLOBAL("relay.qnn.op._make.CSIDense").set_body_typed(MakeQnnCSIDense);

}  // namespace qnn
}  // namespace relay
}  // namespace tvm
