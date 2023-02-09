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
 * \file src/relay/qnn/op/mul.cc
 * \brief QNN mul operator.
 */
#include <tvm/relay/analysis.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/qnn/attrs.h>

#include "../op/op_common.h"
#include "../utils.h"

namespace tvm {
namespace relay {
namespace qnn {

TVM_REGISTER_NODE_TYPE(QnnBinaryOpAttrs);

bool QnnCSIMinimumRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                      const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 3);
  const auto* data1 = types[0].as<TensorTypeNode>();
  const auto* data2 = types[1].as<TensorTypeNode>();
  if (data1 == nullptr) {
    CHECK(types[0].as<IncompleteTypeNode>())
        << "minimum or maximum: expect input type to be TensorType but get " << types[0];
    return false;
  }
  if (data2 == nullptr) {
    CHECK(types[1].as<IncompleteTypeNode>())
        << "minimum or maximum: expect input type to be TensorType but get " << types[1];
    return false;
  }

  std::vector<IndexExpr> in_shape1(data1->shape.begin(), data1->shape.end());
  std::vector<IndexExpr> in_shape2(data2->shape.begin(), data2->shape.end());
  std::vector<IndexExpr> oshape;

  oshape = in_shape1.size() > in_shape2.size() ? in_shape1 : in_shape2;

  reporter->Assign(types[2], TensorType(oshape, data1->dtype));
  return true;
}

Expr MakeQnnCSIMinimum(Expr lhs, Expr rhs, Array<Array<IndexExpr>> q_params, String layer_name) {
  auto attrs = make_object<QnnBinaryOpAttrs>();
  attrs->q_params = std::move(q_params);
  attrs->layer_name = std::move(layer_name);

  static const Op& op = Op::Get("qnn.csi.minimum");
  return Call(op, {lhs, rhs}, Attrs(attrs), {});
}

RELAY_REGISTER_OP("qnn.csi.minimum")
    .describe(R"code(Minimum the data into a new data type.

)code" TVM_ADD_FILELINE)
    .set_attrs_type<QnnBinaryOpAttrs>()
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The quantized data tensor.")
    .set_support_level(11)
    .add_type_rel("QnnCSIMinimumRel", QnnCSIMinimumRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);

TVM_REGISTER_GLOBAL("relay.qnn.op._make.CSIMinimum").set_body_typed(MakeQnnCSIMinimum);

Expr MakeQnnCSIMaximum(Expr lhs, Expr rhs, Array<Array<IndexExpr>> q_params, String layer_name) {
  auto attrs = make_object<QnnBinaryOpAttrs>();
  attrs->q_params = std::move(q_params);
  attrs->layer_name = std::move(layer_name);

  static const Op& op = Op::Get("qnn.csi.maximum");
  return Call(op, {lhs, rhs}, Attrs(attrs), {});
}

RELAY_REGISTER_OP("qnn.csi.maximum")
    .describe(R"code(Maximum the data into a new data type.

)code" TVM_ADD_FILELINE)
    .set_attrs_type<QnnBinaryOpAttrs>()
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The quantized data tensor.")
    .set_support_level(11)
    .add_type_rel("QnnCSIMinimumRel", QnnCSIMinimumRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);

TVM_REGISTER_GLOBAL("relay.qnn.op._make.CSIMaximum").set_body_typed(MakeQnnCSIMaximum);

}  // namespace qnn
}  // namespace relay
}  // namespace tvm
