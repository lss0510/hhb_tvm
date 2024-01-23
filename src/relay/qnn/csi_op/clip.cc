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

TVM_REGISTER_NODE_TYPE(QnnCSIClipAttrs);

bool QnnCSIClipRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                   const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) {
    CHECK(types[0].as<IncompleteTypeNode>())
        << "clip: expect input type to be TensorType but get " << types[0];
    return false;
  }
  const auto param = attrs.as<QnnCSIClipAttrs>();
  CHECK(param != nullptr);

  reporter->Assign(types[1], TensorType(data->shape, data->dtype));
  return true;
}

Expr MakeQnnCSIClip(Expr data, double a_min, double a_max, DataType out_dtype,
                    Array<Array<IndexExpr>> q_params, String layer_name) {
  auto attrs = make_object<QnnCSIClipAttrs>();
  attrs->a_min = a_min;
  attrs->a_max = a_max;
  attrs->out_dtype = out_dtype;
  attrs->q_params = std::move(q_params);
  attrs->layer_name = std::move(layer_name);

  static const Op& op = Op::Get("qnn.csi.clip");
  return Call(op, {data}, Attrs(attrs), {});
}

RELAY_REGISTER_OP("qnn.csi.clip")
    .describe(R"code(Returns element-wise smallest integer not less than x.

)code" TVM_ADD_FILELINE)
    .set_attrs_type<QnnCSIClipAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The quantized data tensor.")
    .set_support_level(11)
    .add_type_rel("QnnCSIClipRel", QnnCSIClipRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);

TVM_REGISTER_GLOBAL("relay.qnn.op._make.CSIClip").set_body_typed(MakeQnnCSIClip);

}  // namespace qnn
}  // namespace relay
}  // namespace tvm
