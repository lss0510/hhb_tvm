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
TVM_REGISTER_NODE_TYPE(QnnCSISegmentAttrs);

bool QnnCSISegmentRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                      const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  const auto* segment_ids = types[1].as<TensorTypeNode>();

  if (data == nullptr) return false;
  if (segment_ids == nullptr) return false;
  const auto dshape = data->shape;
  const auto segment_shape = segment_ids->shape;

  CHECK(num_inputs == 2) << "num_inputs must be 2.";
  CHECK(segment_shape.size() == 1) << "segment_ids only support 1-D";

  const auto param = attrs.as<QnnCSISegmentAttrs>();
  CHECK(param != nullptr);

  std::vector<IndexExpr> oshape;
  if (dshape.size() == 1) {
    oshape.push_back(param->num_segments);
  } else if (dshape.size() == 2) {
    oshape.push_back(dshape[0]);
    oshape.push_back(param->num_segments);
  } else {
    oshape.push_back(param->num_segments);
    for (uint i = 1; i < dshape.size(); i++) {
      oshape.push_back(dshape[i]);
    }
  }

  // assign output type
  reporter->Assign(types[2], TensorType(oshape, data->dtype));
  return true;
}

// QNN segment max operator.
Expr MakeQnnCSISegmentMax(Expr data, Expr segment_ids, int32_t num_segments, DataType out_dtype,
                          Array<Array<IndexExpr>> q_params, String layer_name) {
  auto attrs = make_object<QnnCSISegmentAttrs>();
  attrs->num_segments = std::move(num_segments);
  attrs->out_dtype = out_dtype;
  attrs->q_params = std::move(q_params);
  attrs->layer_name = std::move(layer_name);

  static const Op& op = Op::Get("qnn.csi.segment_max");
  return Call(op, {data, segment_ids}, Attrs(attrs), {});
}

RELAY_REGISTER_OP("qnn.csi.segment_max")
    .describe(R"code(Returns segment_max operator.

.. math::
   segment_max(x, l)

)code" TVM_ADD_FILELINE)
    .set_attrs_type<QnnCSISegmentAttrs>()
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The quantized data tensor.")
    .add_argument("segment_ids", "Tensor",
                  "1-D tensor whose size is equal to the size"
                  "of data's first dimension. Values should be sorted and can be repeated.")
    .set_support_level(11)
    .add_type_rel("QnnCSISegmentMaxRel", QnnCSISegmentRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);

TVM_REGISTER_GLOBAL("relay.qnn.op._make.CSISegmentMax").set_body_typed(MakeQnnCSISegmentMax);

// QNN segment min operator.
Expr MakeQnnCSISegmentMin(Expr data, Expr segment_ids, int32_t num_segments, DataType out_dtype,
                          Array<Array<IndexExpr>> q_params, String layer_name) {
  auto attrs = make_object<QnnCSISegmentAttrs>();
  attrs->num_segments = std::move(num_segments);
  attrs->out_dtype = out_dtype;
  attrs->q_params = std::move(q_params);
  attrs->layer_name = std::move(layer_name);

  static const Op& op = Op::Get("qnn.csi.segment_min");
  return Call(op, {data, segment_ids}, Attrs(attrs), {});
}

RELAY_REGISTER_OP("qnn.csi.segment_min")
    .describe(R"code(Returns segment_min operator.

.. math::
   segment_min(x, l)

)code" TVM_ADD_FILELINE)
    .set_attrs_type<QnnCSISegmentAttrs>()
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The quantized data tensor.")
    .add_argument("segment_ids", "Tensor",
                  "1-D tensor whose size is equal to the size"
                  "of data's first dimension. Values should be sorted and can be repeated.")
    .set_support_level(11)
    .add_type_rel("QnnCSISegmentMinRel", QnnCSISegmentRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);

TVM_REGISTER_GLOBAL("relay.qnn.op._make.CSISegmentMin").set_body_typed(MakeQnnCSISegmentMin);

// QNN segment mean operator.
Expr MakeQnnCSISegmentMean(Expr data, Expr segment_ids, int32_t num_segments, DataType out_dtype,
                           Array<Array<IndexExpr>> q_params, String layer_name) {
  auto attrs = make_object<QnnCSISegmentAttrs>();
  attrs->num_segments = std::move(num_segments);
  attrs->out_dtype = out_dtype;
  attrs->q_params = std::move(q_params);
  attrs->layer_name = std::move(layer_name);

  static const Op& op = Op::Get("qnn.csi.segment_mean");
  return Call(op, {data, segment_ids}, Attrs(attrs), {});
}

RELAY_REGISTER_OP("qnn.csi.segment_mean")
    .describe(R"code(Returns segment_mean operator.

.. math::
   segment_mean(x, l)

)code" TVM_ADD_FILELINE)
    .set_attrs_type<QnnCSISegmentAttrs>()
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The quantized data tensor.")
    .add_argument("segment_ids", "Tensor",
                  "1-D tensor whose size is equal to the size"
                  "of data's first dimension. Values should be sorted and can be repeated.")
    .set_support_level(11)
    .add_type_rel("QnnCSISegmentMeanRel", QnnCSISegmentRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);

TVM_REGISTER_GLOBAL("relay.qnn.op._make.CSISegmentMean").set_body_typed(MakeQnnCSISegmentMean);

// QNN segment prod operator.
Expr MakeQnnCSISegmentProd(Expr data, Expr segment_ids, int32_t num_segments, DataType out_dtype,
                           Array<Array<IndexExpr>> q_params, String layer_name) {
  auto attrs = make_object<QnnCSISegmentAttrs>();
  attrs->num_segments = std::move(num_segments);
  attrs->out_dtype = out_dtype;
  attrs->q_params = std::move(q_params);
  attrs->layer_name = std::move(layer_name);

  static const Op& op = Op::Get("qnn.csi.segment_prod");
  return Call(op, {data, segment_ids}, Attrs(attrs), {});
}

RELAY_REGISTER_OP("qnn.csi.segment_prod")
    .describe(R"code(Returns segment_prod operator.

.. math::
   segment_prod(x, l)

)code" TVM_ADD_FILELINE)
    .set_attrs_type<QnnCSISegmentAttrs>()
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The quantized data tensor.")
    .add_argument("segment_ids", "Tensor",
                  "1-D tensor whose size is equal to the size"
                  "of data's first dimension. Values should be sorted and can be repeated.")
    .set_support_level(11)
    .add_type_rel("QnnCSISegmentProdRel", QnnCSISegmentRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);

TVM_REGISTER_GLOBAL("relay.qnn.op._make.CSISegmentProd").set_body_typed(MakeQnnCSISegmentProd);

// QNN segment sum operator.
Expr MakeQnnCSISegmentSum(Expr data, Expr segment_ids, int32_t num_segments, DataType out_dtype,
                          Array<Array<IndexExpr>> q_params, String layer_name) {
  auto attrs = make_object<QnnCSISegmentAttrs>();
  attrs->num_segments = std::move(num_segments);
  attrs->out_dtype = out_dtype;
  attrs->q_params = std::move(q_params);
  attrs->layer_name = std::move(layer_name);

  static const Op& op = Op::Get("qnn.csi.segment_sum");
  return Call(op, {data, segment_ids}, Attrs(attrs), {});
}

RELAY_REGISTER_OP("qnn.csi.segment_sum")
    .describe(R"code(Returns segment_sum operator.

.. math::
   segment_sum(x, l)

)code" TVM_ADD_FILELINE)
    .set_attrs_type<QnnCSISegmentAttrs>()
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The quantized data tensor.")
    .add_argument("segment_ids", "Tensor",
                  "1-D tensor whose size is equal to the size"
                  "of data's first dimension. Values should be sorted and can be repeated.")
    .set_support_level(11)
    .add_type_rel("QnnCSISegmentSumRel", QnnCSISegmentRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);

TVM_REGISTER_GLOBAL("relay.qnn.op._make.CSISegmentSum").set_body_typed(MakeQnnCSISegmentSum);

}  // namespace qnn
}  // namespace relay
}  // namespace tvm
