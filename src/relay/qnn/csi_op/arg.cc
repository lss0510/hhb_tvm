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

#include <numeric>

#include "../op/op_common.h"
#include "../utils.h"

namespace tvm {
namespace relay {
namespace qnn {
TVM_REGISTER_NODE_TYPE(QnnCSIReduceAttrs);

inline std::vector<int64_t> GetCSIReduceAxes(const uint32_t indim, const Array<Integer>& inaxis,
                                             bool exclude) {
  if (!inaxis.defined()) {
    std::vector<int64_t> r_axes(indim);
    for (uint32_t i = 0; i < indim; i++) {
      r_axes[i] = 0;
    }
    return r_axes;
  }

  std::vector<int64_t> in_axes;
  for (auto i : inaxis) {
    int64_t axis = i->value;
    if (axis < 0) {
      axis = axis + indim;
    }

    // Check out of bounds error
    CHECK(axis >= 0) << "Axis out of bounds in reduce operator.";
    CHECK(axis < indim) << "Axis out of bounds in reduce operator.";
    in_axes.push_back(axis);
  }

  CHECK(in_axes[in_axes.size() - 1] < indim)
      << "Reduction axis " << in_axes[in_axes.size() - 1] << " exceeds input dimensions " << indim;

  std::sort(in_axes.begin(), in_axes.end());

  if (!exclude) {
    return in_axes;
  }

  auto r_size = indim - in_axes.size();
  std::vector<int64_t> r_axes(r_size);
  for (uint32_t i = 0, j = 0, k = 0; i < indim; ++i) {
    if (j < in_axes.size() && in_axes[j] == i) {
      ++j;
      continue;
    }
    r_axes[k++] = i;
  }
  return r_axes;
}

inline std::vector<int64_t> GetReduceAxes(const uint32_t indim, const Array<Integer>& inaxis,
                                          bool exclude) {
  if (!inaxis.defined() || inaxis.empty()) {
    std::vector<int64_t> r_axes(indim);
    std::iota(r_axes.begin(), r_axes.end(), 0);
    return r_axes;
  }

  std::vector<int64_t> in_axes;
  for (auto i : inaxis) {
    int64_t axis = i->value;
    if (axis < 0) {
      axis = axis + indim;
    }

    // Check out of bounds error
    ICHECK(axis >= 0) << "Axis out of bounds in reduce operator.";
    ICHECK(axis < indim) << "Axis out of bounds in reduce operator.";
    in_axes.push_back(axis);
  }

  ICHECK(in_axes[in_axes.size() - 1] < indim)
      << "Reduction axis " << in_axes[in_axes.size() - 1] << " exceeds input dimensions " << indim;

  std::sort(in_axes.begin(), in_axes.end());

  if (!exclude) {
    return in_axes;
  }

  auto r_size = indim - in_axes.size();
  std::vector<int64_t> r_axes(r_size);
  for (uint32_t i = 0, j = 0, k = 0; i < indim; ++i) {
    if (j < in_axes.size() && in_axes[j] == i) {
      ++j;
      continue;
    }
    r_axes[k++] = i;
  }
  return r_axes;
}

inline std::vector<IndexExpr> ReduceShapeImpl(const std::vector<IndexExpr>& in_shape,
                                              const QnnCSIReduceAttrs* param,
                                              const TypeReporter& reporter) {
  uint32_t indim = in_shape.size();
  auto r_axes = GetReduceAxes(indim, param->axis, param->exclude);
  if (!r_axes.size()) {
    return in_shape;
  }

  auto max_shape = tir::make_const(DataType::Int(64), 1);
  bool is_dynamic_input = false;
  for (int64_t axis : r_axes) {
    if (in_shape[axis].as<IntImmNode>()) {
      max_shape *= in_shape[axis];
    } else {
      is_dynamic_input = true;
      break;
    }
  }

  if (is_dynamic_input) {
    ICHECK(reporter->Assert(
        max_shape < tir::make_const(DataType::Int(64), std::numeric_limits<int32_t>::max())))
        << "The maximum possible index of reduced shape cannot be more than int32 max.";
  }

  if (param->keepdims) {
    std::vector<IndexExpr> oshape(in_shape);
    for (unsigned i = 0, j = 0; i < indim; ++i) {
      if (j >= r_axes.size() || !(r_axes[j] == i)) {
        continue;
      }
      oshape[i] = 1;
      ++j;
    }
    return oshape;
  } else {
    auto osize = indim - r_axes.size();
    std::vector<IndexExpr> oshape(osize);
    for (unsigned i = 0, j = 0, k = 0; i < indim; ++i) {
      if (j < r_axes.size() && (r_axes[j] == i)) {
        ++j;
        continue;
      }
      oshape[k++] = in_shape[i];
    }
    return oshape;
  }
}

bool QnnCSIArgReduceRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                        const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;
  CHECK(static_cast<int>(data->shape.size()) != 0);
  std::vector<IndexExpr> in_shape(data->shape.begin(), data->shape.end());

  const QnnCSIReduceAttrs* param = attrs.as<QnnCSIReduceAttrs>();
  CHECK(param != nullptr);

  // assign output type and shape
  auto oshape = ReduceShapeImpl(in_shape, param, reporter);
  reporter->Assign(types[1], TensorType(oshape, DataType::Int(32)));
  return true;
}

// QNN argmax operator.
Expr MakeQnnCSIArgmax(Expr data, Array<Integer> axis, bool keepdims, bool exclude,
                      DataType out_dtype, Array<Array<IndexExpr>> q_params, String layer_name) {
  auto attrs = make_object<QnnCSIReduceAttrs>();

  attrs->axis = axis;
  attrs->keepdims = keepdims;
  attrs->exclude = exclude;

  attrs->out_dtype = out_dtype;
  attrs->q_params = std::move(q_params);
  attrs->layer_name = std::move(layer_name);

  static const Op& op = Op::Get("qnn.csi.argmax");
  return Call(op, {data}, Attrs(attrs), {});
}

RELAY_REGISTER_OP("qnn.csi.argmax")
    .describe(R"code(Returns the argmax input array.)code" TVM_ADD_FILELINE)
    .set_attrs_type<QnnCSIReduceAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The quantized data tensor.")
    .add_argument("axis", "Array", "axis.")
    .set_support_level(11)
    .add_type_rel("QnnCSIArgReduceRel", QnnCSIArgReduceRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);

TVM_REGISTER_GLOBAL("relay.qnn.op._make.CSIArgmax").set_body_typed(MakeQnnCSIArgmax);

// QNN argmin operator.
Expr MakeQnnCSIArgmin(Expr data, Array<Integer> axis, bool keepdims, bool exclude,
                      DataType out_dtype, Array<Array<IndexExpr>> q_params, String layer_name) {
  auto attrs = make_object<QnnCSIReduceAttrs>();

  attrs->axis = axis;
  attrs->keepdims = keepdims;
  attrs->exclude = exclude;

  attrs->out_dtype = out_dtype;
  attrs->q_params = std::move(q_params);
  attrs->layer_name = std::move(layer_name);

  static const Op& op = Op::Get("qnn.csi.argmin");
  return Call(op, {data}, Attrs(attrs), {});
}

RELAY_REGISTER_OP("qnn.csi.argmin")
    .describe(R"code(Returns the argmin input array.)code" TVM_ADD_FILELINE)
    .set_attrs_type<QnnCSIReduceAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The quantized data tensor.")
    .add_argument("axis", "Array", "axis.")
    .set_support_level(11)
    .add_type_rel("QnnCSIArgReduceRel", QnnCSIArgReduceRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);

TVM_REGISTER_GLOBAL("relay.qnn.op._make.CSIArgmin").set_body_typed(MakeQnnCSIArgmin);

}  // namespace qnn
}  // namespace relay
}  // namespace tvm
