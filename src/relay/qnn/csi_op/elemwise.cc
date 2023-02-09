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
TVM_REGISTER_NODE_TYPE(QnnCSIUnaryAttrs);

bool QnnCSIElemwiseRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                       const TypeReporter& reporter) {
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;
  for (size_t i = 1; i < types.size(); ++i) {
    reporter->Assign(types[i], types[0]);
  }
  return true;
}

// QNN sin operator.
Expr MakeQnnCSISin(Expr data, DataType out_dtype, Array<Array<IndexExpr>> q_params,
                   String layer_name) {
  auto attrs = make_object<QnnCSIUnaryAttrs>();
  attrs->out_dtype = out_dtype;
  attrs->q_params = std::move(q_params);
  attrs->layer_name = std::move(layer_name);

  static const Op& op = Op::Get("qnn.csi.sin");
  return Call(op, {data}, Attrs(attrs), {});
}

RELAY_REGISTER_OP("qnn.csi.sin")
    .describe(R"code(Returns the sin input array, computed element-wise.

.. math::
   sin(x)

)code" TVM_ADD_FILELINE)
    .set_attrs_type<QnnCSIUnaryAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The quantized data tensor.")
    .set_support_level(11)
    .add_type_rel("QnnCSISinRel", QnnCSIElemwiseRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);

TVM_REGISTER_GLOBAL("relay.qnn.op._make.CSISin").set_body_typed(MakeQnnCSISin);

Expr MakeQnnCSICos(Expr data, DataType out_dtype, Array<Array<IndexExpr>> q_params,
                   String layer_name) {
  auto attrs = make_object<QnnCSIUnaryAttrs>();
  attrs->out_dtype = out_dtype;
  attrs->q_params = std::move(q_params);
  attrs->layer_name = std::move(layer_name);

  static const Op& op = Op::Get("qnn.csi.cos");
  return Call(op, {data}, Attrs(attrs), {});
}

RELAY_REGISTER_OP("qnn.csi.cos")
    .describe(R"code(Returns the cos input array, computed element-wise.

.. math::
   cos(x)

)code" TVM_ADD_FILELINE)
    .set_attrs_type<QnnCSIUnaryAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The quantized data tensor.")
    .set_support_level(11)
    .add_type_rel("QnnCSICosRel", QnnCSIElemwiseRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);

TVM_REGISTER_GLOBAL("relay.qnn.op._make.CSICos").set_body_typed(MakeQnnCSICos);

Expr MakeQnnCSITan(Expr data, DataType out_dtype, Array<Array<IndexExpr>> q_params,
                   String layer_name) {
  auto attrs = make_object<QnnCSIUnaryAttrs>();
  attrs->out_dtype = out_dtype;
  attrs->q_params = std::move(q_params);
  attrs->layer_name = std::move(layer_name);

  static const Op& op = Op::Get("qnn.csi.tan");
  return Call(op, {data}, Attrs(attrs), {});
}

RELAY_REGISTER_OP("qnn.csi.tan")
    .describe(R"code(Returns the tan input array, computed element-wise.

.. math::
   tan(x)

)code" TVM_ADD_FILELINE)
    .set_attrs_type<QnnCSIUnaryAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The quantized data tensor.")
    .set_support_level(11)
    .add_type_rel("QnnCSITanRel", QnnCSIElemwiseRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);

TVM_REGISTER_GLOBAL("relay.qnn.op._make.CSITan").set_body_typed(MakeQnnCSITan);

Expr MakeQnnCSIAsin(Expr data, DataType out_dtype, Array<Array<IndexExpr>> q_params,
                    String layer_name) {
  auto attrs = make_object<QnnCSIUnaryAttrs>();
  attrs->out_dtype = out_dtype;
  attrs->q_params = std::move(q_params);
  attrs->layer_name = std::move(layer_name);

  static const Op& op = Op::Get("qnn.csi.asin");
  return Call(op, {data}, Attrs(attrs), {});
}

RELAY_REGISTER_OP("qnn.csi.asin")
    .describe(R"code(Returns the asin input array, computed element-wise.

.. math::
   asin(x)

)code" TVM_ADD_FILELINE)
    .set_attrs_type<QnnCSIUnaryAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The quantized data tensor.")
    .set_support_level(11)
    .add_type_rel("QnnCSIAsinRel", QnnCSIElemwiseRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);

TVM_REGISTER_GLOBAL("relay.qnn.op._make.CSIAsin").set_body_typed(MakeQnnCSIAsin);

Expr MakeQnnCSIAcos(Expr data, DataType out_dtype, Array<Array<IndexExpr>> q_params,
                    String layer_name) {
  auto attrs = make_object<QnnCSIUnaryAttrs>();
  attrs->out_dtype = out_dtype;
  attrs->q_params = std::move(q_params);
  attrs->layer_name = std::move(layer_name);

  static const Op& op = Op::Get("qnn.csi.acos");
  return Call(op, {data}, Attrs(attrs), {});
}

RELAY_REGISTER_OP("qnn.csi.acos")
    .describe(R"code(Returns the acos input array, computed element-wise.

.. math::
   acos(x)

)code" TVM_ADD_FILELINE)
    .set_attrs_type<QnnCSIUnaryAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The quantized data tensor.")
    .set_support_level(11)
    .add_type_rel("QnnCSIAcosRel", QnnCSIElemwiseRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);

TVM_REGISTER_GLOBAL("relay.qnn.op._make.CSIAcos").set_body_typed(MakeQnnCSIAcos);

Expr MakeQnnCSIAtan(Expr data, DataType out_dtype, Array<Array<IndexExpr>> q_params,
                    String layer_name) {
  auto attrs = make_object<QnnCSIUnaryAttrs>();
  attrs->out_dtype = out_dtype;
  attrs->q_params = std::move(q_params);
  attrs->layer_name = std::move(layer_name);

  static const Op& op = Op::Get("qnn.csi.atan");
  return Call(op, {data}, Attrs(attrs), {});
}

RELAY_REGISTER_OP("qnn.csi.atan")
    .describe(R"code(Returns the atan input array, computed element-wise.

.. math::
   atan(x)

)code" TVM_ADD_FILELINE)
    .set_attrs_type<QnnCSIUnaryAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The quantized data tensor.")
    .set_support_level(11)
    .add_type_rel("QnnCSIAtanRel", QnnCSIElemwiseRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);

TVM_REGISTER_GLOBAL("relay.qnn.op._make.CSIAtan").set_body_typed(MakeQnnCSIAtan);

Expr MakeQnnCSISinh(Expr data, DataType out_dtype, Array<Array<IndexExpr>> q_params,
                    String layer_name) {
  auto attrs = make_object<QnnCSIUnaryAttrs>();
  attrs->out_dtype = out_dtype;
  attrs->q_params = std::move(q_params);
  attrs->layer_name = std::move(layer_name);

  static const Op& op = Op::Get("qnn.csi.sinh");
  return Call(op, {data}, Attrs(attrs), {});
}

RELAY_REGISTER_OP("qnn.csi.sinh")
    .describe(R"code(Returns the sinh input array, computed element-wise.

.. math::
   sinh(x)

)code" TVM_ADD_FILELINE)
    .set_attrs_type<QnnCSIUnaryAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The quantized data tensor.")
    .set_support_level(11)
    .add_type_rel("QnnCSISinhRel", QnnCSIElemwiseRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);

TVM_REGISTER_GLOBAL("relay.qnn.op._make.CSISinh").set_body_typed(MakeQnnCSISinh);

Expr MakeQnnCSICosh(Expr data, DataType out_dtype, Array<Array<IndexExpr>> q_params,
                    String layer_name) {
  auto attrs = make_object<QnnCSIUnaryAttrs>();
  attrs->out_dtype = out_dtype;
  attrs->q_params = std::move(q_params);
  attrs->layer_name = std::move(layer_name);

  static const Op& op = Op::Get("qnn.csi.cosh");
  return Call(op, {data}, Attrs(attrs), {});
}

RELAY_REGISTER_OP("qnn.csi.cosh")
    .describe(R"code(Returns the cosh input array, computed element-wise.

.. math::
   cosh(x)

)code" TVM_ADD_FILELINE)
    .set_attrs_type<QnnCSIUnaryAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The quantized data tensor.")
    .set_support_level(11)
    .add_type_rel("QnnCSICoshRel", QnnCSIElemwiseRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);

TVM_REGISTER_GLOBAL("relay.qnn.op._make.CSICosh").set_body_typed(MakeQnnCSICosh);

Expr MakeQnnCSITanh(Expr data, DataType out_dtype, Array<Array<IndexExpr>> q_params,
                    String layer_name) {
  auto attrs = make_object<QnnCSIUnaryAttrs>();
  attrs->out_dtype = out_dtype;
  attrs->q_params = std::move(q_params);
  attrs->layer_name = std::move(layer_name);

  static const Op& op = Op::Get("qnn.csi.tanh");
  return Call(op, {data}, Attrs(attrs), {});
}

RELAY_REGISTER_OP("qnn.csi.tanh")
    .describe(R"code(Returns the tanh input array, computed element-wise.

.. math::
   tanh(x)

)code" TVM_ADD_FILELINE)
    .set_attrs_type<QnnCSIUnaryAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The quantized data tensor.")
    .set_support_level(11)
    .add_type_rel("QnnCSITanhRel", QnnCSIElemwiseRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);

TVM_REGISTER_GLOBAL("relay.qnn.op._make.CSITanh").set_body_typed(MakeQnnCSITanh);

Expr MakeQnnCSIAsinh(Expr data, DataType out_dtype, Array<Array<IndexExpr>> q_params,
                     String layer_name) {
  auto attrs = make_object<QnnCSIUnaryAttrs>();
  attrs->out_dtype = out_dtype;
  attrs->q_params = std::move(q_params);
  attrs->layer_name = std::move(layer_name);

  static const Op& op = Op::Get("qnn.csi.asinh");
  return Call(op, {data}, Attrs(attrs), {});
}

RELAY_REGISTER_OP("qnn.csi.asinh")
    .describe(R"code(Returns the asinh input array, computed element-wise.

.. math::
   asinh(x)

)code" TVM_ADD_FILELINE)
    .set_attrs_type<QnnCSIUnaryAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The quantized data tensor.")
    .set_support_level(11)
    .add_type_rel("QnnCSIAsinhRel", QnnCSIElemwiseRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);

TVM_REGISTER_GLOBAL("relay.qnn.op._make.CSIAsinh").set_body_typed(MakeQnnCSIAsinh);

Expr MakeQnnCSIAcosh(Expr data, DataType out_dtype, Array<Array<IndexExpr>> q_params,
                     String layer_name) {
  auto attrs = make_object<QnnCSIUnaryAttrs>();
  attrs->out_dtype = out_dtype;
  attrs->q_params = std::move(q_params);
  attrs->layer_name = std::move(layer_name);

  static const Op& op = Op::Get("qnn.csi.acosh");
  return Call(op, {data}, Attrs(attrs), {});
}

RELAY_REGISTER_OP("qnn.csi.acosh")
    .describe(R"code(Returns the acosh input array, computed element-wise.

.. math::
   acosh(x)

)code" TVM_ADD_FILELINE)
    .set_attrs_type<QnnCSIUnaryAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The quantized data tensor.")
    .set_support_level(11)
    .add_type_rel("QnnCSIAcoshRel", QnnCSIElemwiseRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);

TVM_REGISTER_GLOBAL("relay.qnn.op._make.CSIAcosh").set_body_typed(MakeQnnCSIAcosh);

Expr MakeQnnCSIAtanh(Expr data, DataType out_dtype, Array<Array<IndexExpr>> q_params,
                     String layer_name) {
  auto attrs = make_object<QnnCSIUnaryAttrs>();
  attrs->out_dtype = out_dtype;
  attrs->q_params = std::move(q_params);
  attrs->layer_name = std::move(layer_name);

  static const Op& op = Op::Get("qnn.csi.atanh");
  return Call(op, {data}, Attrs(attrs), {});
}

RELAY_REGISTER_OP("qnn.csi.atanh")
    .describe(R"code(Returns the atanh input array, computed element-wise.

.. math::
   atanh(x)

)code" TVM_ADD_FILELINE)
    .set_attrs_type<QnnCSIUnaryAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The quantized data tensor.")
    .set_support_level(11)
    .add_type_rel("QnnCSIAtanhRel", QnnCSIElemwiseRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);

TVM_REGISTER_GLOBAL("relay.qnn.op._make.CSIAtanh").set_body_typed(MakeQnnCSIAtanh);

Expr MakeQnnCSIErf(Expr data, DataType out_dtype, Array<Array<IndexExpr>> q_params,
                   String layer_name) {
  auto attrs = make_object<QnnCSIUnaryAttrs>();
  attrs->out_dtype = out_dtype;
  attrs->q_params = std::move(q_params);
  attrs->layer_name = std::move(layer_name);

  static const Op& op = Op::Get("qnn.csi.erf");
  return Call(op, {data}, Attrs(attrs), {});
}

RELAY_REGISTER_OP("qnn.csi.erf")
    .describe(R"code(Returns the erf input array, computed element-wise.

.. math::
   erf(x)

)code" TVM_ADD_FILELINE)
    .set_attrs_type<QnnCSIUnaryAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The quantized data tensor.")
    .set_support_level(11)
    .add_type_rel("QnnCSIErfRel", QnnCSIElemwiseRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);

TVM_REGISTER_GLOBAL("relay.qnn.op._make.CSIErf").set_body_typed(MakeQnnCSIErf);

Expr MakeQnnCSISqrt(Expr data, DataType out_dtype, Array<Array<IndexExpr>> q_params,
                    String layer_name) {
  auto attrs = make_object<QnnCSIUnaryAttrs>();
  attrs->out_dtype = out_dtype;
  attrs->q_params = std::move(q_params);
  attrs->layer_name = std::move(layer_name);

  static const Op& op = Op::Get("qnn.csi.sqrt");
  return Call(op, {data}, Attrs(attrs), {});
}

RELAY_REGISTER_OP("qnn.csi.sqrt")
    .describe(R"code(Returns the sqrt input array, computed element-wise.

.. math::
   sqrt(x)

)code" TVM_ADD_FILELINE)
    .set_attrs_type<QnnCSIUnaryAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The quantized data tensor.")
    .set_support_level(11)
    .add_type_rel("QnnCSISqrtRel", QnnCSIElemwiseRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);

TVM_REGISTER_GLOBAL("relay.qnn.op._make.CSISqrt").set_body_typed(MakeQnnCSISqrt);

Expr MakeQnnCSIRsqrt(Expr data, DataType out_dtype, Array<Array<IndexExpr>> q_params,
                     String layer_name) {
  auto attrs = make_object<QnnCSIUnaryAttrs>();
  attrs->out_dtype = out_dtype;
  attrs->q_params = std::move(q_params);
  attrs->layer_name = std::move(layer_name);

  static const Op& op = Op::Get("qnn.csi.rsqrt");
  return Call(op, {data}, Attrs(attrs), {});
}

RELAY_REGISTER_OP("qnn.csi.rsqrt")
    .describe(R"code(Returns the sqrt input array, computed element-wise.

.. math::
   rsqrt(x)

)code" TVM_ADD_FILELINE)
    .set_attrs_type<QnnCSIUnaryAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The quantized data tensor.")
    .set_support_level(11)
    .add_type_rel("QnnCSIRsqrtRel", QnnCSIElemwiseRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);

TVM_REGISTER_GLOBAL("relay.qnn.op._make.CSIRsqrt").set_body_typed(MakeQnnCSIRsqrt);

Expr MakeQnnCSILog(Expr data, DataType out_dtype, Array<Array<IndexExpr>> q_params,
                   String layer_name) {
  auto attrs = make_object<QnnCSIUnaryAttrs>();
  attrs->out_dtype = out_dtype;
  attrs->q_params = std::move(q_params);
  attrs->layer_name = std::move(layer_name);

  static const Op& op = Op::Get("qnn.csi.log");
  return Call(op, {data}, Attrs(attrs), {});
}

RELAY_REGISTER_OP("qnn.csi.log")
    .describe(R"code(Returns the log input array, computed element-wise.

.. math::
   log(x)

)code" TVM_ADD_FILELINE)
    .set_attrs_type<QnnCSIUnaryAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The quantized data tensor.")
    .set_support_level(11)
    .add_type_rel("QnnCSILogRel", QnnCSIElemwiseRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);

TVM_REGISTER_GLOBAL("relay.qnn.op._make.CSILog").set_body_typed(MakeQnnCSILog);

Expr MakeQnnCSINegative(Expr data, DataType out_dtype,

                        Array<Array<IndexExpr>> q_params, String layer_name) {
  auto attrs = make_object<QnnCSIUnaryAttrs>();
  attrs->out_dtype = out_dtype;
  attrs->q_params = std::move(q_params);
  attrs->layer_name = std::move(layer_name);

  static const Op& op = Op::Get("qnn.csi.negative");
  return Call(op, {data}, Attrs(attrs), {});
}

RELAY_REGISTER_OP("qnn.csi.negative")
    .describe(R"code(Returns the negative input array, computed element-wise.

.. math::
   negative(x)

)code" TVM_ADD_FILELINE)
    .set_attrs_type<QnnCSIUnaryAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The quantized data tensor.")
    .set_support_level(11)
    .add_type_rel("QnnCSINegativeRel", QnnCSIElemwiseRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);

TVM_REGISTER_GLOBAL("relay.qnn.op._make.CSINegative").set_body_typed(MakeQnnCSINegative);

// Expr MakeQnnCSICumProd(Expr data, int32_t axis, bool exclusive=False, bool reverse,
//                          //                         DataType out_dtype) {
//   auto attrs = make_object<QnnCSIUnaryAttrs>();
//   attrs->out_dtype = out_dtype;
//   static const Op& op = Op::Get("qnn.csi.atanh");
//   return Call(op, {data}, Attrs(attrs), {});
// }

// RELAY_REGISTER_OP("qnn.csi.atanh")
// .describe(R"code(Returns the atanh input array, computed element-wise.

// .. math::
//    atanh(x)

// )code" TVM_ADD_FILELINE)
// .set_attrs_type<QnnCSIUnaryAttrs>()
// .set_num_inputs(1)
// .add_argument("data", "Tensor", "The quantized data tensor.")
// .set_support_level(11)
// .add_type_rel("QnnCSIAtanhRel", QnnCSIElemwiseRel)
// .set_attr<TOpPattern>("TOpPattern", kOpaque);

// TVM_REGISTER_GLOBAL("relay.qnn.op._make.CSIAtanh")
// .set_body_typed(MakeQnnCSIAtanh);

}  // namespace qnn
}  // namespace relay
}  // namespace tvm
