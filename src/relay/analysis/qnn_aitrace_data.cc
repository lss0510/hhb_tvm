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
 * \file qnn_aitrace_data.cc
 * \brief qnn aitrace data implementations.
 */

#include "qnn_aitrace_data.h"

#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/attrs/vision.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/qnn/attrs.h>

#include <iostream>
#include <string>

#include "../backend/utils.h"
using namespace tvm::relay::qnn;
namespace tvm {
namespace relay {
namespace aitrace {

//------------------------------------------------------------------------------
// Common utils implements
//------------------------------------------------------------------------------

AiTraceData QnnConvert2ATData(Array<AiTraceDataFrame> origin_data) {
  AiTraceData atdata;
  atdata.at_version_.major_ = 1;
  atdata.at_version_.minor_ = 8;
  atdata.at_version_.patch_ = 0;

  for (auto pd : origin_data) {
    AiTraceBlock atblock;

    String op_type = Downcast<String>(pd["op"]["type"]);

    std::string op_type_str = std::string(op_type);
    if (op_type_str == "unknown") {
      continue;
    }
    atblock.insn_type_ = qnn_map.at(op_type_str);
    atblock.insn_name_ = std::string(Downcast<String>(pd["op"]["name"]));

    // get calculation amount data
    if (pd.find("calculation_amount") != pd.end()) {
      atblock.at_cal_data_.have_cal_data_ = true;
      for (auto it : pd["calculation_amount"]) {
        if (it.first == "fused_mul_add") {
          atblock.at_cal_data_.fused_mul_add_ = int64_t(Downcast<Integer>(it.second));
        } else if (it.first == "mul") {
          atblock.at_cal_data_.mul_ = int64_t(Downcast<Integer>(it.second));
        } else if (it.first == "div") {
          atblock.at_cal_data_.div_ = int64_t(Downcast<Integer>(it.second));
        } else if (it.first == "add") {
          atblock.at_cal_data_.add_ = int64_t(Downcast<Integer>(it.second));
        } else if (it.first == "sub") {
          atblock.at_cal_data_.sub_ = int64_t(Downcast<Integer>(it.second));
        } else if (it.first == "exp") {
          atblock.at_cal_data_.exp_ = int64_t(Downcast<Integer>(it.second));
        } else if (it.first == "comp") {
          atblock.at_cal_data_.comp_ = int64_t(Downcast<Integer>(it.second));
        }
      }
    }
    // get memory data
    if (pd.find("memory") != pd.end()) {
      atblock.at_mem_data_.have_mem_data_ = true;
      for (auto it : pd["memory"]) {
        if (it.first == "params") {
          atblock.at_mem_data_.params_ = int64_t(Downcast<Integer>(it.second));
        } else if (it.first == "output") {
          atblock.at_mem_data_.output_ = int64_t(Downcast<Integer>(it.second));
        }
      }
    }
    atdata.at_block_.push_back(atblock);
  }
  return atdata;
}

//------------------------------------------------------------------------------
// QNN Add profiler implementation
//------------------------------------------------------------------------------

AiTraceDataFrame QnnAddProfiler::GetCalculationAmount(const Call& call_node) {
  const auto* attrs = call_node->attrs.as<QnnBinaryOpAttrs>();
  SetLayerName(attrs->layer_name);
  return GetEltwiseCalAmountCommon(call_node, "add");
}

AiTraceDataFrame QnnAddProfiler::GetMemory(const Call& call_node) {
  const auto* attrs = call_node->attrs.as<QnnBinaryOpAttrs>();
  SetLayerName(attrs->layer_name);
  return GetMemoryCommon(call_node);
}

RELAY_REGISTER_OP("qnn.csi.add")
    .set_attr<FCalAmount>("FCalAmount", QnnAddProfiler::GetCalculationAmount);
RELAY_REGISTER_OP("qnn.csi.add").set_attr<FMemory>("FMemory", QnnAddProfiler::GetMemory);
RELAY_REGISTER_OP("qnn.csi.add").set_attr<FOpName>("FOpName", QnnAddProfiler::GetLayerName);

//------------------------------------------------------------------------------
// QNN AvgPool2d profiler implementation
//------------------------------------------------------------------------------

AiTraceDataFrame QnnAvgPool2dProfiler::GetCalculationAmount(const Call& call_node) {
  const auto* avgpool2d_attr = call_node->attrs.as<QnnCSIAvgPool2DAttrs>();
  SetLayerName(avgpool2d_attr->layer_name);
  return GetPoolCalAmountCommon(call_node, avgpool2d_attr, avgpool2d_attr->pool_size, "avg", false);
}

AiTraceDataFrame QnnAvgPool2dProfiler::GetMemory(const Call& call_node) {
  const auto* attrs = call_node->attrs.as<QnnCSIAvgPool2DAttrs>();
  SetLayerName(attrs->layer_name);
  return GetMemoryCommon(call_node);
}

RELAY_REGISTER_OP("qnn.csi.avgpool2d")
    .set_attr<FCalAmount>("FCalAmount", QnnAvgPool2dProfiler::GetCalculationAmount);
RELAY_REGISTER_OP("qnn.csi.avgpool2d")
    .set_attr<FMemory>("FMemory", QnnAvgPool2dProfiler::GetMemory);
RELAY_REGISTER_OP("qnn.csi.avgpool2d")
    .set_attr<FOpName>("FOpName", QnnAvgPool2dProfiler::GetLayerName);

//------------------------------------------------------------------------------
// QNN BatchNorm profiler implementation
//------------------------------------------------------------------------------

// according to https://tvm.apache.org/docs/api/python/relay/nn.html#tvm.relay.nn.batch_norm
AiTraceDataFrame QnnBatchNormProfiler::GetCalculationAmount(const Call& call_node) {
  AiTraceDataFrame res;
  if (!call_node->checked_type_.defined()) {
    LOG(WARNING) << "The infer type pass should be called before the aitrace pass";
    return res;
  }
  Array<Expr> args = call_node->args;
  CHECK_EQ(args.size(), 5) << "The number of input arguments of a batch_norm node should be 5.";
  const auto* batch_norm_attr = call_node->attrs.as<QnnCSIBatchNormAttrs>();
  int axis = batch_norm_attr->axis;
  bool center = batch_norm_attr->center;
  bool scale = batch_norm_attr->scale;
  Array<IndexExpr> data_shape = args[0]->checked_type().as<TensorTypeNode>()->shape;
  int64_t size = GetCartesianProd(data_shape);

  int64_t axis_shape = data_shape[axis].as<IntImmNode>()->value;
  CalculationAmontIndicator cai;
  cai.exp = axis_shape;
  cai.add = cai.exp;
  cai.sub = size;
  cai.div = size;
  if (center && scale) {
    cai.fused_mul_add = size;
    cai.mul = size;
    cai.add += size;
  } else if (!center && scale) {  // ignore beta
    cai.fused_mul_add = size;
    cai.mul = size;
  } else if (center && !scale) {  // ignore gamma
    cai.add += size;
  }

  res = cai.GetIndicatorMap();

  SetLayerName(batch_norm_attr->layer_name);
  return res;
}

AiTraceDataFrame QnnBatchNormProfiler::GetMemory(const Call& call_node) {
  AiTraceDataFrame res;
  if (!call_node->checked_type_.defined()) {
    LOG(WARNING) << "The infer type pass should be called before the aitrace pass";
    return res;
  }
  Array<Expr> args = call_node->args;
  CHECK_EQ(args.size(), 5) << "The number of input arguments of a batch_norm node should be 5.";
  Array<IndexExpr> in_shape = args[0]->checked_type().as<TensorTypeNode>()->shape;
  Array<IndexExpr> gamma_shape = args[1]->checked_type().as<TensorTypeNode>()->shape;
  Array<IndexExpr> beta_shape = args[2]->checked_type().as<TensorTypeNode>()->shape;
  Array<IndexExpr> mean_shape = args[3]->checked_type().as<TensorTypeNode>()->shape;
  Array<IndexExpr> var_shape = args[4]->checked_type().as<TensorTypeNode>()->shape;

  MemoryIndicator mi;
  mi.params += GetCartesianProd(gamma_shape);
  mi.params += GetCartesianProd(beta_shape);
  mi.params += GetCartesianProd(mean_shape);
  mi.params += GetCartesianProd(var_shape);

  // output is same shape as input
  mi.output += GetCartesianProd(in_shape);

  res = mi.GetIndicatorMap();

  const auto* attrs = call_node->attrs.as<QnnCSIBatchNormAttrs>();
  SetLayerName(attrs->layer_name);
  return res;
}

RELAY_REGISTER_OP("qnn.csi.bn")
    .set_attr<FCalAmount>("FCalAmount", QnnBatchNormProfiler::GetCalculationAmount);
RELAY_REGISTER_OP("qnn.csi.bn").set_attr<FMemory>("FMemory", QnnBatchNormProfiler::GetMemory);
RELAY_REGISTER_OP("qnn.csi.bn").set_attr<FOpName>("FOpName", QnnBatchNormProfiler::GetLayerName);

//------------------------------------------------------------------------------
// QNN BiasAdd profiler implementation
//------------------------------------------------------------------------------

AiTraceDataFrame QnnBiasAddProfiler::GetCalculationAmount(const Call& call_node) {
  const auto* attrs = call_node->attrs.as<QnnBinaryOpAttrs>();
  SetLayerName(attrs->layer_name);
  return GetEltwiseCalAmountCommon(call_node, "add");
}

AiTraceDataFrame QnnBiasAddProfiler::GetMemory(const Call& call_node) {
  AiTraceDataFrame res;
  if (!call_node->checked_type_.defined()) {
    LOG(WARNING) << "The infer type pass should be called before the aitrace pass";
    return res;
  }
  Array<Expr> args = call_node->args;
  Array<IndexExpr> bias_shape = args[1]->checked_type().as<TensorTypeNode>()->shape;
  Array<IndexExpr> output_shape = call_node->checked_type().as<TensorTypeNode>()->shape;

  MemoryIndicator mi;
  mi.params += GetCartesianProd(bias_shape);
  mi.output += GetCartesianProd(output_shape);

  res = mi.GetIndicatorMap();

  const auto* attrs = call_node->attrs.as<QnnBinaryOpAttrs>();
  SetLayerName(attrs->layer_name);
  return res;
}

RELAY_REGISTER_OP("qnn.csi.bias_add")
    .set_attr<FCalAmount>("FCalAmount", QnnBiasAddProfiler::GetCalculationAmount);
RELAY_REGISTER_OP("qnn.csi.bias_add").set_attr<FMemory>("FMemory", QnnBiasAddProfiler::GetMemory);
RELAY_REGISTER_OP("qnn.csi.bias_add")
    .set_attr<FOpName>("FOpName", QnnBiasAddProfiler::GetLayerName);

//------------------------------------------------------------------------------
// QNN Concatenate profiler implementation
//------------------------------------------------------------------------------

AiTraceDataFrame QnnConcatenateProfiler::GetCalculationAmount(const Call& call_node) {
  const auto* attrs = call_node->attrs.as<QnnConcatenateAttrs>();
  SetLayerName(attrs->layer_name);
  return GetZeroCalAmountCommon(call_node);
}

AiTraceDataFrame QnnConcatenateProfiler::GetMemory(const Call& call_node) {
  const auto* attrs = call_node->attrs.as<QnnConcatenateAttrs>();
  SetLayerName(attrs->layer_name);
  return GetMemoryCommon(call_node);
}

RELAY_REGISTER_OP("qnn.csi.concatenate")
    .set_attr<FCalAmount>("FCalAmount", QnnConcatenateProfiler::GetCalculationAmount);
RELAY_REGISTER_OP("qnn.csi.concatenate")
    .set_attr<FMemory>("FMemory", QnnConcatenateProfiler::GetMemory);
RELAY_REGISTER_OP("qnn.csi.concatenate")
    .set_attr<FOpName>("FOpName", QnnConcatenateProfiler::GetLayerName);

//------------------------------------------------------------------------------
// Qnn Conv2d profiler implementation
//------------------------------------------------------------------------------

// calculate formula:
// macc = Cin * (Hk * Wk) * Hout * Wout * Cout * batch / group
// flops = (2Cin * (Hk * Wk) -1) *Hout * Wout * Cout * batch / group
AiTraceDataFrame QnnConv2dProfiler::GetCalculationAmount(const Call& call_node) {
  AiTraceDataFrame res;

  if (!call_node->checked_type_.defined()) {
    LOG(WARNING) << "The infer type pass should be called before the aitrace pass";
    return res;
  }
  Array<Expr> args = call_node->args;
  CHECK_EQ(args.size(), 3) << "The number of input arguments of a QNN CONV 2D node should be 3.";
  const auto* conv_2d_attr = call_node->attrs.as<QnnCSIConv2DAttrs>();
  const auto* data_type = args[0]->checked_type().as<TensorTypeNode>();
  Array<IndexExpr> data_shape = data_type->shape;
  Array<IndexExpr> bias_shape = args[2]->checked_type().as<TensorTypeNode>()->shape;
  std::string data_layout = conv_2d_attr->data_layout;
  int32_t C_ind = Layout(data_layout).IndexOf(LayoutAxis::Get('C'));
  int32_t c_ind = Layout(data_layout).IndexOf(LayoutAxis::Get('c'));
  CHECK_NE(C_ind, -1) << "There is no input channel dimension.";
  int64_t input_channel = static_cast<int64_t>(data_shape[C_ind].as<IntImmNode>()->value);
  if (c_ind != -1) input_channel *= static_cast<int64_t>(data_shape[c_ind].as<IntImmNode>()->value);
  Array<IndexExpr> kernel_size = conv_2d_attr->kernel_size;
  CHECK_EQ(kernel_size.size(), 2) << "The dimension of the kernel in Conv 2D should be 2.";
  const auto* expr = call_node->checked_type().as<TensorTypeNode>();
  Array<IndexExpr> output_tensor = expr->shape;
  CHECK(output_tensor.size() == 4 || output_tensor.size() == 5)
      << "The dimension of the output tensor in Conv 2D should be 4 or 5.";
  CHECK_EQ(input_channel % conv_2d_attr->groups, 0)
      << "The number of input channels is not divisble by groups.";

  CalculationAmontIndicator cai;
  cai.fused_mul_add = GetCartesianProd(output_tensor) * GetCartesianProd(kernel_size);
  cai.fused_mul_add *= input_channel / conv_2d_attr->groups;
  cai.mul = cai.fused_mul_add;
  cai.add = (GetCartesianProd(kernel_size) * input_channel / conv_2d_attr->groups - 1) *
                GetCartesianProd(output_tensor) +
            GetCartesianProd(bias_shape);

  res = cai.GetIndicatorMap();

  SetLayerName(conv_2d_attr->layer_name);
  return res;
}

AiTraceDataFrame QnnConv2dProfiler::GetMemory(const Call& call_node) {
  AiTraceDataFrame res;
  if (!call_node->checked_type_.defined()) {
    LOG(WARNING) << "The infer type pass should be called before the aitrace pass";
    return res;
  }
  Array<Expr> args = call_node->args;
  CHECK_EQ(args.size(), 3) << "The number of input arguments of a QNN CONV 2D node should be 3.";
  const auto* conv_2d_attr = call_node->attrs.as<QnnCSIConv2DAttrs>();
  const auto* data_type = args[0]->checked_type().as<TensorTypeNode>();
  Array<IndexExpr> data_shape = data_type->shape;
  std::string data_layout = conv_2d_attr->data_layout;
  int32_t C_ind = Layout(data_layout).IndexOf(LayoutAxis::Get('C'));
  int32_t c_ind = Layout(data_layout).IndexOf(LayoutAxis::Get('c'));
  CHECK_NE(C_ind, -1) << "There is no input channel dimension.";
  int64_t input_channel = static_cast<int64_t>(data_shape[C_ind].as<IntImmNode>()->value);
  if (c_ind != -1) input_channel *= static_cast<int64_t>(data_shape[c_ind].as<IntImmNode>()->value);
  Array<IndexExpr> kernel_size = conv_2d_attr->kernel_size;
  Array<IndexExpr> bias_size = args[2]->checked_type().as<TensorTypeNode>()->shape;
  CHECK_EQ(kernel_size.size(), 2) << "The dimension of the kernel in Conv 2D should be 2.";
  const auto* expr = call_node->checked_type().as<TensorTypeNode>();
  Array<IndexExpr> output_tensor = expr->shape;
  CHECK(output_tensor.size() == 4 || output_tensor.size() == 5)
      << "The dimension of the output tensor in Conv 2D should be 4 or 5.";
  CHECK_EQ(input_channel % conv_2d_attr->groups, 0)
      << "The number of input channels is not divisble by groups.";
  int64_t output_channel = static_cast<int64_t>(output_tensor[C_ind].as<IntImmNode>()->value);
  if (c_ind != -1)
    output_channel *= static_cast<int64_t>(output_tensor[c_ind].as<IntImmNode>()->value);

  MemoryIndicator mi;
  mi.params +=
      GetCartesianProd(kernel_size) * input_channel * output_channel / conv_2d_attr->groups +
      GetCartesianProd(bias_size);
  mi.output += GetCartesianProd(output_tensor);

  res = mi.GetIndicatorMap();

  SetLayerName(conv_2d_attr->layer_name);
  return res;
}

RELAY_REGISTER_OP("qnn.csi.conv2d")
    .set_attr<FCalAmount>("FCalAmount", QnnConv2dProfiler::GetCalculationAmount);
RELAY_REGISTER_OP("qnn.csi.conv2d").set_attr<FMemory>("FMemory", QnnConv2dProfiler::GetMemory);
RELAY_REGISTER_OP("qnn.csi.conv2d").set_attr<FOpName>("FOpName", QnnConv2dProfiler::GetLayerName);

//------------------------------------------------------------------------------
// QNN Conv2dTranspose profiler implementation
//------------------------------------------------------------------------------

// calculate formula:
// macc = Cin * (Hk * Wk) * Hout * Wout * Cout * batch / group
// flops = (2Cin * (Hk * Wk) -1) *Hout * Wout * Cout * batch / group
AiTraceDataFrame QnnConv2dTranposeProfiler::GetCalculationAmount(const Call& call_node) {
  AiTraceDataFrame res;

  if (!call_node->checked_type_.defined()) {
    LOG(WARNING) << "The infer type pass should be called before the aitrace pass";
    return res;
  }
  Array<Expr> args = call_node->args;
  CHECK_EQ(args.size(), 3)
      << "The number of input arguments of a QNN CONV 2D Transpose node should be 3.";
  const auto* conv_2d_attr = call_node->attrs.as<QnnCSIDeConv2DAttrs>();
  const auto* data_type = args[0]->checked_type().as<TensorTypeNode>();
  Array<IndexExpr> data_shape = data_type->shape;
  std::string data_layout = conv_2d_attr->data_layout;
  int32_t C_ind = Layout(data_layout).IndexOf(LayoutAxis::Get('C'));
  int32_t c_ind = Layout(data_layout).IndexOf(LayoutAxis::Get('c'));
  CHECK_NE(C_ind, -1) << "There is no input channel dimension.";
  int64_t input_channel = static_cast<int64_t>(data_shape[C_ind].as<IntImmNode>()->value);
  if (c_ind != -1) input_channel *= static_cast<int64_t>(data_shape[c_ind].as<IntImmNode>()->value);
  Array<IndexExpr> kernel_size = conv_2d_attr->kernel_size;
  Array<IndexExpr> bias_size = args[1]->checked_type().as<TensorTypeNode>()->shape;
  CHECK_EQ(kernel_size.size(), 2) << "The dimension of the kernel in Conv 2D should be 2.";
  const auto* expr = call_node->checked_type().as<TensorTypeNode>();
  Array<IndexExpr> output_tensor = expr->shape;
  CHECK(output_tensor.size() == 4 || output_tensor.size() == 5)
      << "The dimension of the output tensor in Conv 2D should be 4 or 5.";
  CHECK_EQ(input_channel % conv_2d_attr->groups, 0)
      << "The number of input channels is not divisble by groups.";

  CalculationAmontIndicator cai;
  cai.fused_mul_add = GetCartesianProd(output_tensor) * GetCartesianProd(kernel_size);
  cai.fused_mul_add *= input_channel / conv_2d_attr->groups;
  cai.mul = cai.fused_mul_add;
  cai.add = (GetCartesianProd(kernel_size) * input_channel / conv_2d_attr->groups - 1) *
                GetCartesianProd(output_tensor) +
            GetCartesianProd(bias_size);

  res = cai.GetIndicatorMap();

  SetLayerName(conv_2d_attr->layer_name);
  return res;
}

AiTraceDataFrame QnnConv2dTranposeProfiler::GetMemory(const Call& call_node) {
  AiTraceDataFrame res;

  if (!call_node->checked_type_.defined()) {
    LOG(WARNING) << "The infer type pass should be called before the aitrace pass";
    return res;
  }
  Array<Expr> args = call_node->args;
  CHECK_EQ(args.size(), 3)
      << "The number of input arguments of a QNN CONV 2D Transpose node should be 3.";
  const auto* conv_2d_attr = call_node->attrs.as<QnnCSIDeConv2DAttrs>();
  const auto* data_type = args[0]->checked_type().as<TensorTypeNode>();
  Array<IndexExpr> data_shape = data_type->shape;
  std::string data_layout = conv_2d_attr->data_layout;
  int32_t C_ind = Layout(data_layout).IndexOf(LayoutAxis::Get('C'));
  int32_t c_ind = Layout(data_layout).IndexOf(LayoutAxis::Get('c'));
  CHECK_NE(C_ind, -1) << "There is no input channel dimension.";
  int64_t input_channel = static_cast<int64_t>(data_shape[C_ind].as<IntImmNode>()->value);
  if (c_ind != -1) input_channel *= static_cast<int64_t>(data_shape[c_ind].as<IntImmNode>()->value);
  Array<IndexExpr> kernel_size = conv_2d_attr->kernel_size;
  Array<IndexExpr> bias_size = args[0]->checked_type().as<TensorTypeNode>()->shape;
  CHECK_EQ(kernel_size.size(), 2) << "The dimension of the kernel in Conv 2D should be 2.";
  const auto* expr = call_node->checked_type().as<TensorTypeNode>();
  Array<IndexExpr> output_tensor = expr->shape;
  CHECK(output_tensor.size() == 4 || output_tensor.size() == 5)
      << "The dimension of the output tensor in Conv 2D should be 4 or 5.";
  CHECK_EQ(input_channel % conv_2d_attr->groups, 0)
      << "The number of input channels is not divisble by groups.";
  int64_t output_channel = static_cast<int64_t>(output_tensor[C_ind].as<IntImmNode>()->value);
  if (c_ind != -1)
    output_channel *= static_cast<int64_t>(output_tensor[c_ind].as<IntImmNode>()->value);

  MemoryIndicator mi;
  mi.params +=
      GetCartesianProd(kernel_size) * input_channel * output_channel / conv_2d_attr->groups +
      GetCartesianProd(bias_size);
  mi.output += GetCartesianProd(output_tensor);

  res = mi.GetIndicatorMap();

  SetLayerName(conv_2d_attr->layer_name);
  return res;
}

RELAY_REGISTER_OP("qnn.csi.deconv2d")
    .set_attr<FCalAmount>("FCalAmount", QnnConv2dTranposeProfiler::GetCalculationAmount);
RELAY_REGISTER_OP("qnn.csi.deconv2d")
    .set_attr<FMemory>("FMemory", QnnConv2dTranposeProfiler::GetMemory);
RELAY_REGISTER_OP("qnn.csi.deconv2d")
    .set_attr<FOpName>("FOpName", QnnConv2dTranposeProfiler::GetLayerName);

//------------------------------------------------------------------------------
// QNN Dense profiler implementation
//------------------------------------------------------------------------------

// calculate formula:
//                      out        =             X           *        W^T          +         bias
// shape: (d1,d2,...,dn, unit_out) = (d1,d2,...,dn, unit_in) * (unit_in, unit_out) + (d1,d2,...,dn,
// unit_out) macc = prod(d1, d2, ..., dn) * unit_in * unit_out flops = prod(d1, d2, ..., dn) *
// (2unit_in-1) * unit_out
AiTraceDataFrame QnnDenseProfiler::GetCalculationAmount(const Call& call_node) {
  AiTraceDataFrame res;
  if (!call_node->checked_type_.defined()) {
    LOG(WARNING) << "The infer type pass should be called before the aitrace pass";
    return res;
  }
  Array<Expr> args = call_node->args;
  CHECK_EQ(args.size(), 3) << "The number of input arguments of a QNN Dense node should be 3.";
  const auto* data_type = args[0]->checked_type().as<TensorTypeNode>();
  const auto* weight_type = args[1]->checked_type().as<TensorTypeNode>();
  Array<IndexExpr> data_shape = data_type->shape;
  Array<IndexExpr> weight_shape = weight_type->shape;
  int64_t unit_in = static_cast<int64_t>(data_shape[data_shape.size() - 1].as<IntImmNode>()->value);
  CHECK(weight_shape.size() == 2) << "The dimension of an weight tensor to Dense node should be 2.";
  int64_t unit_out = static_cast<int64_t>(weight_shape[0].as<IntImmNode>()->value);
  int64_t unit_in_w = static_cast<int64_t>(weight_shape[1].as<IntImmNode>()->value);
  CHECK_EQ(unit_in, unit_in_w) << "The dimensions of input arguments do not match.";

  CalculationAmontIndicator cai;
  int64_t d_prod = GetCartesianProd(data_shape) / unit_in;
  cai.fused_mul_add = d_prod * unit_in * unit_out;
  cai.mul = d_prod * unit_in * unit_out;
  cai.add = d_prod * (unit_in - 1) * unit_out + d_prod * unit_out;
  res = cai.GetIndicatorMap();

  const auto* attrs = call_node->attrs.as<QnnCSIDenseAttrs>();
  SetLayerName(attrs->layer_name);
  return res;
}

AiTraceDataFrame QnnDenseProfiler::GetMemory(const Call& call_node) {
  AiTraceDataFrame res;
  if (!call_node->checked_type_.defined()) {
    LOG(WARNING) << "The infer type pass should be called before the aitrace pass";
    return res;
  }
  Array<Expr> args = call_node->args;
  CHECK_EQ(args.size(), 3) << "The number of input arguments of a QNN Dense node should be 3.";
  Array<IndexExpr> weight_shape = args[1]->checked_type().as<TensorTypeNode>()->shape;
  Array<IndexExpr> bias_shape = args[2]->checked_type().as<TensorTypeNode>()->shape;
  Array<IndexExpr> output_shape = call_node->checked_type().as<TensorTypeNode>()->shape;

  MemoryIndicator mi;
  mi.params = GetCartesianProd(weight_shape) + GetCartesianProd(bias_shape);
  mi.output = GetCartesianProd(output_shape);

  res = mi.GetIndicatorMap();

  const auto* attrs = call_node->attrs.as<QnnCSIDenseAttrs>();
  SetLayerName(attrs->layer_name);
  return res;
}

RELAY_REGISTER_OP("qnn.csi.dense")
    .set_attr<FCalAmount>("FCalAmount", QnnDenseProfiler::GetCalculationAmount);
RELAY_REGISTER_OP("qnn.csi.dense").set_attr<FMemory>("FMemory", QnnDenseProfiler::GetMemory);
RELAY_REGISTER_OP("qnn.csi.dense").set_attr<FOpName>("FOpName", QnnDenseProfiler::GetLayerName);

//------------------------------------------------------------------------------
// QNN GlobalAvgpool2d profiler implementation
//------------------------------------------------------------------------------

AiTraceDataFrame QnnGlobalAvgPool2dProfiler::GetCalculationAmount(const Call& call_node) {
  const auto* globalpool2d_attr = call_node->attrs.as<QnnCSIGlobalAvgPoolAttrs>();
  SetLayerName(globalpool2d_attr->layer_name);
  return GetPoolCalAmountCommon(call_node, globalpool2d_attr, Array<IndexExpr>({1, 1}), "avg",
                                true);
}

AiTraceDataFrame QnnGlobalAvgPool2dProfiler::GetMemory(const Call& call_node) {
  const auto* attrs = call_node->attrs.as<QnnCSIGlobalAvgPoolAttrs>();
  SetLayerName(attrs->layer_name);
  return GetMemoryCommon(call_node);
}

RELAY_REGISTER_OP("qnn.csi.global_avgpool2d")
    .set_attr<FCalAmount>("FCalAmount", QnnGlobalAvgPool2dProfiler::GetCalculationAmount);
RELAY_REGISTER_OP("qnn.csi.global_avgpool2d")
    .set_attr<FMemory>("FMemory", QnnGlobalAvgPool2dProfiler::GetMemory);
RELAY_REGISTER_OP("qnn.csi.global_avgpool2d")
    .set_attr<FOpName>("FOpName", QnnGlobalAvgPool2dProfiler::GetLayerName);

//------------------------------------------------------------------------------
// QNN GlobalMaxpool2d profiler implementation
//------------------------------------------------------------------------------

AiTraceDataFrame QnnGlobalMaxPool2dProfiler::GetCalculationAmount(const Call& call_node) {
  const auto* globalpool2d_attr = call_node->attrs.as<QnnCSIGlobalMaxPoolAttrs>();
  SetLayerName(globalpool2d_attr->layer_name);
  return GetPoolCalAmountCommon(call_node, globalpool2d_attr, Array<IndexExpr>({1, 1}), "max",
                                true);
}

AiTraceDataFrame QnnGlobalMaxPool2dProfiler::GetMemory(const Call& call_node) {
  const auto* attrs = call_node->attrs.as<QnnCSIGlobalMaxPoolAttrs>();
  SetLayerName(attrs->layer_name);
  return GetMemoryCommon(call_node);
}

RELAY_REGISTER_OP("qnn.csi.global_maxpool2d")
    .set_attr<FCalAmount>("FCalAmount", QnnGlobalMaxPool2dProfiler::GetCalculationAmount);
RELAY_REGISTER_OP("qnn.csi.global_maxpool2d")
    .set_attr<FMemory>("FMemory", QnnGlobalMaxPool2dProfiler::GetMemory);
RELAY_REGISTER_OP("qnn.csi.global_maxpool2d")
    .set_attr<FOpName>("FOpName", QnnGlobalMaxPool2dProfiler::GetLayerName);

//------------------------------------------------------------------------------
// QNN LRN profiler implementation
//------------------------------------------------------------------------------

// output = in_data / (bias + (alpha/size)sum(in_data^2))^beta
AiTraceDataFrame QnnLRNProfiler::GetCalculationAmount(const Call& call_node) {
  AiTraceDataFrame res;
  if (!call_node->checked_type_.defined()) {
    LOG(WARNING) << "The infer type pass should be called before the aitrace pass";
    return res;
  }
  Array<Expr> args = call_node->args;
  CHECK_EQ(args.size(), 1) << "The number of input arguments of a LRN node should be 1.";

  Array<IndexExpr> in_shape = args[0]->checked_type().as<TensorTypeNode>()->shape;
  const auto* lrn_attrs = call_node->attrs.as<QnnCSILRNAttrs>();
  int size = lrn_attrs->size;

  int64_t num_inputs = GetCartesianProd(in_shape);
  CalculationAmontIndicator cai;
  cai.fused_mul_add = num_inputs * (size + 1) + num_inputs;  // sum(in_data^2) and (alpha/size)*...
  cai.mul = cai.fused_mul_add;                               // in_data^2
  cai.add = num_inputs * size + num_inputs;                  // (bias + ...) and sum(...)
  cai.exp = num_inputs;                                      // (...)^beta;
  cai.div = num_inputs;                                      // in_data/(...)
  res = cai.GetIndicatorMap();

  SetLayerName(lrn_attrs->layer_name);
  return res;
}

AiTraceDataFrame QnnLRNProfiler::GetMemory(const Call& call_node) {
  const auto* attrs = call_node->attrs.as<QnnCSILRNAttrs>();
  SetLayerName(attrs->layer_name);
  return GetMemoryCommon(call_node);
}

RELAY_REGISTER_OP("qnn.csi.lrn")
    .set_attr<FCalAmount>("FCalAmount", QnnLRNProfiler::GetCalculationAmount);
RELAY_REGISTER_OP("qnn.csi.lrn").set_attr<FMemory>("FMemory", QnnLRNProfiler::GetMemory);
RELAY_REGISTER_OP("qnn.csi.lrn").set_attr<FOpName>("FOpName", QnnLRNProfiler::GetLayerName);

//------------------------------------------------------------------------------
// QNN Maximum profiler implementation
//------------------------------------------------------------------------------

AiTraceDataFrame QnnMaximumProfiler::GetCalculationAmount(const Call& call_node) {
  const auto* attrs = call_node->attrs.as<QnnBinaryOpAttrs>();
  SetLayerName(attrs->layer_name);
  return GetEltwiseCalAmountCommon(call_node, "max");
}

AiTraceDataFrame QnnMaximumProfiler::GetMemory(const Call& call_node) {
  const auto* attrs = call_node->attrs.as<QnnBinaryOpAttrs>();
  SetLayerName(attrs->layer_name);
  return GetMemoryCommon(call_node);
}

RELAY_REGISTER_OP("qnn.csi.maximum")
    .set_attr<FCalAmount>("FCalAmount", QnnMaximumProfiler::GetCalculationAmount);
RELAY_REGISTER_OP("qnn.csi.maximum").set_attr<FMemory>("FMemory", QnnMaximumProfiler::GetMemory);
RELAY_REGISTER_OP("qnn.csi.maximum").set_attr<FOpName>("FOpName", QnnMaximumProfiler::GetLayerName);

//------------------------------------------------------------------------------
// QNN Maxpool2d profiler implementation
//------------------------------------------------------------------------------

AiTraceDataFrame QnnMaxPool2dProfiler::GetCalculationAmount(const Call& call_node) {
  const auto* maxpool2d_attr = call_node->attrs.as<QnnCSIMaxPool2DAttrs>();
  SetLayerName(maxpool2d_attr->layer_name);
  return GetPoolCalAmountCommon(call_node, maxpool2d_attr, maxpool2d_attr->pool_size, "max", false);
}

AiTraceDataFrame QnnMaxPool2dProfiler::GetMemory(const Call& call_node) {
  const auto* attrs = call_node->attrs.as<QnnCSIMaxPool2DAttrs>();
  SetLayerName(attrs->layer_name);
  return GetMemoryCommon(call_node);
}

RELAY_REGISTER_OP("qnn.csi.maxpool2d")
    .set_attr<FCalAmount>("FCalAmount", QnnMaxPool2dProfiler::GetCalculationAmount);
RELAY_REGISTER_OP("qnn.csi.maxpool2d")
    .set_attr<FMemory>("FMemory", QnnMaxPool2dProfiler::GetMemory);
RELAY_REGISTER_OP("qnn.csi.maxpool2d")
    .set_attr<FOpName>("FOpName", QnnMaxPool2dProfiler::GetLayerName);

//------------------------------------------------------------------------------
// QNN Maxpool2dLocation profiler implementation
//------------------------------------------------------------------------------

AiTraceDataFrame QnnMaxPool2dLocationProfiler::GetCalculationAmount(const Call& call_node) {
  const auto* maxpool2d_attr = call_node->attrs.as<QnnCSIMaxPool2DLocatAttrs>();
  SetLayerName(maxpool2d_attr->layer_name);
  return GetPoolCalAmountCommon(call_node, maxpool2d_attr, maxpool2d_attr->pool_size, "max", false);
}

AiTraceDataFrame QnnMaxPool2dLocationProfiler::GetMemory(const Call& call_node) {
  const auto* attrs = call_node->attrs.as<QnnCSIMaxPool2DLocatAttrs>();
  SetLayerName(attrs->layer_name);
  return GetMemoryCommon(call_node);
}

RELAY_REGISTER_OP("qnn.csi.maxpool2d_locat")
    .set_attr<FCalAmount>("FCalAmount", QnnMaxPool2dLocationProfiler::GetCalculationAmount);
RELAY_REGISTER_OP("qnn.csi.maxpool2d_locat")
    .set_attr<FMemory>("FMemory", QnnMaxPool2dLocationProfiler::GetMemory);
RELAY_REGISTER_OP("qnn.csi.maxpool2d_locat")
    .set_attr<FOpName>("FOpName", QnnMaxPool2dLocationProfiler::GetLayerName);

//------------------------------------------------------------------------------
// QNN Maxpool2dWithArgmax profiler implementation
//------------------------------------------------------------------------------

AiTraceDataFrame QnnMaxPool2dWithArgmaxProfiler::GetCalculationAmount(const Call& call_node) {
  const auto* maxpool2d_attr = call_node->attrs.as<QnnCSIMaxPool2DAttrs>();
  SetLayerName(maxpool2d_attr->layer_name);
  return GetPoolCalAmountCommon(call_node, maxpool2d_attr, maxpool2d_attr->pool_size, "max", false);
}

AiTraceDataFrame QnnMaxPool2dWithArgmaxProfiler::GetMemory(const Call& call_node) {
  const auto* attrs = call_node->attrs.as<QnnCSIMaxPool2DAttrs>();
  SetLayerName(attrs->layer_name);
  return GetMemoryCommon(call_node);
}

RELAY_REGISTER_OP("qnn.csi.maxpool2d_with_argmax")
    .set_attr<FCalAmount>("FCalAmount", QnnMaxPool2dWithArgmaxProfiler::GetCalculationAmount);
RELAY_REGISTER_OP("qnn.csi.maxpool2d_with_argmax")
    .set_attr<FMemory>("FMemory", QnnMaxPool2dWithArgmaxProfiler::GetMemory);
RELAY_REGISTER_OP("qnn.csi.maxpool2d_with_argmax")
    .set_attr<FOpName>("FOpName", QnnMaxPool2dWithArgmaxProfiler::GetLayerName);

//------------------------------------------------------------------------------
// QNN Multiply profiler implementation
//------------------------------------------------------------------------------

AiTraceDataFrame QnnMultiplyProfiler::GetCalculationAmount(const Call& call_node) {
  const auto* attrs = call_node->attrs.as<QnnBinaryOpAttrs>();
  SetLayerName(attrs->layer_name);
  return GetEltwiseCalAmountCommon(call_node, "mul");
}

AiTraceDataFrame QnnMultiplyProfiler::GetMemory(const Call& call_node) {
  const auto* attrs = call_node->attrs.as<QnnBinaryOpAttrs>();
  SetLayerName(attrs->layer_name);
  return GetMemoryCommon(call_node);
}

RELAY_REGISTER_OP("qnn.csi.mul")
    .set_attr<FCalAmount>("FCalAmount", QnnMultiplyProfiler::GetCalculationAmount);
RELAY_REGISTER_OP("qnn.csi.mul").set_attr<FMemory>("FMemory", QnnMultiplyProfiler::GetMemory);
RELAY_REGISTER_OP("qnn.csi.mul").set_attr<FOpName>("FOpName", QnnMultiplyProfiler::GetLayerName);

//------------------------------------------------------------------------------
// QNN PRelu profiler implementation
//------------------------------------------------------------------------------

AiTraceDataFrame QnnPreluProfiler::GetCalculationAmount(const Call& call_node) {
  const auto* attrs = call_node->attrs.as<QnnCSIPReluAttrs>();
  SetLayerName(attrs->layer_name);
  return GetReluCalAmountCommon(call_node);
}

AiTraceDataFrame QnnPreluProfiler::GetMemory(const Call& call_node) {
  AiTraceDataFrame res;
  if (!call_node->checked_type_.defined()) {
    LOG(WARNING) << "The infer type pass should be called before the aitrace pass";
    return res;
  }
  Array<Expr> args = call_node->args;
  Array<IndexExpr> alpha_shape = args[1]->checked_type().as<TensorTypeNode>()->shape;
  Array<IndexExpr> output_shape = call_node->checked_type().as<TensorTypeNode>()->shape;

  MemoryIndicator mi;
  mi.params += GetCartesianProd(alpha_shape);
  mi.output += GetCartesianProd(output_shape);

  res = mi.GetIndicatorMap();

  const auto* attrs = call_node->attrs.as<QnnCSIPReluAttrs>();
  SetLayerName(attrs->layer_name);
  return res;
}

RELAY_REGISTER_OP("qnn.csi.prelu")
    .set_attr<FCalAmount>("FCalAmount", QnnPreluProfiler::GetCalculationAmount);
RELAY_REGISTER_OP("qnn.csi.prelu").set_attr<FMemory>("FMemory", QnnPreluProfiler::GetMemory);
RELAY_REGISTER_OP("qnn.csi.prelu").set_attr<FOpName>("FOpName", QnnPreluProfiler::GetLayerName);

// ------------------------------------------------------------------------------
// QNN Proposal profiler implementation
// ------------------------------------------------------------------------------

// according to https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/rpn/proposal_layer.py
// proposal(cls_prob, bbox_pred, im_info, ...) -> output
// cls_prob: [N, 2*num_anchors, H, W]
// bbox_pred: [N, 4*num_anchors, H, W]
// output: [N*rpn_post_nms_top_n, 5]
// Steps:
//    1. generate proposals by shifted anchors
//        anchors: [H*W*num_anchors, 4], bbox_deltas: [H*W*num_anchors, 4]
//        generate shifted anchors
//        proposals = transform(anchors, bbox_deltas)
//    2. clip proposals into images size
//    3. remove proposals whose size is too small
//    4. sort proposals by score from highest to lowest
//    5. take rpn_pre_nms_top_n proposals
//    6. nms
//    7. take rpn_post_nms_top_n proposale
//    8. return output.
AiTraceDataFrame QnnProposalProfiler::GetCalculationAmount(const Call& call_node) {
  AiTraceDataFrame res;

  if (!call_node->checked_type_.defined()) {
    LOG(WARNING) << "The infer type pass should be called before the aitrace pass";
    return res;
  }
  Array<Expr> args = call_node->args;
  CHECK_EQ(args.size(), 3) << "The number of input arguments of a Proposal node should be 3.";
  const auto* proposal_attr = call_node->attrs.as<QnnCSIProposalAttrs>();
  int pre_nms_n = proposal_attr->rpn_pre_nms_top_n;
  Array<IndexExpr> cls_prob_shape = args[0]->checked_type().as<TensorTypeNode>()->shape;
  int64_t batch = static_cast<int64_t>(cls_prob_shape[0].as<IntImmNode>()->value);
  int64_t num_anchors = static_cast<int64_t>(cls_prob_shape[1].as<IntImmNode>()->value);
  num_anchors /= 2;
  int64_t H = static_cast<int64_t>(cls_prob_shape[2].as<IntImmNode>()->value);
  int64_t W = static_cast<int64_t>(cls_prob_shape[3].as<IntImmNode>()->value);
  int64_t all_num_anchors = H * W * num_anchors;

  CalculationAmontIndicator cai;
  // 1. generate proposals by shifted anchors
  cai.add += all_num_anchors * 4;  // generate shifted anchors
  cai.add += all_num_anchors * 2;
  cai.sub += all_num_anchors * 4;
  cai.mul += all_num_anchors * 10;
  cai.exp += all_num_anchors * 2;
  cai.fused_mul_add += cai.mul;

  // 2. clip proposals into images size
  cai.comp += (all_num_anchors * 4 * 2);

  // 3. remove proposals whose size is too small
  cai.sub += (all_num_anchors * 2);
  cai.comp += (all_num_anchors * 2);

  // 4. sort proposals by score from highest to lowest
  // FIXME(chenf): actual number of proposals during this step is smaller than all_num_anchors
  // O(N^2)
  cai.comp += (all_num_anchors * all_num_anchors);

  // 5. take rpn_pre_nms_top_n proposals
  // no ops

  // 6. nms
  // O(N^2)
  cai.sub += (pre_nms_n * 2);
  cai.mul += pre_nms_n;
  cai.fused_mul_add += pre_nms_n;
  cai.comp += (pre_nms_n * pre_nms_n);

  cai.comp += (pre_nms_n * pre_nms_n * 6);
  cai.sub += (pre_nms_n * pre_nms_n * 3);
  cai.div += (pre_nms_n * pre_nms_n);
  cai.add += (pre_nms_n * pre_nms_n);
  cai.comp += (pre_nms_n * pre_nms_n);

  // 7. take rpn_post_nms_top_n proposale
  // no ops
  cai.fused_mul_add *= batch;
  cai.mul *= batch;
  cai.div *= batch;
  cai.add *= batch;
  cai.sub *= batch;
  cai.exp *= batch;
  cai.comp *= batch;
  res = cai.GetIndicatorMap();

  SetLayerName(proposal_attr->layer_name);
  return res;
}

AiTraceDataFrame QnnProposalProfiler::GetMemory(const Call& call_node) {
  const auto* attrs = call_node->attrs.as<QnnCSIProposalAttrs>();
  SetLayerName(attrs->layer_name);
  return GetMemoryCommon(call_node);
}

RELAY_REGISTER_OP("qnn.csi.proposal")
    .set_attr<FCalAmount>("FCalAmount", QnnProposalProfiler::GetCalculationAmount);
RELAY_REGISTER_OP("qnn.csi.proposal").set_attr<FMemory>("FMemory", QnnProposalProfiler::GetMemory);
RELAY_REGISTER_OP("qnn.csi.proposal")
    .set_attr<FOpName>("FOpName", QnnProposalProfiler::GetLayerName);

//------------------------------------------------------------------------------
// QNN PSRoiPooling profiler implementation
//------------------------------------------------------------------------------

// according to: python/tvm/topi/vision/psroipooling.py
AiTraceDataFrame QnnPsroipoolingProfiler::GetCalculationAmount(const Call& call_node) {
  AiTraceDataFrame res;
  if (!call_node->checked_type_.defined()) {
    LOG(WARNING) << "The infer type pass should be called before the aitrace pass";
    return res;
  }
  Array<Expr> args = call_node->args;
  CHECK_EQ(args.size(), 2) << "The number of input arguments of a PSRoiPooling node should be 2.";

  Array<IndexExpr> data_shape = args[0]->checked_type().as<TensorTypeNode>()->shape;
  Array<IndexExpr> rois_shape = args[1]->checked_type().as<TensorTypeNode>()->shape;
  int num_rois = rois_shape[0].as<IntImmNode>()->value;
  int batch = data_shape[0].as<IntImmNode>()->value;
  // int channel = data_shape[1].as<IntImmNode>()->value;
  int feature_h = data_shape[2].as<IntImmNode>()->value;
  int feature_w = data_shape[3].as<IntImmNode>()->value;
  const auto* psroi_pool_attr = call_node->attrs.as<QnnCSIPSROIPoolingAttrs>();
  int group_size = psroi_pool_attr->group_size;
  int output_dim = psroi_pool_attr->output_dim;

  CalculationAmontIndicator cai;
  cai.mul += (num_rois * 4);
  cai.fused_mul_add += (num_rois * 4);

  // ignore block split
  // cai.sub += (num_rois * 2);
  // cai.div += (num_rois * 2);

  // ignore calculation amout of infering location.
  // cai.mul += (num_rois * output_dim * group_size * group_size * (4 + 2));
  // cai.add += (num_rois * output_dim * group_size * group_size * (4 + 2));
  // cai.fused_mul_add += (num_rois * output_dim * group_size * group_size * (4 + 2));

  // FIXME(chenf): add ops in pool should be computed accorrding to actual rois shapes
  cai.add += (num_rois * output_dim * group_size * group_size * (feature_w * feature_h - 1));

  cai.div += (num_rois * output_dim * group_size * group_size);

  cai.fused_mul_add *= batch;
  cai.mul *= batch;
  cai.div *= batch;
  cai.add *= batch;
  cai.sub *= batch;
  cai.exp *= batch;
  cai.comp *= batch;
  res = cai.GetIndicatorMap();

  SetLayerName(psroi_pool_attr->layer_name);
  return res;
}

AiTraceDataFrame QnnPsroipoolingProfiler::GetMemory(const Call& call_node) {
  const auto* attrs = call_node->attrs.as<QnnCSIPSROIPoolingAttrs>();
  SetLayerName(attrs->layer_name);
  return GetMemoryCommon(call_node);
}

RELAY_REGISTER_OP("qnn.csi.psroipooling")
    .set_attr<FCalAmount>("FCalAmount", QnnPsroipoolingProfiler::GetCalculationAmount);
RELAY_REGISTER_OP("qnn.csi.psroipooling")
    .set_attr<FMemory>("FMemory", QnnPsroipoolingProfiler::GetMemory);
RELAY_REGISTER_OP("qnn.csi.psroipooling")
    .set_attr<FOpName>("FOpName", QnnPsroipoolingProfiler::GetLayerName);

// ------------------------------------------------------------------------------
// QNN Relu profiler implementation
// ------------------------------------------------------------------------------

AiTraceDataFrame QnnReluProfiler::GetCalculationAmount(const Call& call_node) {
  const auto* attrs = call_node->attrs.as<QnnCSIUnaryAttrs>();
  SetLayerName(attrs->layer_name);
  return GetReluCalAmountCommon(call_node);
}

AiTraceDataFrame QnnReluProfiler::GetMemory(const Call& call_node) {
  const auto* attrs = call_node->attrs.as<QnnCSIUnaryAttrs>();
  SetLayerName(attrs->layer_name);
  return GetMemoryCommon(call_node);
}

RELAY_REGISTER_OP("qnn.csi.relu")
    .set_attr<FCalAmount>("FCalAmount", QnnReluProfiler::GetCalculationAmount);
RELAY_REGISTER_OP("qnn.csi.relu").set_attr<FMemory>("FMemory", QnnReluProfiler::GetMemory);
RELAY_REGISTER_OP("qnn.csi.relu").set_attr<FOpName>("FOpName", QnnReluProfiler::GetLayerName);

//------------------------------------------------------------------------------
// QNN Reshape profiler implementation
//------------------------------------------------------------------------------

AiTraceDataFrame QnnReshapeProfiler::GetCalculationAmount(const Call& call_node) {
  const auto* attrs = call_node->attrs.as<QnnCSIReshapeAttrs>();
  SetLayerName(attrs->layer_name);
  return GetZeroCalAmountCommon(call_node);
}

AiTraceDataFrame QnnReshapeProfiler::GetMemory(const Call& call_node) {
  const auto* attrs = call_node->attrs.as<QnnCSIReshapeAttrs>();
  SetLayerName(attrs->layer_name);
  return GetMemoryCommon(call_node);
}

RELAY_REGISTER_OP("qnn.csi.reshape")
    .set_attr<FCalAmount>("FCalAmount", ReshapeProfiler::GetCalculationAmount);
RELAY_REGISTER_OP("qnn.csi.reshape").set_attr<FMemory>("FMemory", QnnReshapeProfiler::GetMemory);
RELAY_REGISTER_OP("qnn.csi.reshape").set_attr<FOpName>("FOpName", QnnReshapeProfiler::GetLayerName);

//------------------------------------------------------------------------------
// QNN RoiPool profiler implementation
//------------------------------------------------------------------------------

// according to:
// https://github.com/rbgirshick/caffe-fast-rcnn/blob/0dcd397b29507b8314e252e850518c5695efbb83/src/caffe/layers/roi_pooling_layer.cpp
AiTraceDataFrame QnnRoiPoolProfiler::GetCalculationAmount(const Call& call_node) {
  AiTraceDataFrame res;
  if (!call_node->checked_type_.defined()) {
    LOG(WARNING) << "The infer type pass should be called before the aitrace pass";
    return res;
  }
  Array<Expr> args = call_node->args;
  CHECK_EQ(args.size(), 2) << "The number of input arguments of a RoiPool node should be 2.";

  Array<IndexExpr> data_shape = args[0]->checked_type().as<TensorTypeNode>()->shape;
  Array<IndexExpr> rois_shape = args[1]->checked_type().as<TensorTypeNode>()->shape;
  int num_rois = rois_shape[0].as<IntImmNode>()->value;
  int batch = data_shape[0].as<IntImmNode>()->value;
  // int channel = data_shape[1].as<IntImmNode>()->value;
  int feature_h = data_shape[2].as<IntImmNode>()->value;
  int feature_w = data_shape[3].as<IntImmNode>()->value;
  const auto* roi_pool_attr = call_node->attrs.as<QnnCSIROIPoolingAttrs>();
  Array<IndexExpr> pooled_size = roi_pool_attr->pooled_size;
  CHECK_EQ(pooled_size.size(), 2) << "The number of pooled_size should be 2.";
  int pooled_h = pooled_size[0].as<IntImmNode>()->value;
  int pooled_w = pooled_size[1].as<IntImmNode>()->value;

  CalculationAmontIndicator cai;
  cai.mul += (num_rois * 4);
  cai.fused_mul_add += (num_rois * 4);

  /* ignore block split. */
  // cai.sub += (num_rois * 2);
  // cai.div += (num_rois * 2);

  cai.comp +=
      num_rois * ((feature_h / pooled_h) * (feature_w / pooled_h) - 1) * pooled_h * pooled_w;

  cai.fused_mul_add *= batch;
  cai.mul *= batch;
  cai.div *= batch;
  cai.add *= batch;
  cai.sub *= batch;
  cai.exp *= batch;
  cai.comp *= batch;

  res = cai.GetIndicatorMap();

  SetLayerName(roi_pool_attr->layer_name);
  return res;
}

AiTraceDataFrame QnnRoiPoolProfiler::GetMemory(const Call& call_node) {
  const auto* attrs = call_node->attrs.as<QnnCSIROIPoolingAttrs>();
  SetLayerName(attrs->layer_name);
  return GetMemoryCommon(call_node);
}

RELAY_REGISTER_OP("qnn.csi.roipooling")
    .set_attr<FCalAmount>("FCalAmount", QnnRoiPoolProfiler::GetCalculationAmount);
RELAY_REGISTER_OP("qnn.csi.roipooling").set_attr<FMemory>("FMemory", QnnRoiPoolProfiler::GetMemory);
RELAY_REGISTER_OP("qnn.csi.roipooling")
    .set_attr<FOpName>("FOpName", QnnRoiPoolProfiler::GetLayerName);

//------------------------------------------------------------------------------
// QNN Sigmoid profiler implementation
//------------------------------------------------------------------------------

AiTraceDataFrame QnnSigmoidProfiler::GetCalculationAmount(const Call& call_node) {
  AiTraceDataFrame res;
  if (!call_node->checked_type_.defined()) {
    LOG(WARNING) << "The infer type pass should be called before the aitrace pass";
    return res;
  }
  Array<Expr> args = call_node->args;
  CHECK_EQ(args.size(), 1) << "The number of input arguments of a Sigmoid node should be 1.";

  Array<IndexExpr> in_shape = args[0]->checked_type().as<TensorTypeNode>()->shape;
  int64_t num_inputs = GetCartesianProd(in_shape);

  CalculationAmontIndicator cai;
  cai.exp = num_inputs;
  cai.add = num_inputs;
  cai.div = num_inputs;
  res = cai.GetIndicatorMap();

  const auto* attrs = call_node->attrs.as<QnnCSIUnaryAttrs>();
  SetLayerName(attrs->layer_name);
  return res;
}

AiTraceDataFrame QnnSigmoidProfiler::GetMemory(const Call& call_node) {
  const auto* attrs = call_node->attrs.as<QnnCSIUnaryAttrs>();
  SetLayerName(attrs->layer_name);
  return GetMemoryCommon(call_node);
}

RELAY_REGISTER_OP("qnn.csi.sigmoid")
    .set_attr<FCalAmount>("FCalAmount", QnnSigmoidProfiler::GetCalculationAmount);
RELAY_REGISTER_OP("qnn.csi.sigmoid").set_attr<FMemory>("FMemory", QnnSigmoidProfiler::GetMemory);
RELAY_REGISTER_OP("qnn.csi.sigmoid").set_attr<FOpName>("FOpName", QnnSigmoidProfiler::GetLayerName);

//------------------------------------------------------------------------------
// QNN Softmax profiler implementation
//------------------------------------------------------------------------------

AiTraceDataFrame QnnSoftmaxProfiler::GetCalculationAmount(const Call& call_node) {
  AiTraceDataFrame res;
  if (!call_node->checked_type_.defined()) {
    LOG(WARNING) << "The infer type pass should be called before the aitrace pass";
    return res;
  }
  Array<Expr> args = call_node->args;
  CHECK_EQ(args.size(), 1) << "The number of input arguments of a Softmax node should be 1.";

  Array<IndexExpr> in_shape = args[0]->checked_type().as<TensorTypeNode>()->shape;
  int64_t num_inputs = GetCartesianProd(in_shape);
  const auto* soft_attr = call_node->attrs.as<QnnCSIAxisAttrs>();
  int axis = soft_attr->axis;
  int64_t axis_shape = in_shape[axis].as<IntImmNode>()->value;
  CalculationAmontIndicator cai;
  cai.exp = num_inputs;
  cai.add = (num_inputs / axis_shape - 1) * axis_shape;
  cai.div = num_inputs;
  res = cai.GetIndicatorMap();

  SetLayerName(soft_attr->layer_name);
  return res;
}

AiTraceDataFrame QnnSoftmaxProfiler::GetMemory(const Call& call_node) {
  const auto* attrs = call_node->attrs.as<QnnCSIAxisAttrs>();
  SetLayerName(attrs->layer_name);
  return GetMemoryCommon(call_node);
}

RELAY_REGISTER_OP("qnn.csi.softmax")
    .set_attr<FCalAmount>("FCalAmount", QnnSoftmaxProfiler::GetCalculationAmount);
RELAY_REGISTER_OP("qnn.csi.softmax").set_attr<FMemory>("FMemory", QnnSoftmaxProfiler::GetMemory);
RELAY_REGISTER_OP("qnn.csi.softmax").set_attr<FOpName>("FOpName", QnnSoftmaxProfiler::GetLayerName);

//------------------------------------------------------------------------------
// QNN Split profiler implementation
//------------------------------------------------------------------------------

AiTraceDataFrame QnnSplitProfiler::GetCalculationAmount(const Call& call_node) {
  const auto* attrs = call_node->attrs.as<QnnCSISplitAttrs>();
  SetLayerName(attrs->layer_name);
  return GetZeroCalAmountCommon(call_node);
}

AiTraceDataFrame QnnSplitProfiler::GetMemory(const Call& call_node) {
  AiTraceDataFrame res;
  if (!call_node->checked_type_.defined()) {
    LOG(WARNING) << "The infer type pass should be called before the aitrace pass";
    return res;
  }
  Array<Expr> args = call_node->args;
  Array<IndexExpr> input_shape = args[0]->checked_type().as<TensorTypeNode>()->shape;

  MemoryIndicator mi;
  mi.output += GetCartesianProd(input_shape);

  res = mi.GetIndicatorMap();

  const auto* attrs = call_node->attrs.as<QnnCSISplitAttrs>();
  SetLayerName(attrs->layer_name);
  return res;
}

RELAY_REGISTER_OP("qnn.csi.split")
    .set_attr<FCalAmount>("FCalAmount", QnnSplitProfiler::GetCalculationAmount);
RELAY_REGISTER_OP("qnn.csi.split").set_attr<FMemory>("FMemory", QnnSplitProfiler::GetMemory);
RELAY_REGISTER_OP("qnn.csi.split").set_attr<FOpName>("FOpName", QnnSplitProfiler::GetLayerName);

//------------------------------------------------------------------------------
// QNN StridedSlice profiler implementation
//------------------------------------------------------------------------------

AiTraceDataFrame QnnStridedSliceProfiler::GetCalculationAmount(const Call& call_node) {
  const auto* attrs = call_node->attrs.as<QnnCSIStridedSliceAttrs>();
  SetLayerName(attrs->layer_name);
  return GetZeroCalAmountCommon(call_node);
}

AiTraceDataFrame QnnStridedSliceProfiler::GetMemory(const Call& call_node) {
  AiTraceDataFrame res;
  if (!call_node->checked_type_.defined()) {
    LOG(WARNING) << "The infer type pass should be called before the aitrace pass";
    return res;
  }
  Array<IndexExpr> output_shape = call_node->checked_type().as<TensorTypeNode>()->shape;

  MemoryIndicator mi;
  mi.output += GetCartesianProd(output_shape);

  res = mi.GetIndicatorMap();

  const auto* attrs = call_node->attrs.as<QnnCSIStridedSliceAttrs>();
  SetLayerName(attrs->layer_name);
  return res;
}

RELAY_REGISTER_OP("qnn.csi.strided_slice")
    .set_attr<FCalAmount>("FCalAmount", QnnStridedSliceProfiler::GetCalculationAmount);
RELAY_REGISTER_OP("qnn.csi.strided_slice")
    .set_attr<FMemory>("FMemory", QnnStridedSliceProfiler::GetMemory);
RELAY_REGISTER_OP("qnn.csi.strided_slice")
    .set_attr<FOpName>("FOpName", QnnStridedSliceProfiler::GetLayerName);

//------------------------------------------------------------------------------
// QNN Tanh profiler implementation
//------------------------------------------------------------------------------

AiTraceDataFrame QnnTanhProfiler::GetCalculationAmount(const Call& call_node) {
  AiTraceDataFrame res;
  if (!call_node->checked_type_.defined()) {
    LOG(WARNING) << "The infer type pass should be called before the aitrace pass";
    return res;
  }
  Array<Expr> args = call_node->args;
  CHECK_EQ(args.size(), 1) << "The number of input arguments of a tanh node should be 1.";

  Array<IndexExpr> in_shape = args[0]->checked_type().as<TensorTypeNode>()->shape;
  int64_t num_inputs = GetCartesianProd(in_shape);

  CalculationAmontIndicator cai;
  cai.exp = 2 * num_inputs;
  cai.add = num_inputs;
  cai.sub = 2 * num_inputs;
  cai.div = num_inputs;
  res = cai.GetIndicatorMap();

  const auto* attrs = call_node->attrs.as<QnnCSIUnaryAttrs>();
  SetLayerName(attrs->layer_name);
  return res;
}

AiTraceDataFrame QnnTanhProfiler::GetMemory(const Call& call_node) {
  const auto* attrs = call_node->attrs.as<QnnCSIUnaryAttrs>();
  SetLayerName(attrs->layer_name);
  return GetMemoryCommon(call_node);
}

RELAY_REGISTER_OP("qnn.csi.tanh")
    .set_attr<FCalAmount>("FCalAmount", QnnTanhProfiler::GetCalculationAmount);
RELAY_REGISTER_OP("qnn.csi.tanh").set_attr<FMemory>("FMemory", QnnTanhProfiler::GetMemory);
RELAY_REGISTER_OP("qnn.csi.tanh").set_attr<FOpName>("FOpName", QnnTanhProfiler::GetLayerName);

//------------------------------------------------------------------------------
// QNN Transpose profiler implementation
//------------------------------------------------------------------------------

AiTraceDataFrame QnnTransposeProfiler::GetCalculationAmount(const Call& call_node) {
  const auto* attrs = call_node->attrs.as<QnnCSITransposeAttrs>();
  SetLayerName(attrs->layer_name);
  return GetZeroCalAmountCommon(call_node);
}

AiTraceDataFrame QnnTransposeProfiler::GetMemory(const Call& call_node) {
  AiTraceDataFrame res;
  if (!call_node->checked_type_.defined()) {
    LOG(WARNING) << "The infer type pass should be called before the aitrace pass";
    return res;
  }
  Array<Expr> args = call_node->args;
  Array<IndexExpr> input_shape = args[0]->checked_type().as<TensorTypeNode>()->shape;
  Array<IndexExpr> output_shape = call_node->checked_type().as<TensorTypeNode>()->shape;

  MemoryIndicator mi;
  mi.params += input_shape.size();
  mi.output += GetCartesianProd(output_shape);

  res = mi.GetIndicatorMap();

  const auto* attrs = call_node->attrs.as<QnnCSITransposeAttrs>();
  SetLayerName(attrs->layer_name);
  return res;
}

RELAY_REGISTER_OP("qnn.csi.transpose")
    .set_attr<FCalAmount>("FCalAmount", QnnTransposeProfiler::GetCalculationAmount);
RELAY_REGISTER_OP("qnn.csi.transpose")
    .set_attr<FMemory>("FMemory", QnnTransposeProfiler::GetMemory);
RELAY_REGISTER_OP("qnn.csi.transpose")
    .set_attr<FOpName>("FOpName", QnnTransposeProfiler::GetLayerName);

//------------------------------------------------------------------------------
// QNN unpooling profiler implementation
//------------------------------------------------------------------------------

AiTraceDataFrame QnnUnpoolingProfiler::GetCalculationAmount(const Call& call_node) {
  const auto* attrs = call_node->attrs.as<QnnCSIUnPoolingAttrs>();
  SetLayerName(attrs->layer_name);
  return GetZeroCalAmountCommon(call_node);
}

AiTraceDataFrame QnnUnpoolingProfiler::GetMemory(const Call& call_node) {
  const auto* attrs = call_node->attrs.as<QnnCSIUnPoolingAttrs>();
  SetLayerName(attrs->layer_name);
  return GetMemoryCommon(call_node);
}

RELAY_REGISTER_OP("qnn.csi.unpooling")
    .set_attr<FCalAmount>("FCalAmount", QnnUnpoolingProfiler::GetCalculationAmount);
RELAY_REGISTER_OP("qnn.csi.unpooling")
    .set_attr<FMemory>("FMemory", QnnUnpoolingProfiler::GetMemory);
RELAY_REGISTER_OP("qnn.csi.unpooling")
    .set_attr<FOpName>("FOpName", QnnUnpoolingProfiler::GetLayerName);

//------------------------------------------------------------------------------
// QNN upsampling profiler implementation
//------------------------------------------------------------------------------

AiTraceDataFrame QnnUpsamplingProfiler::GetCalculationAmount(const Call& call_node) {
  AiTraceDataFrame res;
  if (!call_node->checked_type_.defined()) {
    LOG(WARNING) << "The infer type pass should be called before the aitrace pass";
    return res;
  }
  Array<Expr> args = call_node->args;
  CHECK_EQ(args.size(), 1) << "The number of input arguments of a upsampling node should be 1.";

  const auto* attr = call_node->attrs.as<QnnCSIUpSamplingAttrs>();
  std::string layout = attr->layout;
  CHECK_EQ(layout, "NCHW") << "The layout of input data should be NCHW.";
  std::string method = attr->method;

  Array<IndexExpr> output_shape = call_node->checked_type().as<TensorTypeNode>()->shape;
  int64_t o_batch = output_shape[0].as<IntImmNode>()->value;
  int64_t o_channel = output_shape[1].as<IntImmNode>()->value;
  int64_t o_height = output_shape[2].as<IntImmNode>()->value;
  int64_t o_width = output_shape[3].as<IntImmNode>()->value;

  CalculationAmontIndicator cai;
  if (method == "bilinear") {
    // reference to https://en.wikipedia.org/wiki/Bilinear_interpolation
    cai.mul += o_height * o_width * o_channel * 6;
    cai.fused_mul_add += o_height * o_width * o_channel * 6;
    cai.sub += o_height * o_width * o_channel * 2;
    cai.add += o_height * o_width * o_channel * 3;
  } else if (method == "nearest_neighbor") {
    // zero ops
    cai.add += 0;
  } else if (method == "bicubic") {
    LOG(ERROR) << "Unsupport method: " << method;
  } else {
    LOG(ERROR) << "Unsupport method: " << method;
  }

  cai.mul *= o_batch;
  cai.div *= o_batch;
  cai.add *= o_batch;
  cai.sub *= o_batch;
  cai.exp *= o_batch;
  cai.comp *= o_batch;

  res = cai.GetIndicatorMap();

  SetLayerName(attr->layer_name);
  return res;
}

AiTraceDataFrame QnnUpsamplingProfiler::GetMemory(const Call& call_node) {
  const auto* attrs = call_node->attrs.as<QnnCSIUpSamplingAttrs>();
  SetLayerName(attrs->layer_name);
  return GetMemoryCommon(call_node);
}

RELAY_REGISTER_OP("qnn.csi.upsampling")
    .set_attr<FCalAmount>("FCalAmount", QnnUpsamplingProfiler::GetCalculationAmount);
RELAY_REGISTER_OP("qnn.csi.upsampling")
    .set_attr<FMemory>("FMemory", QnnUpsamplingProfiler::GetMemory);
RELAY_REGISTER_OP("qnn.csi.upsampling")
    .set_attr<FOpName>("FOpName", QnnUpsamplingProfiler::GetLayerName);

}  // namespace aitrace
}  // namespace relay
}  // namespace tvm
