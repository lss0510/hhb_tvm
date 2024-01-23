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
 * \file get_aitrace_data.cc
 * \brief get aitrace data implementations.
 */

#include "get_aitrace_data.h"

#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/attrs/vision.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>

#include <iostream>
#include <string>

#include "../backend/utils.h"

namespace tvm {
namespace relay {
namespace aitrace {

//------------------------------------------------------------------------------
// Common utils implements
//------------------------------------------------------------------------------

AiTraceData Convert2ATData(Array<AiTraceDataFrame> origin_data) {
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
    atblock.insn_type_ = op_map.at(op_type_str);
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

std::unordered_map<std::string, int64_t> PoolCalAmountCommon(
    const std::vector<int64_t>& in_shape, const std::vector<int64_t>& out_shape,
    const std::vector<int64_t>& kernel_shape, const std::string& data_layout,
    const std::string& op_type, const bool& is_global) {
  int64_t batch = in_shape[0];
  int32_t C_ind = Layout(data_layout).IndexOf(LayoutAxis::Get('C'));
  int64_t channel = in_shape[C_ind];
  int64_t in_spatial_size = GetSize(in_shape) / (batch * channel);
  int64_t out_spatial_size = GetSize(out_shape) / (batch * channel);
  int64_t kernel_size = is_global ? 1 : GetSize(kernel_shape);
  int64_t ops = is_global ? (in_spatial_size - 1) * batch * channel
                          : out_spatial_size * (kernel_size - 1) * batch * channel;

  int64_t comp = 0;
  int64_t add = 0;
  int64_t div = 0;
  if (op_type == "max") {
    comp = ops;
  } else if (op_type == "avg") {
    add = ops;
    div = out_spatial_size * batch * channel;
  }
  std::unordered_map<std::string, int64_t> res;
  res.insert({{"div", div}, {"add", add}, {"comp", comp}});
  return res;
}

template <typename T>
AiTraceDataFrame GetPoolCalAmountCommon(const Call& call_node, T attrs,
                                        Array<IndexExpr> kernel_shape, const std::string& op_type,
                                        const bool& is_global) {
  AiTraceDataFrame res;

  if (!call_node->checked_type_.defined()) {
    LOG(WARNING) << "The infer type pass should be called before the aitrace pass";
    return res;
  }
  Array<Expr> args = call_node->args;
  CHECK_EQ(args.size(), 1) << "The number of input arguments of the node should be 1.";
  std::string data_layout = attrs->layout;
  Array<IndexExpr> in_shape = args[0]->checked_type().as<TensorTypeNode>()->shape;
  Array<IndexExpr> out_shape = call_node->checked_type().as<TensorTypeNode>()->shape;

  std::unordered_map<std::string, int64_t> cal_amount = PoolCalAmountCommon(
      tvm::relay::backend::GetIntShape(in_shape), tvm::relay::backend::GetIntShape(out_shape),
      tvm::relay::backend::GetIntShape(kernel_shape), data_layout, op_type, is_global);

  CalculationAmontIndicator cai(cal_amount);
  res = cai.GetIndicatorMap();
  return res;
}

AiTraceDataFrame GetReluCalAmountCommon(const Call& call_node) {
  AiTraceDataFrame res;

  if (!call_node->checked_type_.defined()) {
    LOG(WARNING) << "The infer type pass should be called before the aitrace pass";
    return res;
  }
  Array<Expr> args = call_node->args;
  Array<IndexExpr> in_shape = args[0]->checked_type().as<TensorTypeNode>()->shape;

  CalculationAmontIndicator cai;
  cai.comp = GetCartesianProd(in_shape);
  res = cai.GetIndicatorMap();
  return res;
}

AiTraceDataFrame GetEltwiseCalAmountCommon(const Call& call_node, const std::string& op_type) {
  AiTraceDataFrame res;

  if (!call_node->checked_type_.defined()) {
    LOG(WARNING) << "The infer type pass should be called before the aitrace pass";
    return res;
  }
  Array<Expr> args = call_node->args;
  Array<IndexExpr> in_shape = args[0]->checked_type().as<TensorTypeNode>()->shape;
  int64_t in_size = GetCartesianProd(in_shape);

  CalculationAmontIndicator cai;
  if (op_type == "add") {
    cai.add = in_size;
  } else if (op_type == "mul") {
    cai.mul = in_size;
    cai.fused_mul_add = cai.mul;
  } else if (op_type == "max") {
    cai.comp = in_size;
  } else if (op_type == "min") {
    cai.comp = in_size;
  } else {
    CHECK(false) << "Unsupport type: " << op_type;
  }
  res = cai.GetIndicatorMap();
  return res;
}

AiTraceDataFrame GetZeroCalAmountCommon(const Call& call_node) {
  AiTraceDataFrame res;
  if (!call_node->checked_type_.defined()) {
    LOG(WARNING) << "The infer type pass should be called before the aitrace pass";
    return res;
  }
  CalculationAmontIndicator cai;
  res = cai.GetIndicatorMap();
  return res;
}

AiTraceDataFrame GetMemoryCommon(const Call& call_node) {
  AiTraceDataFrame res;
  if (!call_node->checked_type_.defined()) {
    LOG(WARNING) << "The infer type pass should be called before the aitrace pass";
    return res;
  }
  Array<IndexExpr> output_shape = call_node->checked_type().as<TensorTypeNode>()->shape;

  MemoryIndicator mi;
  mi.output = GetCartesianProd(output_shape);

  res = mi.GetIndicatorMap();
  return res;
}

//------------------------------------------------------------------------------
// Add profiler implementation
//------------------------------------------------------------------------------

AiTraceDataFrame AddProfiler::GetCalculationAmount(const Call& call_node) {
  SetLayerName(call_node);
  return GetEltwiseCalAmountCommon(call_node, "add");
}

AiTraceDataFrame AddProfiler::GetMemory(const Call& call_node) {
  SetLayerName(call_node);
  return GetMemoryCommon(call_node);
}

RELAY_REGISTER_OP("add").set_attr<FCalAmount>("FCalAmount", AddProfiler::GetCalculationAmount);
RELAY_REGISTER_OP("add").set_attr<FMemory>("FMemory", AddProfiler::GetMemory);
RELAY_REGISTER_OP("add").set_attr<FOpName>("FOpName", AddProfiler::GetLayerName);

//------------------------------------------------------------------------------
// AvgPool2d profiler implementation
//------------------------------------------------------------------------------

AiTraceDataFrame AvgPool2dProfiler::GetCalculationAmount(const Call& call_node) {
  const auto* avgpool2d_attr = call_node->attrs.as<AvgPool2DAttrs>();
  SetLayerName(call_node);
  return GetPoolCalAmountCommon(call_node, avgpool2d_attr, avgpool2d_attr->pool_size, "avg", false);
}

AiTraceDataFrame AvgPool2dProfiler::GetMemory(const Call& call_node) {
  SetLayerName(call_node);
  return GetMemoryCommon(call_node);
}

RELAY_REGISTER_OP("nn.avg_pool2d")
    .set_attr<FCalAmount>("FCalAmount", AvgPool2dProfiler::GetCalculationAmount);
RELAY_REGISTER_OP("nn.avg_pool2d").set_attr<FMemory>("FMemory", AvgPool2dProfiler::GetMemory);
RELAY_REGISTER_OP("nn.avg_pool2d").set_attr<FOpName>("FOpName", AvgPool2dProfiler::GetLayerName);

//------------------------------------------------------------------------------
// BatchFlatten profiler implementation
//------------------------------------------------------------------------------

AiTraceDataFrame BatchFlattenProfiler::GetCalculationAmount(const Call& call_node) {
  SetLayerName(call_node);
  return GetZeroCalAmountCommon(call_node);
}

AiTraceDataFrame BatchFlattenProfiler::GetMemory(const Call& call_node) {
  SetLayerName(call_node);
  return GetMemoryCommon(call_node);
}

RELAY_REGISTER_OP("nn.batch_flatten")
    .set_attr<FCalAmount>("FCalAmount", BatchFlattenProfiler::GetCalculationAmount);
RELAY_REGISTER_OP("nn.batch_flatten").set_attr<FMemory>("FMemory", BatchFlattenProfiler::GetMemory);
RELAY_REGISTER_OP("nn.batch_flatten")
    .set_attr<FOpName>("FOpName", BatchFlattenProfiler::GetLayerName);

//------------------------------------------------------------------------------
// BatchNorm profiler implementation
//------------------------------------------------------------------------------

// according to https://tvm.apache.org/docs/api/python/relay/nn.html#tvm.relay.nn.batch_norm
AiTraceDataFrame BatchNormProfiler::GetCalculationAmount(const Call& call_node) {
  AiTraceDataFrame res;
  if (!call_node->checked_type_.defined()) {
    LOG(WARNING) << "The infer type pass should be called before the aitrace pass";
    return res;
  }
  Array<Expr> args = call_node->args;
  CHECK_EQ(args.size(), 5) << "The number of input arguments of a batch_norm node should be 5.";
  const auto* batch_norm_attr = call_node->attrs.as<BatchNormAttrs>();
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
  SetLayerName(call_node);
  return res;
}

AiTraceDataFrame BatchNormProfiler::GetMemory(const Call& call_node) {
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
  SetLayerName(call_node);
  return res;
}

RELAY_REGISTER_OP("nn.batch_norm")
    .set_attr<FCalAmount>("FCalAmount", BatchNormProfiler::GetCalculationAmount);
RELAY_REGISTER_OP("nn.batch_norm").set_attr<FMemory>("FMemory", BatchNormProfiler::GetMemory);
RELAY_REGISTER_OP("nn.batch_norm").set_attr<FOpName>("FOpName", BatchNormProfiler::GetLayerName);

//------------------------------------------------------------------------------
// BiasAdd profiler implementation
//------------------------------------------------------------------------------

AiTraceDataFrame BiasAddProfiler::GetCalculationAmount(const Call& call_node) {
  SetLayerName(call_node);
  return GetEltwiseCalAmountCommon(call_node, "add");
}

AiTraceDataFrame BiasAddProfiler::GetMemory(const Call& call_node) {
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
  SetLayerName(call_node);
  return res;
}

RELAY_REGISTER_OP("nn.bias_add")
    .set_attr<FCalAmount>("FCalAmount", BiasAddProfiler::GetCalculationAmount);
RELAY_REGISTER_OP("nn.bias_add").set_attr<FMemory>("FMemory", BiasAddProfiler::GetMemory);
RELAY_REGISTER_OP("nn.bias_add").set_attr<FOpName>("FOpName", BiasAddProfiler::GetLayerName);

//------------------------------------------------------------------------------
// Concatenate profiler implementation
//------------------------------------------------------------------------------

AiTraceDataFrame ConcatenateProfiler::GetCalculationAmount(const Call& call_node) {
  SetLayerName(call_node);
  return GetZeroCalAmountCommon(call_node);
}

AiTraceDataFrame ConcatenateProfiler::GetMemory(const Call& call_node) {
  SetLayerName(call_node);
  return GetMemoryCommon(call_node);
}

RELAY_REGISTER_OP("concatenate")
    .set_attr<FCalAmount>("FCalAmount", ConcatenateProfiler::GetCalculationAmount);
RELAY_REGISTER_OP("concatenate").set_attr<FMemory>("FMemory", ConcatenateProfiler::GetMemory);
RELAY_REGISTER_OP("concatenate").set_attr<FOpName>("FOpName", ConcatenateProfiler::GetLayerName);

//------------------------------------------------------------------------------
// Conv2d profiler implementation
//------------------------------------------------------------------------------

// calculate formula:
// macc = Cin * (Hk * Wk) * Hout * Wout * Cout * batch / group
// flops = (2Cin * (Hk * Wk) -1) *Hout * Wout * Cout * batch / group
AiTraceDataFrame Conv2dProfiler::GetCalculationAmount(const Call& call_node) {
  AiTraceDataFrame res;

  if (!call_node->checked_type_.defined()) {
    LOG(WARNING) << "The infer type pass should be called before the aitrace pass";
    return res;
  }
  Array<Expr> args = call_node->args;
  CHECK_EQ(args.size(), 2) << "The number of input arguments of a CONV 2D node should be 2.";
  const auto* conv_2d_attr = call_node->attrs.as<Conv2DAttrs>();
  const auto* data_type = args[0]->checked_type().as<TensorTypeNode>();
  Array<IndexExpr> data_shape = data_type->shape;
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
            GetCartesianProd(output_tensor);

  res = cai.GetIndicatorMap();

  SetLayerName(call_node);
  return res;
}

AiTraceDataFrame Conv2dProfiler::GetMemory(const Call& call_node) {
  AiTraceDataFrame res;

  if (!call_node->checked_type_.defined()) {
    LOG(WARNING) << "The infer type pass should be called before the aitrace pass";
    return res;
  }
  Array<Expr> args = call_node->args;
  CHECK_EQ(args.size(), 2) << "The number of input arguments of a CONV 2D node should be 2.";
  const auto* conv_2d_attr = call_node->attrs.as<Conv2DAttrs>();
  const auto* data_type = args[0]->checked_type().as<TensorTypeNode>();
  Array<IndexExpr> data_shape = data_type->shape;
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
  int64_t output_channel = static_cast<int64_t>(output_tensor[C_ind].as<IntImmNode>()->value);
  if (c_ind != -1)
    output_channel *= static_cast<int64_t>(output_tensor[c_ind].as<IntImmNode>()->value);

  MemoryIndicator mi;
  mi.params +=
      GetCartesianProd(kernel_size) * input_channel * output_channel / conv_2d_attr->groups;
  mi.output += GetCartesianProd(output_tensor);

  res = mi.GetIndicatorMap();
  SetLayerName(call_node);
  return res;
}

RELAY_REGISTER_OP("nn.conv2d")
    .set_attr<FCalAmount>("FCalAmount", Conv2dProfiler::GetCalculationAmount);
RELAY_REGISTER_OP("nn.conv2d").set_attr<FMemory>("FMemory", Conv2dProfiler::GetMemory);
RELAY_REGISTER_OP("nn.conv2d").set_attr<FOpName>("FOpName", Conv2dProfiler::GetLayerName);

//------------------------------------------------------------------------------
// Conv2dTranspose profiler implementation
//------------------------------------------------------------------------------

// calculate formula:
// macc = Cin * (Hk * Wk) * Hout * Wout * Cout * batch / group
// flops = (2Cin * (Hk * Wk) -1) *Hout * Wout * Cout * batch / group
AiTraceDataFrame Conv2dTranposeProfiler::GetCalculationAmount(const Call& call_node) {
  AiTraceDataFrame res;

  if (!call_node->checked_type_.defined()) {
    LOG(WARNING) << "The infer type pass should be called before the aitrace pass";
    return res;
  }
  Array<Expr> args = call_node->args;
  CHECK_EQ(args.size(), 2)
      << "The number of input arguments of a CONV 2D Transpose node should be 2.";
  const auto* conv_2d_attr = call_node->attrs.as<Conv2DTransposeAttrs>();
  const auto* data_type = args[0]->checked_type().as<TensorTypeNode>();
  Array<IndexExpr> data_shape = data_type->shape;
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
            GetCartesianProd(output_tensor);

  res = cai.GetIndicatorMap();

  SetLayerName(call_node);
  return res;
}

AiTraceDataFrame Conv2dTranposeProfiler::GetMemory(const Call& call_node) {
  AiTraceDataFrame res;

  if (!call_node->checked_type_.defined()) {
    LOG(WARNING) << "The infer type pass should be called before the aitrace pass";
    return res;
  }
  Array<Expr> args = call_node->args;
  CHECK_EQ(args.size(), 2)
      << "The number of input arguments of a CONV 2D Transpose node should be 2.";
  const auto* conv_2d_attr = call_node->attrs.as<Conv2DTransposeAttrs>();
  const auto* data_type = args[0]->checked_type().as<TensorTypeNode>();
  Array<IndexExpr> data_shape = data_type->shape;
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
  int64_t output_channel = static_cast<int64_t>(output_tensor[C_ind].as<IntImmNode>()->value);
  if (c_ind != -1)
    output_channel *= static_cast<int64_t>(output_tensor[c_ind].as<IntImmNode>()->value);

  MemoryIndicator mi;
  mi.params +=
      GetCartesianProd(kernel_size) * input_channel * output_channel / conv_2d_attr->groups;
  mi.output += GetCartesianProd(output_tensor);

  res = mi.GetIndicatorMap();
  SetLayerName(call_node);
  return res;
}

RELAY_REGISTER_OP("nn.conv2d_transpose")
    .set_attr<FCalAmount>("FCalAmount", Conv2dTranposeProfiler::GetCalculationAmount);
RELAY_REGISTER_OP("nn.conv2d_transpose")
    .set_attr<FMemory>("FMemory", Conv2dTranposeProfiler::GetMemory);
RELAY_REGISTER_OP("nn.conv2d_transpose")
    .set_attr<FOpName>("FOpName", Conv2dTranposeProfiler::GetLayerName);

//------------------------------------------------------------------------------
// Dense profiler implementation
//------------------------------------------------------------------------------

// calculate formula:
//                      out        =             X           *        W^T
// shape: (d1,d2,...,dn, unit_out) = (d1,d2,...,dn, unit_in) * (unit_in, unit_out)
// macc = prod(d1, d2, ..., dn) * unit_in * unit_out
// flops = prod(d1, d2, ..., dn) * (2unit_in-1) * unit_out
AiTraceDataFrame DenseProfiler::GetCalculationAmount(const Call& call_node) {
  AiTraceDataFrame res;
  if (!call_node->checked_type_.defined()) {
    LOG(WARNING) << "The infer type pass should be called before the aitrace pass";
    return res;
  }
  Array<Expr> args = call_node->args;
  CHECK_EQ(args.size(), 2) << "The number of input arguments of a Dense node should be 2.";
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
  cai.add = d_prod * (unit_in - 1) * unit_out;
  res = cai.GetIndicatorMap();

  SetLayerName(call_node);
  return res;
}

AiTraceDataFrame DenseProfiler::GetMemory(const Call& call_node) {
  AiTraceDataFrame res;
  if (!call_node->checked_type_.defined()) {
    LOG(WARNING) << "The infer type pass should be called before the aitrace pass";
    return res;
  }
  Array<Expr> args = call_node->args;
  CHECK_EQ(args.size(), 2) << "The number of input arguments of a Dense node should be 2.";
  Array<IndexExpr> weight_shape = args[1]->checked_type().as<TensorTypeNode>()->shape;
  Array<IndexExpr> output_shape = call_node->checked_type().as<TensorTypeNode>()->shape;

  MemoryIndicator mi;
  mi.params = GetCartesianProd(weight_shape);
  mi.output = GetCartesianProd(output_shape);

  res = mi.GetIndicatorMap();
  SetLayerName(call_node);
  return res;
}

RELAY_REGISTER_OP("nn.dense")
    .set_attr<FCalAmount>("FCalAmount", DenseProfiler::GetCalculationAmount);
RELAY_REGISTER_OP("nn.dense").set_attr<FMemory>("FMemory", DenseProfiler::GetMemory);
RELAY_REGISTER_OP("nn.dense").set_attr<FOpName>("FOpName", DenseProfiler::GetLayerName);

//------------------------------------------------------------------------------
// Dropout profiler implementation
//------------------------------------------------------------------------------

AiTraceDataFrame DropoutProfiler ::GetCalculationAmount(const Call& call_node) {
  SetLayerName(call_node);
  return GetZeroCalAmountCommon(call_node);
}

AiTraceDataFrame DropoutProfiler::GetMemory(const Call& call_node) {
  AiTraceDataFrame res;
  if (!call_node->checked_type_.defined()) {
    LOG(WARNING) << "The infer type pass should be called before the aitrace pass";
    return res;
  }
  Array<Expr> args = call_node->args;
  Array<IndexExpr> input_shape = args[0]->checked_type().as<TensorTypeNode>()->shape;

  MemoryIndicator mi;
  // There are two outputs of dropout in relay
  // the second output: a mask tensor (1.0 where element not dropped, 0.0 where dropped)
  mi.output = GetCartesianProd(input_shape) * 2;

  res = mi.GetIndicatorMap();
  SetLayerName(call_node);
  return res;
}

RELAY_REGISTER_OP("nn.dropout")
    .set_attr<FCalAmount>("FCalAmount", DropoutProfiler::GetCalculationAmount);
RELAY_REGISTER_OP("nn.dropout").set_attr<FMemory>("FMemory", DropoutProfiler::GetMemory);
RELAY_REGISTER_OP("nn.dropout").set_attr<FOpName>("FOpName", DropoutProfiler::GetLayerName);

//------------------------------------------------------------------------------
// GlobalAvgpool2d profiler implementation
//------------------------------------------------------------------------------

AiTraceDataFrame GlobalAvgPool2dProfiler::GetCalculationAmount(const Call& call_node) {
  const auto* globalpool2d_attr = call_node->attrs.as<GlobalPool2DAttrs>();
  SetLayerName(call_node);
  return GetPoolCalAmountCommon(call_node, globalpool2d_attr, Array<IndexExpr>({1, 1}), "avg",
                                true);
}

AiTraceDataFrame GlobalAvgPool2dProfiler::GetMemory(const Call& call_node) {
  SetLayerName(call_node);
  return GetMemoryCommon(call_node);
}

RELAY_REGISTER_OP("nn.global_avg_pool2d")
    .set_attr<FCalAmount>("FCalAmount", GlobalAvgPool2dProfiler::GetCalculationAmount);
RELAY_REGISTER_OP("nn.global_avg_pool2d")
    .set_attr<FMemory>("FMemory", GlobalAvgPool2dProfiler::GetMemory);
RELAY_REGISTER_OP("nn.global_avg_pool2d")
    .set_attr<FOpName>("FOpName", GlobalAvgPool2dProfiler::GetLayerName);

//------------------------------------------------------------------------------
// GlobalMaxpool2d profiler implementation
//------------------------------------------------------------------------------

AiTraceDataFrame GlobalMaxPool2dProfiler::GetCalculationAmount(const Call& call_node) {
  const auto* globalpool2d_attr = call_node->attrs.as<GlobalPool2DAttrs>();
  SetLayerName(call_node);
  return GetPoolCalAmountCommon(call_node, globalpool2d_attr, Array<IndexExpr>({1, 1}), "max",
                                true);
}

AiTraceDataFrame GlobalMaxPool2dProfiler::GetMemory(const Call& call_node) {
  SetLayerName(call_node);
  return GetMemoryCommon(call_node);
}

RELAY_REGISTER_OP("nn.global_max_pool2d")
    .set_attr<FCalAmount>("FCalAmount", GlobalMaxPool2dProfiler::GetCalculationAmount);
RELAY_REGISTER_OP("nn.global_max_pool2d")
    .set_attr<FMemory>("FMemory", GlobalMaxPool2dProfiler::GetMemory);
RELAY_REGISTER_OP("nn.global_max_pool2d")
    .set_attr<FOpName>("FOpName", GlobalMaxPool2dProfiler::GetLayerName);

//------------------------------------------------------------------------------
// LRN profiler implementation
//------------------------------------------------------------------------------

// output = in_data / (bias + (alpha/size)sum(in_data^2))^beta
AiTraceDataFrame LRNProfiler::GetCalculationAmount(const Call& call_node) {
  AiTraceDataFrame res;
  if (!call_node->checked_type_.defined()) {
    LOG(WARNING) << "The infer type pass should be called before the aitrace pass";
    return res;
  }
  Array<Expr> args = call_node->args;
  CHECK_EQ(args.size(), 1) << "The number of input arguments of a LRN node should be 1.";

  Array<IndexExpr> in_shape = args[0]->checked_type().as<TensorTypeNode>()->shape;
  const auto* lrn_attrs = call_node->attrs.as<LRNAttrs>();
  int size = lrn_attrs->size;

  int64_t num_inputs = GetCartesianProd(in_shape);
  CalculationAmontIndicator cai;
  cai.fused_mul_add = num_inputs * (size + 1) + num_inputs;  // sum(in_data^2) and (alpha/size)*...
  cai.mul = cai.fused_mul_add;                               // in_data^2
  cai.add = num_inputs * size + num_inputs;                  // (bias + ...) and sum(...)
  cai.exp = num_inputs;                                      // (...)^beta;
  cai.div = num_inputs;                                      // in_data/(...)
  res = cai.GetIndicatorMap();
  SetLayerName(call_node);
  return res;
}

AiTraceDataFrame LRNProfiler::GetMemory(const Call& call_node) {
  SetLayerName(call_node);
  return GetMemoryCommon(call_node);
}

RELAY_REGISTER_OP("nn.lrn").set_attr<FCalAmount>("FCalAmount", LRNProfiler::GetCalculationAmount);
RELAY_REGISTER_OP("nn.lrn").set_attr<FMemory>("FMemory", LRNProfiler::GetMemory);
RELAY_REGISTER_OP("nn.lrn").set_attr<FOpName>("FOpName", LRNProfiler::GetLayerName);

//------------------------------------------------------------------------------
// L2_Normalize profiler implementation
//------------------------------------------------------------------------------

// output(i,j) = x(i, j) / sqrt(max(sum(x^2), eps))
AiTraceDataFrame L2NormalizeNProfiler::GetCalculationAmount(const Call& call_node) {
  AiTraceDataFrame res;
  if (!call_node->checked_type_.defined()) {
    LOG(WARNING) << "The infer type pass should be called before the aitrace pass";
    return res;
  }
  Array<Expr> args = call_node->args;
  CHECK_EQ(args.size(), 1) << "The number of input arguments of a l2_normalize node should be 1.";

  Array<IndexExpr> in_shape = args[0]->checked_type().as<TensorTypeNode>()->shape;
  int64_t in_size = GetCartesianProd(in_shape);
  const auto* l2_norm_attrs = call_node->attrs.as<L2NormalizeAttrs>();
  Array<Integer> axis = l2_norm_attrs->axis;
  int64_t axis_size = 1;
  for (auto a : axis) {
    int64_t a_value = static_cast<int64_t>(a.as<IntImmNode>()->value);
    axis_size *= in_shape[a_value].as<IntImmNode>()->value;
  }
  int64_t excluded_axis_size = in_size / axis_size;

  CalculationAmontIndicator cai;
  cai.div = in_size;
  cai.exp = in_size;
  cai.comp = in_size;

  cai.mul = in_size;
  cai.add = (axis_size - 1) * excluded_axis_size;
  cai.fused_mul_add = cai.mul;

  res = cai.GetIndicatorMap();
  SetLayerName(call_node);
  return res;
}

AiTraceDataFrame L2NormalizeNProfiler::GetMemory(const Call& call_node) {
  SetLayerName(call_node);
  return GetMemoryCommon(call_node);
}

RELAY_REGISTER_OP("nn.l2_normalize")
    .set_attr<FCalAmount>("FCalAmount", L2NormalizeNProfiler::GetCalculationAmount);
RELAY_REGISTER_OP("nn.l2_normalize").set_attr<FMemory>("FMemory", L2NormalizeNProfiler::GetMemory);
RELAY_REGISTER_OP("nn.l2_normalize")
    .set_attr<FOpName>("FOpName", L2NormalizeNProfiler::GetLayerName);

//------------------------------------------------------------------------------
// Maximum profiler implementation
//------------------------------------------------------------------------------

AiTraceDataFrame MaximumProfiler::GetCalculationAmount(const Call& call_node) {
  SetLayerName(call_node);
  return GetEltwiseCalAmountCommon(call_node, "max");
}

AiTraceDataFrame MaximumProfiler::GetMemory(const Call& call_node) {
  SetLayerName(call_node);
  return GetMemoryCommon(call_node);
}

RELAY_REGISTER_OP("maximum").set_attr<FCalAmount>("FCalAmount",
                                                  MaximumProfiler::GetCalculationAmount);
RELAY_REGISTER_OP("maximum").set_attr<FMemory>("FMemory", MaximumProfiler::GetMemory);
RELAY_REGISTER_OP("maximum").set_attr<FOpName>("FOpName", MaximumProfiler::GetLayerName);

//------------------------------------------------------------------------------
// Maxpool2d profiler implementation
//------------------------------------------------------------------------------

AiTraceDataFrame MaxPool2dProfiler::GetCalculationAmount(const Call& call_node) {
  const auto* maxpool2d_attr = call_node->attrs.as<MaxPool2DAttrs>();
  SetLayerName(call_node);
  return GetPoolCalAmountCommon(call_node, maxpool2d_attr, maxpool2d_attr->pool_size, "max", false);
}

AiTraceDataFrame MaxPool2dProfiler::GetMemory(const Call& call_node) {
  SetLayerName(call_node);
  return GetMemoryCommon(call_node);
}

RELAY_REGISTER_OP("nn.max_pool2d")
    .set_attr<FCalAmount>("FCalAmount", MaxPool2dProfiler::GetCalculationAmount);
RELAY_REGISTER_OP("nn.max_pool2d").set_attr<FMemory>("FMemory", MaxPool2dProfiler::GetMemory);
RELAY_REGISTER_OP("nn.max_pool2d").set_attr<FOpName>("FOpName", MaxPool2dProfiler::GetLayerName);

//------------------------------------------------------------------------------
// Maxpool2dLocation profiler implementation
//------------------------------------------------------------------------------

AiTraceDataFrame MaxPool2dLocationProfiler::GetCalculationAmount(const Call& call_node) {
  const auto* maxpool2d_attr = call_node->attrs.as<MaxPool2dLocationAttrs>();
  SetLayerName(call_node);
  return GetPoolCalAmountCommon(call_node, maxpool2d_attr, maxpool2d_attr->pool_size, "max", false);
}

AiTraceDataFrame MaxPool2dLocationProfiler::GetMemory(const Call& call_node) {
  SetLayerName(call_node);
  return GetMemoryCommon(call_node);
}

RELAY_REGISTER_OP("vision.max_pool2d_location")
    .set_attr<FCalAmount>("FCalAmount", MaxPool2dLocationProfiler::GetCalculationAmount);
RELAY_REGISTER_OP("vision.max_pool2d_location")
    .set_attr<FMemory>("FMemory", MaxPool2dLocationProfiler::GetMemory);
RELAY_REGISTER_OP("vision.max_pool2d_location")
    .set_attr<FOpName>("FOpName", MaxPool2dLocationProfiler::GetLayerName);

//------------------------------------------------------------------------------
// Maxpool2dWithArgmax profiler implementation
//------------------------------------------------------------------------------

AiTraceDataFrame MaxPool2dWithArgmaxProfiler::GetCalculationAmount(const Call& call_node) {
  const auto* maxpool2d_attr = call_node->attrs.as<MaxPool2DAttrs>();
  SetLayerName(call_node);
  return GetPoolCalAmountCommon(call_node, maxpool2d_attr, maxpool2d_attr->pool_size, "max", false);
}

AiTraceDataFrame MaxPool2dWithArgmaxProfiler::GetMemory(const Call& call_node) {
  SetLayerName(call_node);
  return GetMemoryCommon(call_node);
}

RELAY_REGISTER_OP("nn.max_pool2d_with_argmax")
    .set_attr<FCalAmount>("FCalAmount", MaxPool2dWithArgmaxProfiler::GetCalculationAmount);
RELAY_REGISTER_OP("nn.max_pool2d_with_argmax")
    .set_attr<FMemory>("FMemory", MaxPool2dWithArgmaxProfiler::GetMemory);
RELAY_REGISTER_OP("nn.max_pool2d_with_argmax")
    .set_attr<FOpName>("FOpName", MaxPool2dWithArgmaxProfiler::GetLayerName);

//------------------------------------------------------------------------------
// Multiply profiler implementation
//------------------------------------------------------------------------------

AiTraceDataFrame MultiplyProfiler::GetCalculationAmount(const Call& call_node) {
  SetLayerName(call_node);
  return GetEltwiseCalAmountCommon(call_node, "mul");
}

AiTraceDataFrame MultiplyProfiler::GetMemory(const Call& call_node) {
  SetLayerName(call_node);
  return GetMemoryCommon(call_node);
}

RELAY_REGISTER_OP("multiply")
    .set_attr<FCalAmount>("FCalAmount", MultiplyProfiler::GetCalculationAmount);
RELAY_REGISTER_OP("multiply").set_attr<FMemory>("FMemory", MultiplyProfiler::GetMemory);
RELAY_REGISTER_OP("multiply").set_attr<FOpName>("FOpName", MultiplyProfiler::GetLayerName);

//------------------------------------------------------------------------------
// PRelu profiler implementation
//------------------------------------------------------------------------------

AiTraceDataFrame PreluProfiler::GetCalculationAmount(const Call& call_node) {
  SetLayerName(call_node);
  return GetReluCalAmountCommon(call_node);
}

AiTraceDataFrame PreluProfiler::GetMemory(const Call& call_node) {
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
  SetLayerName(call_node);
  return res;
}

RELAY_REGISTER_OP("nn.prelu")
    .set_attr<FCalAmount>("FCalAmount", PreluProfiler::GetCalculationAmount);
RELAY_REGISTER_OP("nn.prelu").set_attr<FMemory>("FMemory", PreluProfiler::GetMemory);
RELAY_REGISTER_OP("nn.prelu").set_attr<FOpName>("FOpName", PreluProfiler::GetLayerName);

//------------------------------------------------------------------------------
// Proposal profiler implementation
//------------------------------------------------------------------------------

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
AiTraceDataFrame ProposalProfiler::GetCalculationAmount(const Call& call_node) {
  AiTraceDataFrame res;

  if (!call_node->checked_type_.defined()) {
    LOG(WARNING) << "The infer type pass should be called before the aitrace pass";
    return res;
  }
  Array<Expr> args = call_node->args;
  CHECK_EQ(args.size(), 3) << "The number of input arguments of a Proposal node should be 3.";
  const auto* proposal_attr = call_node->attrs.as<ProposalAttrs>();
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

  SetLayerName(call_node);
  return res;
}

AiTraceDataFrame ProposalProfiler::GetMemory(const Call& call_node) {
  SetLayerName(call_node);
  return GetMemoryCommon(call_node);
}

RELAY_REGISTER_OP("vision.proposal")
    .set_attr<FCalAmount>("FCalAmount", ProposalProfiler::GetCalculationAmount);
RELAY_REGISTER_OP("vision.proposal").set_attr<FMemory>("FMemory", ProposalProfiler::GetMemory);
RELAY_REGISTER_OP("vision.proposal").set_attr<FOpName>("FOpName", ProposalProfiler::GetLayerName);

//------------------------------------------------------------------------------
// PSRoiPooling profiler implementation
//------------------------------------------------------------------------------

// according to: python/tvm/topi/vision/psroipooling.py
AiTraceDataFrame PsroipoolingProfiler::GetCalculationAmount(const Call& call_node) {
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
  const auto* psroi_pool_attr = call_node->attrs.as<PSRoIPoolingAttrs>();
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

  SetLayerName(call_node);
  return res;
}

AiTraceDataFrame PsroipoolingProfiler::GetMemory(const Call& call_node) {
  SetLayerName(call_node);
  return GetMemoryCommon(call_node);
}

RELAY_REGISTER_OP("vision.psroipooling")
    .set_attr<FCalAmount>("FCalAmount", PsroipoolingProfiler::GetCalculationAmount);
RELAY_REGISTER_OP("vision.psroipooling")
    .set_attr<FMemory>("FMemory", PsroipoolingProfiler::GetMemory);
RELAY_REGISTER_OP("vision.psroipooling")
    .set_attr<FOpName>("FOpName", PsroipoolingProfiler::GetLayerName);

//------------------------------------------------------------------------------
// Relu profiler implementation
//------------------------------------------------------------------------------

AiTraceDataFrame ReluProfiler::GetCalculationAmount(const Call& call_node) {
  SetLayerName(call_node);
  return GetReluCalAmountCommon(call_node);
}

AiTraceDataFrame ReluProfiler::GetMemory(const Call& call_node) {
  SetLayerName(call_node);
  return GetMemoryCommon(call_node);
}

RELAY_REGISTER_OP("nn.relu").set_attr<FCalAmount>("FCalAmount", ReluProfiler::GetCalculationAmount);
RELAY_REGISTER_OP("nn.relu").set_attr<FMemory>("FMemory", ReluProfiler::GetMemory);
RELAY_REGISTER_OP("nn.relu").set_attr<FOpName>("FOpName", ReluProfiler::GetLayerName);

//------------------------------------------------------------------------------
// Reshape profiler implementation
//------------------------------------------------------------------------------

AiTraceDataFrame ReshapeProfiler::GetCalculationAmount(const Call& call_node) {
  SetLayerName(call_node);
  return GetZeroCalAmountCommon(call_node);
}

AiTraceDataFrame ReshapeProfiler::GetMemory(const Call& call_node) {
  SetLayerName(call_node);
  return GetMemoryCommon(call_node);
}

RELAY_REGISTER_OP("reshape").set_attr<FCalAmount>("FCalAmount",
                                                  ReshapeProfiler::GetCalculationAmount);
RELAY_REGISTER_OP("reshape").set_attr<FMemory>("FMemory", ReshapeProfiler::GetMemory);
RELAY_REGISTER_OP("reshape").set_attr<FOpName>("FOpName", ReshapeProfiler::GetLayerName);

//------------------------------------------------------------------------------
// RoiPool profiler implementation
//------------------------------------------------------------------------------

// according to:
// https://github.com/rbgirshick/caffe-fast-rcnn/blob/0dcd397b29507b8314e252e850518c5695efbb83/src/caffe/layers/roi_pooling_layer.cpp
AiTraceDataFrame RoiPoolProfiler::GetCalculationAmount(const Call& call_node) {
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
  const auto* roi_pool_attr = call_node->attrs.as<ROIPoolAttrs>();
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
  SetLayerName(call_node);
  return res;
}

AiTraceDataFrame RoiPoolProfiler::GetMemory(const Call& call_node) {
  SetLayerName(call_node);
  return GetMemoryCommon(call_node);
}

RELAY_REGISTER_OP("vision.roi_pool")
    .set_attr<FCalAmount>("FCalAmount", RoiPoolProfiler::GetCalculationAmount);
RELAY_REGISTER_OP("vision.roi_pool").set_attr<FMemory>("FMemory", RoiPoolProfiler::GetMemory);
RELAY_REGISTER_OP("vision.roi_pool").set_attr<FOpName>("FOpName", RoiPoolProfiler::GetLayerName);

//------------------------------------------------------------------------------
// Sigmoid profiler implementation
//------------------------------------------------------------------------------

AiTraceDataFrame SigmoidProfiler::GetCalculationAmount(const Call& call_node) {
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
  SetLayerName(call_node);
  return res;
}

AiTraceDataFrame SigmoidProfiler::GetMemory(const Call& call_node) {
  SetLayerName(call_node);
  return GetMemoryCommon(call_node);
}

RELAY_REGISTER_OP("sigmoid").set_attr<FCalAmount>("FCalAmount",
                                                  SigmoidProfiler::GetCalculationAmount);
RELAY_REGISTER_OP("sigmoid").set_attr<FMemory>("FMemory", SigmoidProfiler::GetMemory);
RELAY_REGISTER_OP("sigmoid").set_attr<FOpName>("FOpName", SigmoidProfiler::GetLayerName);

//------------------------------------------------------------------------------
// Softmax profiler implementation
//------------------------------------------------------------------------------

AiTraceDataFrame SoftmaxProfiler::GetCalculationAmount(const Call& call_node) {
  AiTraceDataFrame res;
  if (!call_node->checked_type_.defined()) {
    LOG(WARNING) << "The infer type pass should be called before the aitrace pass";
    return res;
  }
  Array<Expr> args = call_node->args;
  CHECK_EQ(args.size(), 1) << "The number of input arguments of a Softmax node should be 1.";

  Array<IndexExpr> in_shape = args[0]->checked_type().as<TensorTypeNode>()->shape;
  int64_t num_inputs = GetCartesianProd(in_shape);
  const auto* soft_attr = call_node->attrs.as<SoftmaxAttrs>();
  int axis = soft_attr->axis;
  int64_t axis_shape = in_shape[axis].as<IntImmNode>()->value;
  CalculationAmontIndicator cai;
  cai.exp = num_inputs;
  cai.add = (num_inputs / axis_shape - 1) * axis_shape;
  cai.div = num_inputs;
  res = cai.GetIndicatorMap();
  SetLayerName(call_node);
  return res;
}

AiTraceDataFrame SoftmaxProfiler::GetMemory(const Call& call_node) {
  SetLayerName(call_node);
  return GetMemoryCommon(call_node);
}

RELAY_REGISTER_OP("nn.softmax")
    .set_attr<FCalAmount>("FCalAmount", SoftmaxProfiler::GetCalculationAmount);
RELAY_REGISTER_OP("nn.softmax").set_attr<FMemory>("FMemory", SoftmaxProfiler::GetMemory);
RELAY_REGISTER_OP("nn.softmax").set_attr<FOpName>("FOpName", SoftmaxProfiler::GetLayerName);

//------------------------------------------------------------------------------
// Split profiler implementation
//------------------------------------------------------------------------------

AiTraceDataFrame SplitProfiler::GetCalculationAmount(const Call& call_node) {
  SetLayerName(call_node);
  return GetZeroCalAmountCommon(call_node);
}

AiTraceDataFrame SplitProfiler::GetMemory(const Call& call_node) {
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
  SetLayerName(call_node);
  return res;
}

RELAY_REGISTER_OP("split").set_attr<FCalAmount>("FCalAmount", SplitProfiler::GetCalculationAmount);
RELAY_REGISTER_OP("split").set_attr<FMemory>("FMemory", SplitProfiler::GetMemory);
RELAY_REGISTER_OP("split").set_attr<FOpName>("FOpName", SplitProfiler::GetLayerName);

//------------------------------------------------------------------------------
// StridedSlice profiler implementation
//------------------------------------------------------------------------------

AiTraceDataFrame StridedSliceProfiler::GetCalculationAmount(const Call& call_node) {
  SetLayerName(call_node);
  return GetZeroCalAmountCommon(call_node);
}

AiTraceDataFrame StridedSliceProfiler::GetMemory(const Call& call_node) {
  AiTraceDataFrame res;
  if (!call_node->checked_type_.defined()) {
    LOG(WARNING) << "The infer type pass should be called before the aitrace pass";
    return res;
  }
  Array<IndexExpr> output_shape = call_node->checked_type().as<TensorTypeNode>()->shape;

  MemoryIndicator mi;
  mi.output += GetCartesianProd(output_shape);

  res = mi.GetIndicatorMap();
  SetLayerName(call_node);
  return res;
}

RELAY_REGISTER_OP("strided_slice")
    .set_attr<FCalAmount>("FCalAmount", StridedSliceProfiler::GetCalculationAmount);
RELAY_REGISTER_OP("strided_slice").set_attr<FMemory>("FMemory", StridedSliceProfiler::GetMemory);
RELAY_REGISTER_OP("strided_slice").set_attr<FOpName>("FOpName", StridedSliceProfiler::GetLayerName);

//------------------------------------------------------------------------------
// Tanh profiler implementation
//------------------------------------------------------------------------------

AiTraceDataFrame TanhProfiler::GetCalculationAmount(const Call& call_node) {
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
  SetLayerName(call_node);
  return res;
}

AiTraceDataFrame TanhProfiler::GetMemory(const Call& call_node) {
  SetLayerName(call_node);
  return GetMemoryCommon(call_node);
}

RELAY_REGISTER_OP("tanh").set_attr<FCalAmount>("FCalAmount", TanhProfiler::GetCalculationAmount);
RELAY_REGISTER_OP("tanh").set_attr<FMemory>("FMemory", TanhProfiler::GetMemory);
RELAY_REGISTER_OP("tanh").set_attr<FOpName>("FOpName", TanhProfiler::GetLayerName);

//------------------------------------------------------------------------------
// Transpose profiler implementation
//------------------------------------------------------------------------------

AiTraceDataFrame TransposeProfiler::GetCalculationAmount(const Call& call_node) {
  SetLayerName(call_node);
  return GetZeroCalAmountCommon(call_node);
}

AiTraceDataFrame TransposeProfiler::GetMemory(const Call& call_node) {
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
  SetLayerName(call_node);
  return res;
}

RELAY_REGISTER_OP("transpose")
    .set_attr<FCalAmount>("FCalAmount", TransposeProfiler::GetCalculationAmount);
RELAY_REGISTER_OP("transpose").set_attr<FMemory>("FMemory", TransposeProfiler::GetMemory);
RELAY_REGISTER_OP("transpose").set_attr<FOpName>("FOpName", TransposeProfiler::GetLayerName);

//------------------------------------------------------------------------------
// unpooling profiler implementation
//------------------------------------------------------------------------------

AiTraceDataFrame UnpoolingProfiler::GetCalculationAmount(const Call& call_node) {
  SetLayerName(call_node);
  return GetZeroCalAmountCommon(call_node);
}

AiTraceDataFrame UnpoolingProfiler::GetMemory(const Call& call_node) {
  SetLayerName(call_node);
  return GetMemoryCommon(call_node);
}

RELAY_REGISTER_OP("vision.unpooling")
    .set_attr<FCalAmount>("FCalAmount", UnpoolingProfiler::GetCalculationAmount);
RELAY_REGISTER_OP("vision.unpooling").set_attr<FMemory>("FMemory", UnpoolingProfiler::GetMemory);
RELAY_REGISTER_OP("vision.unpooling").set_attr<FOpName>("FOpName", UnpoolingProfiler::GetLayerName);

//------------------------------------------------------------------------------
// upsampling profiler implementation
//------------------------------------------------------------------------------

AiTraceDataFrame UpsamplingProfiler::GetCalculationAmount(const Call& call_node) {
  AiTraceDataFrame res;
  if (!call_node->checked_type_.defined()) {
    LOG(WARNING) << "The infer type pass should be called before the aitrace pass";
    return res;
  }
  Array<Expr> args = call_node->args;
  CHECK_EQ(args.size(), 1) << "The number of input arguments of a upsampling node should be 1.";

  const auto* attr = call_node->attrs.as<UpSamplingAttrs>();
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
  SetLayerName(call_node);
  return res;
}

AiTraceDataFrame UpsamplingProfiler::GetMemory(const Call& call_node) {
  SetLayerName(call_node);
  return GetMemoryCommon(call_node);
}

RELAY_REGISTER_OP("nn.upsampling")
    .set_attr<FCalAmount>("FCalAmount", UpsamplingProfiler::GetCalculationAmount);
RELAY_REGISTER_OP("nn.upsampling").set_attr<FMemory>("FMemory", UpsamplingProfiler::GetMemory);
RELAY_REGISTER_OP("nn.upsampling").set_attr<FOpName>("FOpName", UpsamplingProfiler::GetLayerName);

}  // namespace aitrace
}  // namespace relay
}  // namespace tvm
