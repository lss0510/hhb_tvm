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
 * \file src/relay/backend/contrib/csinn/codegen.cc
 * \brief Implementation of CSINN codegen APIs.
 */

#include "csinn.h"

#include "gref.h"
#include "th1520.h"

using namespace tvm::relay::qnn;
namespace tvm {
namespace relay {
namespace contrib {
namespace csinn {

void CodegenCSINN::phase1() {
  QuantConfig* set_config = new QuantConfig(expr_);
  set_config->set_quant_config(cfg);
  if (hybrid_cfg->quantization_scheme != "unset") {
    set_config->set_hybrid_quant_config(hybrid_cfg, hybrid_layer_name);
  }
  expr_ = set_config->get_expr();

  QuantInfo* qinfo = new QuantInfo(expr_);
  qinfo->calculate_quant_info();
  expr_ = qinfo->get_expr();

  if (hybrid_cfg->quantization_scheme != "unset") {
    DataConvertInserter* insert_dc = new DataConvertInserter(expr_);
    insert_dc->Insert(cfg);
    expr_ = insert_dc->get_expr();
  }
}

void CodegenCSINN::phase2() { visit(expr_); }

void CodegenCSINN::compile(const Expr& expr) {
  expr_ = expr;
  opt_start();
  optimization();
  opt_end();
}

void CodegenCSINN::visit_expr(const VarNode* node) {
  first_visit_expr = false;
  ext_func_args_.push_back(GetRef<Var>(node));
  out_.clear();
  output_element output;
  output.name = node->name_hint();
  output.need_copy = false;
  auto output_shape = node->get_shape();
  output.shape = output_shape;
  output.size = -1;
  out_.push_back(output);
  out_list_.push_back(output);
}

void CodegenCSINN::visit(const Expr& expr) {
  auto it = visit_counter_.find(expr.get());
  if (it != visit_counter_.end()) {
    if (auto const_node = expr.as<ConstantNode>()) {
      constant_.clear();
      CSIConstant* constant =
          new CSIConstant(GetDtypeString(const_node->data.DataType()), const_node->get_shape());
      constant->set_name("constant_" + to_string(const_idx_++));
      const_node->data.CopyToBytes(constant->get_data_buf(), constant->byte_size());
      constant_.push_back(constant);
    }
    ++it->second;
  } else {
    using TParent = CSINNExprFunctor<void(const Expr&)>;
    TParent::visit(expr);
    visit_counter_.insert({expr.get(), 1});
  }
}

void CodegenCSINN::visit_expr(const ConstantNode* node) {
  first_visit_expr = false;
  constant_.clear();
  CSIConstant* constant = new CSIConstant(GetDtypeString(node->data.DataType()), node->get_shape());
  constant->set_name("constant_" + to_string(const_idx_++));
  node->data.CopyToBytes(constant->get_data_buf(), constant->byte_size());
  constant_.push_back(constant);
}

void CodegenCSINN::visit_expr(const TupleNode* op) {
  if (first_visit_expr) {
    // output expr
    first_visit_expr = false;
    for (auto field : op->fields) {
      auto const_node = field.as<ConstantNode>();
      if (const_node) {
        CHECK(0) << "Unsupport constant output\n";
      } else {
        auto call_node = field.as<CallNode>();
        CHECK(call_node);
        output_element output;
        output.call = call_node;
        output_list_.push_back(output);
      }
    }
    for (auto field : op->fields) {
      auto const_node = field.as<ConstantNode>();
      if (!const_node) {
        visit(field);
      }
    }
  } else {
    // other expr
    for (auto field : op->fields) {
      visit(field);
    }
  }
}

bool CodegenCSINN::InOpList(const CallNode* call) {
  for (auto op : target_op_list) {
    if (IsOp(call, op)) {
      return true;
    }
  }
  return false;
}

void CodegenCSINN::visit_expr(const CallNode* call) {
  /* Get the arguments for various CSINN kernels. */
  /* QNN op */
  if (first_visit_expr) {
    first_visit_expr = false;
    output_element output;
    output.call = call;
    output_list_.push_back(output);
  }
  if (IsOp(call, "qnn.csi.abs")) {
    Unary(call, "abs");
  } else if (IsOp(call, "qnn.csi.acos")) {
    Unary(call, "acos");
  } else if (IsOp(call, "qnn.csi.acosh")) {
    Unary(call, "acosh");
  } else if (IsOp(call, "qnn.csi.add")) {
    DisoOp(call, "add");
  } else if (IsOp(call, "qnn.csi.argmax")) {
    Reduce(call, "argmax", "int32_t");
  } else if (IsOp(call, "qnn.csi.argmin")) {
    Reduce(call, "argmin", "int32_t");
  } else if (IsOp(call, "qnn.csi.asin")) {
    Unary(call, "asin");
  } else if (IsOp(call, "qnn.csi.asinh")) {
    Unary(call, "asinh");
  } else if (IsOp(call, "qnn.csi.atan")) {
    Unary(call, "atan");
  } else if (IsOp(call, "qnn.csi.atanh")) {
    Unary(call, "atanh");
  } else if (IsOp(call, "qnn.csi.avgpool2d")) {
    AvgPool2d(call);
  } else if (IsOp(call, "qnn.csi.avgpool3d")) {
    AvgPool3d(call);
  } else if (IsOp(call, "qnn.csi.batch_to_space_nd")) {
    BatchToSpaceND(call);
  } else if (IsOp(call, "qnn.csi.bias_add")) {
    DisoOp(call, "add");
  } else if (IsOp(call, "qnn.csi.broadcast_to")) {
    BroadCastTo(call);
  } else if (IsOp(call, "qnn.csi.cast")) {
    Cast(call);
  } else if (IsOp(call, "qnn.csi.ceil")) {
    Unary(call, "ceil");
  } else if (IsOp(call, "qnn.csi.cache_matmul")) {
    CacheMatMul(call);
  } else if (IsOp(call, "qnn.csi.cache_conv1d")) {
    CacheConv1d(call);
  } else if (IsOp(call, "qnn.csi.clip")) {
    Clip(call);
  } else if (IsOp(call, "qnn.csi.concatenate")) {
    Concat(call);
  } else if (IsOp(call, "qnn.csi.conv1d")) {
    Conv1d(call);
  } else if (IsOp(call, "qnn.csi.conv2d")) {
    Conv2d(call, "conv2d");
  } else if (IsOp(call, "qnn.csi.conv2d_relu")) {
    Conv2d(call, "conv2d_relu");
  } else if (IsOp(call, "qnn.csi.conv2d_relu6")) {
    Conv2d(call, "conv2d_relu6");
  } else if (IsOp(call, "qnn.csi.conv3d")) {
    Conv3d(call);
  } else if (IsOp(call, "qnn.csi.cos")) {
    Unary(call, "cos");
  } else if (IsOp(call, "qnn.csi.cosh")) {
    Unary(call, "cosh");
  } else if (IsOp(call, "qnn.csi.crop_resize")) {
    CropResize(call);
  } else if (IsOp(call, "qnn.csi.deconv2d")) {
    DeConv2d(call);
  } else if (IsOp(call, "qnn.csi.deconv3d")) {
    DeConv3d(call);
  } else if (IsOp(call, "qnn.csi.dense")) {
    Dense(call);
  } else if (IsOp(call, "qnn.csi.depth_to_space")) {
    DepthToSpace(call);
  } else if (IsOp(call, "qnn.csi.dilation2d")) {
    Dilation2d(call);
  } else if (IsOp(call, "qnn.csi.div")) {
    DisoOp(call, "div");
  } else if (IsOp(call, "qnn.csi.equal")) {
    DisoOp(call, "equal", "bool");
  } else if (IsOp(call, "qnn.csi.erf")) {
    Unary(call, "erf");
  } else if (IsOp(call, "qnn.csi.exp")) {
    Unary(call, "exp");
  } else if (IsOp(call, "qnn.csi.expand_dims")) {
    ExpandDims(call);
  } else if (IsOp(call, "qnn.csi.flatten")) {
    Flatten(call);
  } else if (IsOp(call, "qnn.csi.floor")) {
    Unary(call, "floor");
  } else if (IsOp(call, "qnn.csi.floor_div")) {
    DisoOp(call, "floor_divide");
  } else if (IsOp(call, "qnn.csi.floor_mod")) {
    DisoOp(call, "floor_mod");
  } else if (IsOp(call, "qnn.csi.fsmn")) {
    Fsmn(call);
  } else if (IsOp(call, "qnn.csi.full")) {
    Full(call);
  } else if (IsOp(call, "qnn.csi.global_avgpool2d")) {
    GlobalAvgPool2d(call);
  } else if (IsOp(call, "qnn.csi.global_maxpool2d")) {
    GlobalMaxPool2d(call);
  } else if (IsOp(call, "qnn.csi.leaky_relu")) {
    LeakyRelu(call);
  } else if (IsOp(call, "qnn.csi.left_shift")) {
    DisoOp(call, "left_shift");
  } else if (IsOp(call, "qnn.csi.log")) {
    Unary(call, "log");
  } else if (IsOp(call, "qnn.csi.layer_norm")) {
    LayerNorm(call);
  } else if (IsOp(call, "qnn.csi.log_softmax")) {
    LogSoftmax(call);
  } else if (IsOp(call, "qnn.csi.lrn")) {
    LRN(call);
  } else if (IsOp(call, "qnn.csi.max")) {
    Reduce(call, "max", cfg->dtype_weight);
  } else if (IsOp(call, "qnn.csi.maxpool3d")) {
    MaxPool3d(call);
  } else if (IsOp(call, "qnn.csi.maximum")) {
    DisoOp(call, "maximum");
  } else if (IsOp(call, "qnn.csi.matmul")) {
    MatMul(call);
  } else if (IsOp(call, "qnn.csi.maxpool2d")) {
    MaxPool2d(call);
  } else if (IsOp(call, "qnn.csi.maxpool2d_locat")) {
    MaxPool2dLocat(call);
  } else if (IsOp(call, "qnn.csi.maxpool2d_with_argmax")) {
    Maxpool2dWithArgmax(call);
  } else if (IsOp(call, "qnn.csi.mean")) {
    Reduce(call, "mean", cfg->dtype_weight);
  } else if (IsOp(call, "qnn.csi.min")) {
    Reduce(call, "min", cfg->dtype_weight);
  } else if (IsOp(call, "qnn.csi.minimum")) {
    DisoOp(call, "minimum");
  } else if (IsOp(call, "qnn.csi.mod")) {
    DisoOp(call, "mod");
  } else if (IsOp(call, "qnn.csi.mul")) {
    DisoOp(call, "mul");
  } else if (IsOp(call, "qnn.csi.negative")) {
    Unary(call, "negative");
  } else if (IsOp(call, "qnn.csi.pad")) {
    Pad(call);
  } else if (IsOp(call, "qnn.csi.power")) {
    DisoOp(call, "power");
  } else if (IsOp(call, "qnn.csi.prelu")) {
    PRelu(call);
  } else if (IsOp(call, "qnn.csi.prod")) {
    Reduce(call, "prod", cfg->dtype_weight);
  } else if (IsOp(call, "qnn.csi.proposal")) {
    Proposal(call);
  } else if (IsOp(call, "qnn.csi.psroipooling")) {
    PSROIPool(call);
  } else if (IsOp(call, "qnn.csi.relu")) {
    Relu(call);
  } else if (IsOp(call, "qnn.csi.relu6")) {
    Relu6(call);
  } else if (IsOp(call, "qnn.csi.reshape")) {
    Reshape(call);
  } else if (IsOp(call, "qnn.csi.reverse")) {
    Reverse(call);
  } else if (IsOp(call, "qnn.csi.right_shift")) {
    DisoOp(call, "right_shift");
  } else if (IsOp(call, "qnn.csi.roipooling")) {
    ROIPool(call);
  } else if (IsOp(call, "qnn.csi.round")) {
    Unary(call, "round");
  } else if (IsOp(call, "qnn.csi.scatter_nd")) {
    ScatterND(call);
  } else if (IsOp(call, "qnn.csi.segment_max")) {
    Segment(call, "max");
  } else if (IsOp(call, "qnn.csi.segment_mean")) {
    Segment(call, "mean");
  } else if (IsOp(call, "qnn.csi.segment_min")) {
    Segment(call, "min");
  } else if (IsOp(call, "qnn.csi.segment_prod")) {
    Segment(call, "prob");
  } else if (IsOp(call, "qnn.csi.segment_sum")) {
    Segment(call, "sum");
  } else if (IsOp(call, "qnn.csi.sigmoid")) {
    Sigmoid(call);
  } else if (IsOp(call, "qnn.csi.sign")) {
    Unary(call, "sign");
  } else if (IsOp(call, "qnn.csi.sin")) {
    Unary(call, "sin");
  } else if (IsOp(call, "qnn.csi.sinh")) {
    Unary(call, "sinh");
  } else if (IsOp(call, "qnn.csi.softmax")) {
    Softmax(call);
  } else if (IsOp(call, "qnn.csi.space_to_batch_nd")) {
    SpaceToBatchND(call);
  } else if (IsOp(call, "qnn.csi.space_to_depth")) {
    SpaceToDepth(call);
  } else if (IsOp(call, "qnn.csi.split")) {
    Split(call);
  } else if (IsOp(call, "qnn.csi.sqrt")) {
    Unary(call, "sqrt");
  } else if (IsOp(call, "qnn.csi.rsqrt")) {
    Unary(call, "rsqrt");
  } else if (IsOp(call, "qnn.csi.squeeze")) {
    Squeeze(call);
  } else if (IsOp(call, "qnn.csi.strided_slice")) {
    StridedSlice(call);
  } else if (IsOp(call, "qnn.csi.subtract")) {
    DisoOp(call, "sub");
  } else if (IsOp(call, "qnn.csi.sum")) {
    Reduce(call, "sum", cfg->dtype_weight);
  } else if (IsOp(call, "qnn.csi.take")) {
    Take(call);
  } else if (IsOp(call, "qnn.csi.tan")) {
    Unary(call, "tan");
  } else if (IsOp(call, "qnn.csi.tanh")) {
    Unary(call, "tanh");
  } else if (IsOp(call, "qnn.csi.tile")) {
    Tile(call);
  } else if (IsOp(call, "qnn.csi.transpose")) {
    Transpose(call);
  } else if (IsOp(call, "qnn.csi.unpooling")) {
    UnPool2d(call);
  } else if (IsOp(call, "qnn.csi.upsampling")) {
    UpSampling(call);
  } else if (IsOp(call, "qnn.csi.less")) {
    DisoOp(call, "less");
  } else if (IsOp(call, "qnn.csi.one_hot")) {
    OneHot(call);
  } else if (IsOp(call, "qnn.csi.where")) {
    Where(call);
  } else if (IsOp(call, "qnn.csi.where_softmax")) {
    WhereSoftmax(call);
  } else if (IsOp(call, "qnn.csi.data_convert")) {
    DataConvert(call);
  } else {
    LOG(FATAL) << "Unsupported op: " << AsText(call->op, false);
  }
}

string CodegenCSINN::replace(string a) {
  std::string new_name = a;
  int pos;
  std::vector<char> illegal_str = {'.', '/', ':'};
  for (size_t i = 0; i < illegal_str.size(); i++) {
    pos = new_name.find(illegal_str[i]);
    while (pos != -1) {
      new_name.replace(pos, 1, "_");
      pos = new_name.find(illegal_str[i]);
    }
  }

  return new_name;
}

void CodegenCSINN::Axis0Cast(CSIConstant* data, CSIConstant* output, Qinfo* q_infos,
                             string target_dtype, int q_size, int inner_size) {
  float* input_data = reinterpret_cast<float*>(data->get_data_buf());
  if (target_dtype == "uint8_t") {
    uint8_t* out = reinterpret_cast<uint8_t*>(output->get_data_buf());
    for (int c = 0; c < q_size; c++) {
      for (int i = 0; i < inner_size; i++) {
        int index = c * inner_size + i;
        int32_t out_ = std::round(input_data[index] / q_infos[c].scale) + q_infos[c].zero_point;
        out_ = std::max(out_, 0);
        out_ = std::min(out_, 255);
        out[index] = out_;
      }
    }
  } else if (target_dtype == "int8_t") {
    int8_t* out = reinterpret_cast<int8_t*>(output->get_data_buf());
    for (int c = 0; c < q_size; c++) {
      for (int i = 0; i < inner_size; i++) {
        int index = c * inner_size + i;
        int32_t out_ = std::round(input_data[index] / q_infos[c].scale) + q_infos[c].zero_point;
        out_ = std::max(out_, -127);
        out_ = std::min(out_, 127);
        out[index] = out_;
      }
    }
  } else if (target_dtype == "int16_t") {
    int16_t* out = reinterpret_cast<int16_t*>(output->get_data_buf());
    for (int c = 0; c < q_size; c++) {
      for (int i = 0; i < inner_size; i++) {
        int index = c * inner_size + i;
        int32_t out_ = std::round(input_data[index] / q_infos[c].scale) + q_infos[c].zero_point;
        out_ = std::max(out_, -32768);
        out_ = std::min(out_, 32767);
        out[index] = out_;
      }
    }
  } else if (target_dtype == "int4_t") {
    int8_t* out = reinterpret_cast<int8_t*>(output->get_data_buf());
    for (int c = 0; c < q_size; c++) {
      for (int i = 0; i < inner_size; i++) {
        int index = c * inner_size + i;
        int32_t out_ = std::round(input_data[index] / q_infos[c].scale) + q_infos[c].zero_point;
        out_ = std::max(out_, -8);
        out_ = std::min(out_, 7);
        int out_index = c * ((inner_size + 1) / 2) + i / 2;
        /* int4 little endian */
        if (i % 2) {
          out[out_index] = (out[out_index] & 0xF) | (out_ << 4);
        } else {
          out[out_index] = (out[out_index] & 0xF0) | (out_ & 0xF);
        }
      }
    }
  } else {
    LOG(ERROR) << "get error dtype:" << target_dtype;
  }
}

void CodegenCSINN::Axis3Cast(CSIConstant* data, CSIConstant* output, Qinfo* q_infos,
                             string target_dtype, int q_size, int inner_size) {
  float* input_data = reinterpret_cast<float*>(data->get_data_buf());
  if (target_dtype == "uint8_t") {
    uint8_t* out = static_cast<uint8_t*>(output->get_data_buf());
    for (int i = 0; i < inner_size; i++) {
      for (int c = 0; c < q_size; c++) {
        int index = i * q_size + c;
        int32_t out_ = std::round(input_data[index] / q_infos[c].scale) + q_infos[c].zero_point;
        out_ = std::max(out_, 0);
        out_ = std::min(out_, 255);
        out[index] = out_;
      }
    }
  } else if (target_dtype == "int8_t") {
    int8_t* out = static_cast<int8_t*>(output->get_data_buf());
    for (int i = 0; i < inner_size; i++) {
      for (int c = 0; c < q_size; c++) {
        int index = i * q_size + c;
        int32_t out_ = std::round(input_data[index] / q_infos[c].scale) + q_infos[c].zero_point;
        out_ = std::max(out_, -127);
        out_ = std::min(out_, 127);
        out[index] = out_;
      }
    }
  } else if (target_dtype == "int16_t") {
    int16_t* out = static_cast<int16_t*>(output->get_data_buf());
    for (int i = 0; i < inner_size; i++) {
      for (int c = 0; c < q_size; c++) {
        int index = i * q_size + c;
        int32_t out_ = std::round(input_data[index] / q_infos[c].scale) + q_infos[c].zero_point;
        out_ = std::max(out_, -32768);
        out_ = std::min(out_, 32767);
        out[index] = out_;
      }
    }
  } else if (target_dtype == "int4_t") {
    int8_t* out = reinterpret_cast<int8_t*>(output->get_data_buf());
    for (int i = 0; i < inner_size; i++) {
      for (int c = 0; c < q_size; c++) {
        int index = i * q_size + c;
        int32_t out_ = std::round(input_data[index] / q_infos[c].scale) + q_infos[c].zero_point;
        out_ = std::max(out_, -8);
        out_ = std::min(out_, 7);
        int out_index = index / 2;
        /* int4 little endian */
        if (index % 2) {
          out[out_index] = (out[out_index] & 0xF) | (out_ << 4);
        } else {
          out[out_index] = (out[out_index] & 0xF0) | (out_ & 0xF);
        }
      }
    }
  } else {
    LOG(ERROR) << "get error dtype:" << target_dtype;
  }
}

// for per-axis (per-channel) quantize kernel
CSIConstant* CodegenCSINN::CastParams(CSIConstant* data, string target_dtype,
                                      QuantParams* quant_params, bool depthwise_kernel) {
  Qinfo* q_infos = quant_params->qinfo;
  int q_size = quant_params->q_size;

  CSIConstant* output = new CSIConstant(target_dtype, data->get_shape());
  if (data->get_dtype() == target_dtype || target_dtype == "float") {
    return data;
  } else {
    float* input_data = GetFloatData(data);
    output->set_name(data->get_name());

    int size = data->element_number();
    int inner_size = size / q_size;
    if (target_dtype == "int4_t") {
      // int4 only support NHWC
      if (depthwise_kernel) {
        Axis3Cast(data, output, q_infos, target_dtype, q_size, inner_size);
      } else {
        Axis0Cast(data, output, q_infos, target_dtype, q_size, inner_size);
      }
    } else if (target_dtype == "int8_t" || target_dtype == "uint8_t") {
      if ((layout_ == "NHWC") && depthwise_kernel) {
        Axis3Cast(data, output, q_infos, target_dtype, q_size, inner_size);
      } else {
        Axis0Cast(data, output, q_infos, target_dtype, q_size, inner_size);
      }
    } else if (target_dtype == "int16_t") {
      if ((layout_ == "NHWC") && depthwise_kernel) {
        Axis3Cast(data, output, q_infos, target_dtype, q_size, inner_size);
      } else {
        Axis0Cast(data, output, q_infos, target_dtype, q_size, inner_size);
      }
    } else if (target_dtype == "int32_t") {
      int32_t* out = static_cast<int32_t*>(output->get_data_buf());
      for (int i = 0; i < size; i++) {
        int32_t out_ = std::round(input_data[i] / q_infos->scale);
        out[i] = out_;
      }
    } else if (target_dtype == "float16") {
      int16_t* out = static_cast<int16_t*>(output->get_data_buf());
      for (int i = 0; i < size; i++) {
        int16_t out_ = float32_to_float16(input_data[i] / q_infos->scale);
        out[i] = out_;
      }
    } else if (target_dtype == "bfloat16") {
      int16_t* out = static_cast<int16_t*>(output->get_data_buf());
      for (int i = 0; i < size; i++) {
        int16_t out_ = float32_to_bfloat16(input_data[i]);
        out[i] = out_;
      }
    } else {
      LOG(ERROR) << "get error dtype:" << target_dtype;
    }
    free(input_data);
  }

  return output;
}

CSIConstant* CodegenCSINN::CastParams(CSIConstant* data, string target_dtype,
                                      QuantParams integral_input_quant,
                                      QuantParams kernel_quant_params) {
  if (data->get_dtype() == target_dtype) {
    return data;
  }
  float* input_data = reinterpret_cast<float*>(data->get_data_buf());
  int q_size = kernel_quant_params.q_size;
  Qinfo* qinfos = kernel_quant_params.qinfo;
  float iscale = integral_input_quant.qinfo->scale;

  CSIConstant* output = new CSIConstant(target_dtype, data->get_shape());
  output->set_name(data->get_name());
  if (target_dtype == "int32_t") {
    output->set_dtype("int32_t");
    int32_t* out = reinterpret_cast<int32_t*>(output->get_data_buf());
    int size = output->element_number();

    for (int i = 0; i < q_size; i++) {
      for (int j = 0; j < size / q_size; j++) {
        int index = i * (size / q_size) + j;
        float bscale = qinfos[i].scale * iscale;
        if (integral_input_quant.dtype == "int16_t" && kernel_quant_params.dtype == "int16_t" &&
            std::abs(bscale) < 1e-5) {
          bscale = 1e-5;
        }
        float out_ = std::round(input_data[index] / bscale);
        int int32_max = std::numeric_limits<int>::max();
        int int32_min = std::numeric_limits<int>::min();
        if (out_ > int32_max) {
          // LOG(WARNING) << "bias will overflow! Force changed wscale";
          out[index] = int32_max;
        } else if (out_ < int32_min) {
          out[index] = int32_min;
        } else {
          out[index] = out_;
        }
      }
    }
  } else if (target_dtype == "float") {
    float* out = reinterpret_cast<float*>(output->get_data_buf());
    memcpy(out, input_data, output->byte_size());
  } else if (target_dtype == "int16_t") {
    int16_t* out = reinterpret_cast<int16_t*>(output->get_data_buf());
    int size = output->element_number();
    for (int i = 0; i < q_size; i++) {
      for (int j = 0; j < size / q_size; j++) {
        int index = i * (size / q_size) + j;
        int32_t out_ = std::round(input_data[index] / (qinfos[i].scale * iscale));
        out_ = std::max(out_, -32768);
        out_ = std::min(out_, 32767);
        out[index] = out_;
      }
    }
  } else if (target_dtype == "float16") {
    int16_t* out = reinterpret_cast<int16_t*>(output->get_data_buf());
    int size = output->element_number();

    for (int i = 0; i < size / q_size; i++) {
      for (int c = 0; c < q_size; c++) {
        int index = i * q_size + c;

        float q_value = input_data[index] / qinfos[c].scale;
        int16_t out_ = float32_to_float16(q_value);
        out[index] = out_;
      }
    }
  } else if (target_dtype == "bfloat16") {
    int16_t* out = reinterpret_cast<int16_t*>(output->get_data_buf());
    for (uint i = 0; i < output->element_number(); i++) {
      int16_t out_ = float32_to_bfloat16(input_data[i]);
      out[i] = out_;
    }
  } else {
    LOG(ERROR) << "get error dtype:" << target_dtype;
  }
  return output;
}

void CodegenCSINN::EmitHeader(void) {
  std::ostringstream t0;
  func_def_.OneLine("#include <csi_nn.h>");
  func_def_.NewLine();
}

void CodegenCSINN::EmitVersion(void) {
  std::ostringstream t0;
  t0 << "/* auto generate by HHB_VERSION " << HHB_VERSION << " */";
  func_def_.OneLine(t0);
  func_def_.NewLine();
}

void CodegenCSINN::ModelBinarySave() {
  std::ostringstream t0;

  t0 << "sess->base_quant_type = " << cfg->quantization_scheme << ";";
  func_def_.OneLine(t0);

  if (model_save == "run_only") {
    t0 << "sess->model.save_mode = CSINN_RUN_ONLY;";
  } else if (model_save == "save_only") {
    t0 << "sess->model.save_mode = CSINN_SAVE_ONLY;";
  } else if (model_save == "save_and_run") {
    t0 << "sess->model.save_mode = CSINN_SAVE_AND_RUN;";
  } else {
    std::cerr << "Unsupport for model save_mode type: " << model_save << "\n";
    exit(-1);
  }
  func_def_.OneLine(t0);

  if (layout_ == "NCHW") {
    t0 << "sess->base_layout = CSINN_LAYOUT_NCHW;";
  } else if (layout_ == "NHWC") {
    t0 << "sess->base_layout = CSINN_LAYOUT_NHWC;";
  } else {
    std::cerr << "Unsupport layout type: " << layout_ << "\n";
    exit(-1);
  }
  func_def_.OneLine(t0);
}

void CodegenCSINN::EmitSessionSetup(void) {
  std::ostringstream t0;
  t0 << "void *csinn_(char *params_base) {";
  func_def_.OneLine(t0);
  func_def_.EnterScope();

  func_def_.OneLine("struct csinn_session *sess = csinn_alloc_session();");
  SessionRunMode();
  ModelBinarySave();
  t0 << "sess->base_api = " << target_name_ << ";";
  func_def_.OneLine(t0);
  t0 << "sess->base_dtype = " << base_dtype_ << ";";
  func_def_.OneLine(t0);
  t0 << "sess->dynamic_shape = " << (dynamic_shape_ ? "CSINN_TRUE" : "CSINN_FALSE") << ";";
  func_def_.OneLine(t0);
  if (debug_level_ == "INFO") {
    func_def_.OneLine("sess->debug_level = CSINN_DEBUG_LEVEL_INFO;");
  }
  if (std::find(trace_.begin(), trace_.end(), "csinn_acc") != trace_.end()) {
    func_def_.OneLine(
        "sess->profiler_level = CSINN_PROFILER_LEVEL_TRACE + CSINN_PROFILER_LEVEL_DUMP;");
  } else if (std::find(trace_.begin(), trace_.end(), "csinn") != trace_.end()) {
    func_def_.OneLine("sess->profiler_level = CSINN_PROFILER_LEVEL_TRACE;");
  }
  func_def_.OneLine("csinn_session_init(sess);");

  t0 << "csinn_set_input_number(" << ext_func_args_.size() << ", sess);";
  func_def_.OneLine(t0);
  t0 << "csinn_set_output_number(" << output_list_.size() << ", sess);";
  func_def_.OneLine(t0);

  func_def_.NewLine();
  for (uint32_t i = 0; i < ext_func_args_.size(); i++) {
    std::string in_name = CodegenCSINN::replace(ext_func_args_[i]->name_hint());
    std::ostringstream t1;
    t1 << "csinn_set_tensor_entry(" << in_name << ", sess)";
    func_def_.PushDecl(t1);
    t1 << "csinn_set_input(" << i << ", " << in_name << ", sess)";
    func_def_.PushDecl(t1);
  }

  func_def_.BufToCode();

  int output_index = 0;
  // emit normal outputs
  for (uint32_t i = 0; i < output_list_.size(); i++) {
    if (!output_list_[i].is_const) {
      string output_name = output_list_[i].name;
      t0 << "csinn_set_output(" << output_index++ << ", " << output_name << ", sess);";
      func_def_.OneLine(t0);
    }
  }

  // emit constant outputs
  for (uint32_t i = 0; i < output_list_.size(); i++) {
    if (output_list_[i].is_const) {
      t0 << output_list_[i].name << "->name = "
         << "\"" << output_list_[i].name << "\";";
      func_def_.OneLine(t0);
      t0 << output_list_[i].name << "->dtype = CSINN_DTYPE_FLOAT32;";
      func_def_.OneLine(t0);
      t0 << output_list_[i].name << "->is_const = 1;";
      func_def_.OneLine(t0);
      t0 << "csinn_set_output(" << output_index++ << ", " << output_list_[i].name << ", sess);";
      func_def_.OneLine(t0);
    }
  }

  func_def_.NewLine();
  func_def_.OneLine("csinn_session_setup(sess);");
  func_def_.OneLine("return sess;");
  func_def_.ExitScope();
  func_def_.OneLine("}");
}

void CodegenCSINN::EmitSessionRun(void) {
  std::ostringstream t0;
  t0 << "void csinn_update_input_and_run(";
  t0 << "struct csinn_tensor **input_tensors , void *sess) {";
  func_def_.OneLine(t0);
  func_def_.EnterScope();
  for (uint32_t i = 0; i < ext_func_args_.size(); i++) {
    t0 << "csinn_update_input(" << i << ", "
       << "input_tensors[" << i << "], sess);";
    func_def_.OneLine(t0);
  }
  if (show_session_run_time) {
    // init time params
    func_def_.OneLine("uint64_t start_time, end_time;");
    func_def_.OneLine("start_time = shl_get_timespec();");
    func_def_.OneLine("csinn_session_run(sess);");
    func_def_.OneLine("end_time = shl_get_timespec();");
    func_def_.OneLine(
        "printf(\"csinn_session_run execution time: %.5fms, FPS=%.5f\\n\", ((float)(end_time - "
        "start_time)) / 1000000,");
    func_def_.OneLine("       1000000000.0 / ((float)(end_time - start_time)));");
  } else {
    func_def_.OneLine("csinn_session_run(sess);");
  }

  func_def_.ExitScope();
  func_def_.OneLine("}");
}

string CodegenCSINN::get_ccode(void) {
  EmitVersion();
  EmitHeader();
  EmitSessionSetup();
  EmitSessionRun();
  DumpConstant();
  return func_def_.str();
}

void CodegenCSINN::SetConstDim(string name, std::vector<int> shape) {
  std::ostringstream t0;
  if (shape.size() == 0) {
    t0 << name << "->dim[" << 0 << "] = 1";
    func_def_.PushDecl(t0);
    t0 << name << "->dim_count = 1";
    func_def_.PushDecl(t0);
    return;
  }
  for (size_t i = 0; i < shape.size(); i++) {
    t0 << name << "->dim[" << i << "] = " << shape[i];
    func_def_.PushDecl(t0);
  }
  t0 << name << "->dim_count = " << shape.size();
  func_def_.PushDecl(t0);
}

void CodegenCSINN::SetDim(CSINNTensor* t, string name, std::vector<int> shape, bool dynamic_shape) {
  std::ostringstream t0;
  if (shape.size() == 0) {
    t->tensor->dim_count = 1;
    t->tensor->dim[0] = 1;
    return;
  }
  for (size_t i = 0; i < shape.size(); i++) {
    t->tensor->dim[i] = shape[i];
  }
  if (dynamic_shape) {
    t->tensor->dim_count = -1;
  } else {
    t->tensor->dim_count = shape.size();
  }
}

void CodegenCSINN::CreateGraphTensor(QuantParams q_params) {
  std::ostringstream t0;
  t0 << "struct csinn_tensor *" << q_params.name << " = csinn_alloc_tensor(sess);\n";
  for (uint32_t i = 0; i < q_params.shape.size(); i++) {
    t0 << "  " << q_params.name << "->dim[" << to_string(i)
       << "] = " << to_string(q_params.shape[i]) << ";\n";
  }
  t0 << "  " << q_params.name << "->dim_count = " << to_string(q_params.shape.size()) << ";\n";
  t0 << "  " << q_params.name << "->name = "
     << "\"" << q_params.name << "\""
     << ";\n";
  t0 << "  " << q_params.name << "->qinfo->zero_point = " << to_string(q_params.qinfo->zero_point)
     << ";\n";
  t0 << "  " << q_params.name << "->qinfo->scale = " << to_string(q_params.qinfo->scale) << ";\n";
  t0 << "  " << q_params.name << "->qinfo->min = " << to_string(q_params.qinfo->min) << ";\n";
  t0 << "  " << q_params.name << "->qinfo->max = " << to_string(q_params.qinfo->max) << ";\n";
  std::string io_dtype = GetCSINNDtype(q_params.dtype);
  t0 << "  " << q_params.name << "->dtype = " << io_dtype << ";\n";
  t0 << "  " << q_params.name << "->layout = " << GetCSINNActLayout(q_params.shape) << ";\n";
  func_def_.OneLine(t0);
}

CSINNConstantTensor* CodegenCSINN::CreateConstantTensorBase(string name, size_t size,
                                                            std::vector<int> shape,
                                                            string target_dtype, int32_t layout) {
  CSINNConstantTensor* tensor = new CSINNConstantTensor;
  tensor->name = name.c_str();
  SetDim(tensor, name, shape, false);
  tensor->tensor->dtype = GetCSINNTensorDtype(target_dtype);
  tensor->tensor->layout = layout;
  tensor->tensor->is_const = 1;

  constant_offset += size;
  return tensor;
}

void CodegenCSINN::CreateConstantTensor(CSINNOP* op, CSIConstant* data, string name,
                                        std::vector<int> shape, QuantParams* quant_params,
                                        bool depthwise_kernel, bool is_bias) {
  float* input_data = reinterpret_cast<float*>(data->get_data_buf());
  if (is_bias && shape.size() == 0 && std::abs(input_data[0]) < 1e-5) {
    // no bias
    std::ostringstream t0;
    t0 << "struct csinn_tensor *" << name << " = csinn_alloc_tensor(sess)";
    func_def_.PushDecl(t0);
    t0 << name << "->data = NULL";
    func_def_.PushDecl(t0);
    t0 << name << "->name = "
       << "\"" << name << "\"";
    func_def_.PushDecl(t0);
    t0 << name << "->is_const = 1";
    func_def_.PushDecl(t0);
    t0 << name << "->dim_count = 0";
    func_def_.PushDecl(t0);
  } else {
    std::ostringstream t0;
    int32_t constant_layout;
    if (depthwise_kernel) {
      if (layout_ == "NCHW") {
        constant_layout = CSINN_LAYOUT_O1HW;
      } else {
        constant_layout = CSINN_LAYOUT_1HWO;
      }
    } else {
      /* shape has at least 1 dim */
      if (shape.size() == 0) {
        shape.push_back(1);
      }
      constant_layout = GetCSINNTensorWeightLayout(shape);
    }
    quant_params->shape = shape;
    data->layout = constant_layout;
    CSIConstant* data_cast = CastParams(data, quant_params->dtype, quant_params, depthwise_kernel);
    CSINNConstantTensor* ret = CreateConstantTensorBase(
        name, data_cast->byte_size(), quant_params->shape, quant_params->dtype, constant_layout);
    ret->tensor->quant_channel = quant_params->q_size;
    ret->set_const(data_cast);
    ret->set_quant(*quant_params);

    qinfo_list_.push_back(*quant_params);
    op->push_constant(ret);
  }
}

void CodegenCSINN::CreateWeightTensor(CSINNOP* op, CSIConstant* data, string name,
                                      std::vector<int> shape, QuantParams* quant_params) {
  int32_t constant_layout;
  /* shape has at least 1 dim */
  if (shape.size() == 0) {
    shape.push_back(1);
  }
  constant_layout = GetCSINNTensorWeightLayout(shape);
  quant_params->shape = shape;
  data->layout = constant_layout;

  string target_dtype = quant_params->dtype;
  Qinfo* q_infos = quant_params->qinfo;
  int q_size = quant_params->q_size;

  CHECK(layout_ != "NHWC") << "expects layout = NCHW";

  CSIConstant* data_cast = new CSIConstant(target_dtype, data->get_shape());
  float* input_data = GetFloatData(data);
  data_cast->set_name(data->get_name());

  int size = data->element_number();
  int inner_size = size / q_size;

  if (target_dtype == "int8_t") {
    int8_t* out = reinterpret_cast<int8_t*>(data_cast->get_data_buf());
    for (int c = 0; c < q_size; c++) {
      for (int i = 0; i < inner_size; i++) {
        int index = c * inner_size + i;
        int valid_range = std::pow(2, 7) - 1;
        float abs_max = std::max(std::abs(q_infos[c].max), std::abs(q_infos[c].min));
        q_infos[c].scale = abs_max / valid_range;
        q_infos[c].zero_point = 0;
        int32_t out_ = std::round(input_data[index] / q_infos[c].scale) + q_infos[c].zero_point;
        out_ = std::max(out_, -127);
        out_ = std::min(out_, 127);
        out[index] = out_;
      }
    }
  } else {
    LOG(ERROR) << "CreateWeightTensor unsupport dtype:" << target_dtype;
  }
  free(input_data);

  CSINNConstantTensor* ret = CreateConstantTensorBase(
      name, data_cast->byte_size(), quant_params->shape, target_dtype, constant_layout);
  ret->tensor->quant_channel = quant_params->q_size;
  ret->set_const(data_cast);
  ret->set_quant(*quant_params);

  qinfo_list_.push_back(*quant_params);
  op->push_constant(ret);
}

void CodegenCSINN::CreateConstantTensor(CSINNOP* op, CSIConstant* data, string name,
                                        std::vector<int> shape, string target_dtype,
                                        QuantParams* input_quant_params,
                                        QuantParams* kernel_quant_params,
                                        QuantParams* bias_quant_params, QConfig_* quantize_cfg) {
  if (quantize_cfg == NULL) {
    quantize_cfg = cfg;
  }
  float* input_data = reinterpret_cast<float*>(data->get_data_buf());
  if (shape.size() == 0 && std::abs(input_data[0]) < 1e-5) {
    // no bias
    std::ostringstream t0;
    t0 << "struct csinn_tensor *" << name << " = csinn_alloc_tensor(sess)";
    func_def_.PushDecl(t0);
    t0 << name << "->data = NULL";
    func_def_.PushDecl(t0);
    t0 << name << "->name = "
       << "\"" << name << "\"";
    func_def_.PushDecl(t0);
    t0 << name << "->is_const = 1";
    func_def_.PushDecl(t0);
    t0 << name << "->dim_count = 0";
    func_def_.PushDecl(t0);
  } else {
    QuantParams* integral_input_quant =
        GetIntegralQuantParams(input_quant_params, ACTIVATE, quantize_cfg);

    CSIConstant* data_cast;
    if (quantize_cfg->quantization_scheme != "CSINN_QUANT_FLOAT16_W_INT8") {
      data_cast = CastParams(data, target_dtype, *integral_input_quant, *kernel_quant_params);
    } else {
      data_cast = CastParams(data, target_dtype, *integral_input_quant, *bias_quant_params);
    }
    std::ostringstream t0;
    int32_t layout = GetCSINNTensorWeightLayout(shape);
    CSINNConstantTensor* ret =
        CreateConstantTensorBase(name, data_cast->byte_size(), shape, target_dtype, layout);
    if (quantize_cfg->quantization_scheme != "CSINN_QUANT_FLOAT16_W_INT8") {
      for (int i = 0; i < bias_quant_params->q_size; i++) {
        float curr_scale = integral_input_quant->qinfo->scale * kernel_quant_params->qinfo[i].scale;
        if (integral_input_quant->dtype == "int16_t" && kernel_quant_params->dtype == "int16_t" &&
            target_dtype == "int32_t" && std::abs(curr_scale) < 1e-5) {
          curr_scale = 1e-5;
        }
        bias_quant_params->qinfo[i].scale = curr_scale;
        bias_quant_params->qinfo[i].zero_point = 0;
      }
    }

    ret->tensor->quant_channel = bias_quant_params->q_size;
    ret->set_const(data_cast);
    ret->set_quant(*bias_quant_params);
    op->push_constant(ret);
    bias_quant_params->name = name;

    qinfo_list_.push_back(*bias_quant_params);
  }
}

CSINNVarTensor* CodegenCSINN::CreateTensor(string name, string data, std::vector<int> shape,
                                           QuantParams quant_params, string dtype) {
  CSINNVarTensor* tensor = new CSINNVarTensor;
  tensor->name = name.c_str();
  SetDim(tensor, name, shape, dynamic_shape_);
  tensor->tensor->quant_channel = quant_params.q_size;
  tensor->tensor->dtype = GetCSINNTensorDtype(dtype);
  tensor->tensor->layout = GetCSINNTensorActLayout(shape);

  tensor->set_quant(quant_params);

  qinfo_list_.push_back(quant_params);
  return tensor;
}

output_element CodegenCSINN::GetRealInput(const CallNode* call) {
  output_element ret;
  ret.size = -1;
  for (auto out : out_list_) {
    if (out.call == call) {
      return out;
    }
  }
  return ret;
}

output_element CodegenCSINN::GetRealInput(const VarNode* var) {
  output_element ret;
  ret.size = -1;
  for (auto out : out_list_) {
    if (out.name == var->name_hint()) {
      return out;
    }
  }
  return ret;
}

void CodegenCSINN::PushInput(string name, const CallNode* call) {
  for (uint i = 0; i < out_list_.size(); i++) {
    if (out_list_[i].name == name) {
      out_list_[i].call = call;
    }
  }
}

string CodegenCSINN::InputTensorCall(CSINNOP* op, const CallNode* pre_call, int input_index,
                                     QuantParams quant_params, string dtype) {
  auto ishape = pre_call->get_shape();
  auto input = out_[0];

  if (input.call != pre_call) {
    input = GetRealInput(pre_call);
    CHECK_NE(input.size, -1);
  }

  if (input.need_copy == true) {
    return input.name;
  } else {
    string input_name = "input" + to_string(input_index) + "_" + to_string(buf_idx_);
    CSINNVarTensor* ret = CreateTensor(input_name, input.name, ishape, quant_params, dtype);
    op->push_input(ret);
    quant_params.name = input_name;
    return input_name;
  }
}

string CodegenCSINN::InputTensorVar(CSINNOP* op, const VarNode* pre_var, int input_index,
                                    QuantParams quant_params, string dtype) {
  auto ishape = pre_var->get_shape();
  auto input = out_[0];
  string var_name = replace(pre_var->name_hint());

  if (input.name != var_name) {
    input = GetRealInput(pre_var);
    CHECK_EQ(input.size, -1);
  }

  if (io_nodes.end() != io_nodes.find(var_name)) {
    CSINNVarTensor* ret = io_nodes[var_name].second;
    op->push_input(ret);
    return var_name;
  } else {
    string input_name = "input" + to_string(input_index) + "_" + to_string(buf_idx_);
    quant_params.name = input_name;
    quant_params.offset = constant_offset;
    quant_params.shape = ishape;
    CSINNVarTensor* ret = CreateTensor(var_name, "__" + input.name, ishape, quant_params, dtype);
    ret->org_name = var_name + "@@" + op->get_name() + "_" + to_string(buf_idx_);
    io_nodes[var_name] = std::make_pair(quant_params, ret);
    op->push_input(ret);
    return var_name;
  }
}

string CodegenCSINN::InputTensorTupleItem(const TupleGetItemNode* pre_call,
                                          QuantParams quant_params, string dtype) {
  auto input = out_[0];
  CHECK(pre_call);
  auto pre_tuple = pre_call->tuple.as<CallNode>();
  if (input.call != pre_tuple) {
    input = GetRealInput(pre_tuple);
    CHECK_NE(input.size, -1);
  }
  auto input_name = input.names[pre_call->index];
  quant_params.name = input_name;
  return input_name;
}

string CodegenCSINN::InputTensorName(CSINNOP* op, const CallNode* call, int input_index,
                                     QuantParams quant_params, string dtype) {
  string input_name;
  if (auto pre_call = call->args[input_index].as<CallNode>()) {
    input_name = InputTensorCall(op, pre_call, input_index, quant_params, dtype);
  } else if (auto pre_var = call->args[input_index].as<VarNode>()) {
    input_name = InputTensorVar(op, pre_var, input_index, quant_params, dtype);
  } else {
    auto pre_call = call->args[input_index].as<TupleGetItemNode>();
    CHECK(pre_call);
    input_name = InputTensorTupleItem(pre_call, quant_params, dtype);
  }
  return input_name;
}

string CodegenCSINN::CreateInputTensor(CSINNOP* op, std::ostringstream& decl, const CallNode* call,
                                       int input_index, QuantParams* quant_params) {
  string input_name = InputTensorName(op, call, input_index, *quant_params, quant_params->dtype);
  decl << input_name;
  return input_name;
}

void CodegenCSINN::setup_callback(std::ostringstream& decl, string op_name, string params_name) {
  std::ostringstream t0;
  t0 << "csinn_" << op_name << "_init" << decl.str();
  func_def_.PushDecl(t0);
}

void CodegenCSINN::params_common_setup(std::ostringstream& decl, const CallNode* call,
                                       string op_name, string params_name, string layer_name,
                                       string layout) {
  std::ostringstream t0;
  if (!(layout_ == "NCHW" && layout == "CSINN_LAYOUT_NCHW")) {
    t0 << params_name << "->base.layout = CSINN_LAYOUT_" << layout_;
    func_def_.PushDecl(t0);
  }

  string complete_name = get_complete_layer_name(op_name, layer_name);
  t0 << params_name << "->base.name = "
     << "\"" << complete_name << "\"";
  params_idx_++;
  func_def_.PushDecl(t0);

  auto curr_cfg = call->get_quant_config();
  if (curr_cfg->quantization_scheme != "unset" &&
      curr_cfg->quantization_scheme != "CSINN_QUANT_INT4_ASYM_W_SYM" &&
      curr_cfg->quantization_scheme != "CSINN_QUANT_INT8_ASYM_W_SYM") {
    t0 << params_name << "->base.quant_type = " << curr_cfg->quantization_scheme;
    func_def_.PushDecl(t0);
  }

  setup_callback(decl, op_name, params_name);
}

string CodegenCSINN::CreateOutputTensor(CSINNOP* op, std::ostringstream& decl, const CallNode* call,
                                        QuantParams* quant_params) {
  auto out_shape = call->get_shape();
  // if output is a single number, out_shape.size() here is zero
  if (out_shape.size() == 0) {
    out_shape.push_back(1);
  }
  string output_name = "output_" + to_string(buf_idx_);
  quant_params->name = output_name;
  quant_params->offset = constant_offset;
  quant_params->shape = out_shape;
  CSINNVarTensor* ret =
      CreateTensor(output_name, "alloc", out_shape, *quant_params, quant_params->dtype);
  op->push_output(ret);
  decl << ", " << output_name;
  int out_index = CheckOutput(call);
  if (out_index > -1) {
    ret->org_name = op->get_name() + "_" + to_string(buf_idx_);
    io_nodes[output_name] = std::make_pair(*quant_params, ret);
  }
  return output_name;
}

void CodegenCSINN::DumpConstant() { bm_graph.dump_params(params_path_); }

void CodegenCSINN::DumpGraphInfo() { bm_graph.dump_graph_info(graph_info_path_); }

int CodegenCSINN::CheckOutput(const CallNode* call) {
  for (uint i = 0; i < output_list_.size(); i++) {
    if (output_list_[i].call == call) {
      return i;
    }
  }
  return -1;
}

void CodegenCSINN::PushOutput(string name, const CallNode* call, string dtype) {
  if (dtype == "") {
    dtype = cfg->dtype_weight;
  }

  auto out_shape = call->get_shape();
  int out_size = 1;
  for (size_t i = 0; i < out_shape.size(); ++i) {
    out_size *= out_shape[i];
  }

  out_.clear();
  output_element output;
  output.dtype = dtype;
  output.name = name;
  output.size = out_size;
  output.need_copy = true;
  output.call = call;
  output.shape = out_shape;
  output.index = layer_index_;
  output.is_const = false;
  layer_index_++;
  out_.push_back(output);
  out_list_.push_back(output);
  int out_index = CheckOutput(call);
  if (out_index > -1) {
    auto& out = output_list_[out_index];
    out = output;
  }
}

void CodegenCSINN::PushOutput(std::vector<string> names, const CallNode* call) {
  const auto& dtype = "uint8_t";

  out_.clear();
  output_element output;
  output.dtype = dtype;
  output.need_copy = true;
  output.call = call;
  output.index = layer_index_;
  output.is_const = false;
  output.names = names;
  layer_index_++;
  out_.push_back(output);
  out_list_.push_back(output);
}

template <typename T>
void CodegenCSINN::SisoOp(CSINNOP* op, std::ostringstream& decl, const CallNode* call,
                          const T* attr, string op_name) {
  CHECK(call->args.size() == 1) << "op expects 1 args";

  // Make function call with input buffers when visiting arguments
  decl << "(";

  /* Emit input tensor */
  visit(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string complete_name = get_complete_layer_name(op_name, attr->layer_name.c_str());
  op->set_name(complete_name);

  auto input_qinfo = GetCallOPQuant(call, 0);
  CreateInputTensor(op, decl, call, 0, input_qinfo);
  /* Emit output tensor */
  auto output_qinfo = GetCallOPQuant(call, 1);
  string output_name = CreateOutputTensor(op, decl, call, output_qinfo);

  collect_quant_info(complete_name, attr->q_params, 1);

  output2params[output_name] = complete_name;
  PushOutput(output_name, call, output_qinfo->dtype);
}

void CodegenCSINN::malloc_params(string struct_name, string params_name) {
  std::ostringstream t0;
  t0 << "struct " << struct_name << " *" << params_name << " = csinn_alloc_params(sizeof(struct "
     << struct_name << "), sess)";
  func_def_.PushDecl(t0);
}

void CodegenCSINN::Unary(const CallNode* call, string op_name) {
  std::ostringstream decl;
  CSINNOP* op = new CSINNOP;
  const auto* attr = call->attrs.as<QnnCSIUnaryAttrs>();
  SisoOp<QnnCSIUnaryAttrs>(op, decl, call, attr, op_name);

  push_decl(op);
  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";

  malloc_params("csinn_siso_params", params_name);

  params_common_setup(decl, call, op_name, params_name, attr->layer_name.c_str());
  end_stream(decl, op_name);
}

void CodegenCSINN::Cast(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  CSINNOP* op = new CSINNOP;
  const auto* attr = call->attrs.as<QnnCSIUnaryAttrs>();

  CHECK(call->args.size() == 1) << "op expects 1 args";

  // Make function call with input buffers when visiting arguments
  decl << "(";

  /* Emit input tensor */
  visit(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string complete_name = get_complete_layer_name("cast", attr->layer_name.c_str());
  op->set_name(complete_name);

  auto input_qinfo = GetCallOPQuant(call, 0);
  CreateInputTensor(op, decl, call, 0, input_qinfo);
  /* Emit output tensor */
  auto output_qinfo = GetCallOPQuant(call, 1);

  string target_dtype = GetDtypeString(attr->out_dtype);
  if (target_dtype == "int32_t") {
    output_qinfo->qinfo->scale = 1.0;
    output_qinfo->qinfo->zero_point = 0;
    output_qinfo->dtype = target_dtype;
  }

  string output_name = CreateOutputTensor(op, decl, call, output_qinfo);

  collect_quant_info(complete_name, attr->q_params, 1);

  output2params[output_name] = complete_name;
  PushOutput(output_name, call, output_qinfo->dtype);

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";
  push_decl(op);
  malloc_params("csinn_cast_params", params_name);

  t0 << params_name << "->dtype = " << GetCSINNDtype(target_dtype);
  func_def_.PushDecl(t0);

  params_common_setup(decl, call, "cast", params_name, attr->layer_name.c_str());
  end_stream(decl, "cast");
}

void CodegenCSINN::DisoOp(const CallNode* call, string op_name, string out_dtype) {
  std::ostringstream decl;
  std::ostringstream t0;
  CSINNOP* op = new CSINNOP;
  const auto* attr = call->attrs.as<QnnBinaryOpAttrs>();
  CHECK(attr);

  CHECK(call->args.size() == 2) << "op expects 2 args";

  // Make function call with input buffers when visiting arguments
  decl << "(";

  string lhs_name, rhs_name;

  string complete_name = get_complete_layer_name(op_name, attr->layer_name.c_str());
  op->set_name(complete_name);

  /* Emit input0 tensor */
  visit(call->args[0]);

  auto lhs_qinfo = GetCallOPQuant(call, 0);
  auto str_lhs_dtype = GetDtypeString(call->args[0]->hhb_expr_extend_->dtype);

  if (str_lhs_dtype == "int64") {
    lhs_qinfo->dtype = "int64_t";
    lhs_qinfo->qinfo->scale = 1;
    lhs_qinfo->qinfo->zero_point = 0;
    lhs_qinfo->q_size = 1;
  }

  if (call->args[0].as<ConstantNode>()) {
    CHECK(constant_.size() == 1) << "Every args expects a single constant_";
    auto lhs = constant_[0];
    auto lhs_shape = call->args[0]->get_shape();
    lhs_name = "lhs_" + to_string(buf_idx_);
    buf_idx_++;
    CreateConstantTensor(op, lhs, lhs_name, lhs_shape, lhs_qinfo);
    decl << lhs_name;
  } else {
    lhs_name = CreateInputTensor(op, decl, call, 0, lhs_qinfo);
    buf_idx_++;
  }

  decl << ", ";

  /* Emit input1 tensor */
  auto rhs_qinfo = GetCallOPQuant(call, 1);

  auto str_rhs_dtype = GetDtypeString(call->args[1]->hhb_expr_extend_->dtype);
  if (str_rhs_dtype == "int64") {
    rhs_qinfo->dtype = "int64_t";
    rhs_qinfo->qinfo->scale = 1;
    rhs_qinfo->qinfo->zero_point = 0;
    rhs_qinfo->q_size = 1;
  }

  if (call->args[1].as<ConstantNode>()) {
    // add constant arg
    visit(call->args[1]);
    CHECK(constant_.size() == 1) << "Every args expects a single constant_";
    auto rhs = constant_[0];
    auto rhs_shape = call->args[1]->get_shape();
    rhs_name = "rhs_" + to_string(buf_idx_);
    CreateConstantTensor(op, rhs, rhs_name, rhs_shape, rhs_qinfo);
    decl << rhs_name;
  } else {
    visit(call->args[1]);
    CHECK(out_.size() == 1) << "Every args expects a single out_";
    rhs_name = CreateInputTensor(op, decl, call, 1, rhs_qinfo);
  }

  auto output_qinfo = GetCallOPQuant(call, 2);
  if (str_lhs_dtype == "int64_t" || str_rhs_dtype == "int64_t") {
    output_qinfo->dtype = "int64_t";
    output_qinfo->qinfo->scale = 1;
    output_qinfo->qinfo->zero_point = 0;
    output_qinfo->q_size = 1;
  }

  if (out_dtype == "bool") {
    output_qinfo->dtype = out_dtype;
    output_qinfo->qinfo->scale = 1;
    output_qinfo->qinfo->zero_point = 0;
    output_qinfo->q_size = 1;
  }

  /* Emit output tensor */
  string output_name = CreateOutputTensor(op, decl, call, output_qinfo);

  collect_quant_info(complete_name, attr->q_params, 2);

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";
  push_decl(op);
  malloc_params("csinn_diso_params", params_name);

  output2params[output_name] = complete_name;

  PushOutput(output_name, call, output_qinfo->dtype);

  buf_idx_++;
  params_common_setup(decl, call, op_name, params_name, attr->layer_name.c_str());
  end_stream(decl, op_name);
}

template <typename T>
void CodegenCSINN::SetupPadding(string name, const T* attr) {
  Array<IndexExpr> pad = attr->padding;
  std::ostringstream t0;
  if (pad.size() == 4) {
    t0 << name << "->pad_top = " << to_string(pad[0].as<IntImmNode>()->value);
    func_def_.PushDecl(t0);
    t0 << name << "->pad_left = " << to_string(pad[1].as<IntImmNode>()->value);
    func_def_.PushDecl(t0);
    t0 << name << "->pad_down = " << to_string(pad[2].as<IntImmNode>()->value);
    func_def_.PushDecl(t0);
    t0 << name << "->pad_right = " << to_string(pad[3].as<IntImmNode>()->value);
    func_def_.PushDecl(t0);
  } else if (pad.size() == 6) {
    t0 << name << "->pad_front = " << to_string(pad[0].as<IntImmNode>()->value);
    func_def_.PushDecl(t0);
    t0 << name << "->pad_top = " << to_string(pad[1].as<IntImmNode>()->value);
    func_def_.PushDecl(t0);
    t0 << name << "->pad_left = " << to_string(pad[2].as<IntImmNode>()->value);
    func_def_.PushDecl(t0);
    t0 << name << "->pad_back = " << to_string(pad[3].as<IntImmNode>()->value);
    func_def_.PushDecl(t0);
    t0 << name << "->pad_down = " << to_string(pad[4].as<IntImmNode>()->value);
    func_def_.PushDecl(t0);
    t0 << name << "->pad_right = " << to_string(pad[5].as<IntImmNode>()->value);
    func_def_.PushDecl(t0);
  } else {
    CHECK_EQ(pad.size(), 2);
    t0 << name << "->pad_top = " << to_string(pad[0].as<IntImmNode>()->value);
    func_def_.PushDecl(t0);
    t0 << name << "->pad_left = " << to_string(pad[1].as<IntImmNode>()->value);
    func_def_.PushDecl(t0);
    t0 << name << "->pad_down = " << to_string(pad[0].as<IntImmNode>()->value);
    func_def_.PushDecl(t0);
    t0 << name << "->pad_right = " << to_string(pad[1].as<IntImmNode>()->value);
    func_def_.PushDecl(t0);
  }
}
template <typename T>
void CodegenCSINN::Setup1dPadding(string name, const T* attr) {
  Array<IndexExpr> pad = attr->padding;
  std::ostringstream t0;
  if (pad.size() == 2) {
    t0 << name << "->pad_left = " << to_string(pad[0].as<IntImmNode>()->value);
    func_def_.PushDecl(t0);
    t0 << name << "->pad_right = " << to_string(pad[1].as<IntImmNode>()->value);
    func_def_.PushDecl(t0);
  } else {
    CHECK_EQ(pad.size(), 1);
    t0 << name << "->pad_left = " << to_string(pad[0].as<IntImmNode>()->value);
    func_def_.PushDecl(t0);
    t0 << name << "->pad_right = " << to_string(pad[0].as<IntImmNode>()->value);
    func_def_.PushDecl(t0);
  }
}

template <typename T>
void CodegenCSINN::SetupConv2dParams(string name, const T* attr) {
  std::ostringstream t0;
  malloc_params("csinn_conv2d_params", name);
  t0 << name << "->group = " << to_string(attr->groups);
  func_def_.PushDecl(t0);
  Array<IndexExpr> strides = attr->strides;
  t0 << name << "->stride_height = " << to_string(strides[0].as<IntImmNode>()->value);
  func_def_.PushDecl(t0);
  t0 << name << "->stride_width = " << to_string(strides[1].as<IntImmNode>()->value);
  func_def_.PushDecl(t0);
  Array<IndexExpr> dilation = attr->dilation;
  t0 << name << "->dilation_height = " << to_string(dilation[0].as<IntImmNode>()->value);
  func_def_.PushDecl(t0);
  t0 << name << "->dilation_width = " << to_string(dilation[1].as<IntImmNode>()->value);
  func_def_.PushDecl(t0);
  t0 << name << "->conv_extra.kernel_tm = NULL";
  func_def_.PushDecl(t0);
  t0 << name << "->conv_extra.conv_mode = CSINN_DIRECT";
  func_def_.PushDecl(t0);
  SetupPadding(name, attr);
}

template <typename T>
void CodegenCSINN::SetupDilation2dParams(string name, const T* attr) {
  std::ostringstream t0;
  malloc_params("csinn_dilation2d_params", name);
  Array<IndexExpr> strides = attr->strides;
  t0 << name << "->stride_height = " << to_string(strides[0].as<IntImmNode>()->value);
  func_def_.PushDecl(t0);
  t0 << name << "->stride_width = " << to_string(strides[1].as<IntImmNode>()->value);
  func_def_.PushDecl(t0);
  Array<IndexExpr> dilation = attr->dilations;
  t0 << name << "->dilation_height = " << to_string(dilation[0].as<IntImmNode>()->value);
  func_def_.PushDecl(t0);
  t0 << name << "->dilation_width = " << to_string(dilation[1].as<IntImmNode>()->value);
  func_def_.PushDecl(t0);
  SetupPadding(name, attr);
}

template <typename T>
void CodegenCSINN::SetupConv3dParams(string name, const T* attr) {
  std::ostringstream t0;
  malloc_params("csinn_conv3d_params", name);
  t0 << name << "->group = " << to_string(attr->groups);
  func_def_.PushDecl(t0);
  Array<IndexExpr> strides = attr->strides;
  t0 << name << "->stride_depth = " << to_string(strides[0].as<IntImmNode>()->value);
  func_def_.PushDecl(t0);
  t0 << name << "->stride_height = " << to_string(strides[1].as<IntImmNode>()->value);
  func_def_.PushDecl(t0);
  t0 << name << "->stride_width = " << to_string(strides[2].as<IntImmNode>()->value);
  func_def_.PushDecl(t0);
  Array<IndexExpr> dilation = attr->dilation;
  t0 << name << "->dilation_depth = " << to_string(dilation[0].as<IntImmNode>()->value);
  func_def_.PushDecl(t0);
  t0 << name << "->dilation_height = " << to_string(dilation[1].as<IntImmNode>()->value);
  func_def_.PushDecl(t0);
  t0 << name << "->dilation_width = " << to_string(dilation[2].as<IntImmNode>()->value);
  func_def_.PushDecl(t0);
  SetupPadding(name, attr);
}

template <typename T>
void CodegenCSINN::SetupConv1dParams(string name, const T* attr) {
  std::ostringstream t0;
  malloc_params("csinn_conv1d_params", name);
  t0 << name << "->group = " << to_string(attr->groups);
  func_def_.PushDecl(t0);
  Array<IndexExpr> strides = attr->strides;
  t0 << name << "->stride_width = " << to_string(strides[0].as<IntImmNode>()->value);
  func_def_.PushDecl(t0);
  Array<IndexExpr> dilation = attr->dilation;
  t0 << name << "->dilation_width = " << to_string(dilation[0].as<IntImmNode>()->value);
  func_def_.PushDecl(t0);
  Setup1dPadding<T>(name, attr);
}

template <typename T>
void CodegenCSINN::SetupPoolParams(string name, const T* attr) {
  std::ostringstream t0;
  malloc_params("csinn_pool_params", name);
  Array<IndexExpr> strides = attr->strides;
  t0 << name << "->stride_height = " << to_string(strides[0].as<IntImmNode>()->value);
  func_def_.PushDecl(t0);
  t0 << name << "->stride_width = " << to_string(strides[1].as<IntImmNode>()->value);
  func_def_.PushDecl(t0);
  Array<IndexExpr> pool_size = attr->pool_size;
  t0 << name << "->filter_height = " << to_string(pool_size[0].as<IntImmNode>()->value);
  func_def_.PushDecl(t0);
  t0 << name << "->filter_width = " << to_string(pool_size[1].as<IntImmNode>()->value);
  func_def_.PushDecl(t0);
  auto ceil_mode = attr->ceil_mode;
  t0 << name << "->ceil_mode = " << to_string(ceil_mode);
  func_def_.PushDecl(t0);
  SetupPadding(name, attr);
}

template <typename T>
void CodegenCSINN::SetupPool3DParams(string name, const T* attr) {
  std::ostringstream t0;
  malloc_params("csinn_pool_params", name);
  Array<IndexExpr> strides = attr->strides;
  t0 << name << "->stride_depth = " << to_string(strides[0].as<IntImmNode>()->value);
  func_def_.PushDecl(t0);
  t0 << name << "->stride_height = " << to_string(strides[1].as<IntImmNode>()->value);
  func_def_.PushDecl(t0);
  t0 << name << "->stride_width = " << to_string(strides[2].as<IntImmNode>()->value);
  func_def_.PushDecl(t0);
  Array<IndexExpr> pool_size = attr->pool_size;
  t0 << name << "->filter_depth = " << to_string(pool_size[0].as<IntImmNode>()->value);
  func_def_.PushDecl(t0);
  t0 << name << "->filter_height = " << to_string(pool_size[1].as<IntImmNode>()->value);
  func_def_.PushDecl(t0);
  t0 << name << "->filter_width = " << to_string(pool_size[2].as<IntImmNode>()->value);
  func_def_.PushDecl(t0);
  SetupPadding(name, attr);
}

std::shared_ptr<std::vector<int32_t>> CodegenCSINN::FuseZpToBias(CSIConstant* data, CSINNOP* op,
                                                                 const CallNode* call,
                                                                 QuantParams* q_params,
                                                                 bool is_depthwise) {
  if (q_params[0].q_size > 1) {
    LOG(ERROR) << "only support fuse zp to bais in int8_asym_w_sym mode!";
  }
  int8_t* weight_data = reinterpret_cast<int8_t*>(op->get_constant(0)->const_data);

  auto in_params = q_params[0];

  CSIConstant* bias_data_cast = CastParams(data, "int32_t", in_params, q_params[1]);
  int32_t* bias_data = reinterpret_cast<int32_t*>(bias_data_cast->get_data_buf());

  auto b_shape = call->args[2]->get_shape();
  auto w_shape = call->args[1]->get_shape();
  int b_length = b_shape.size() ? b_shape[0] : w_shape[0];
  int32_t in_zp = q_params[0].qinfo->zero_point;
  auto out = std::make_shared<std::vector<int32_t>>();

  if (layout_ == "NHWC" && is_depthwise) {
    int outer_size = 1;
    for (uint i = 0; i < w_shape.size() - 1; i++) {
      outer_size *= w_shape[i];
    }
    for (int i = 0; i < b_length; i++) {
      int32_t new_b = b_shape.size() ? bias_data[i] : 0;
      for (int j = 0; j < outer_size; j++) {
        int w_index = b_length * j + i;
        new_b -= (int32_t)weight_data[w_index] * in_zp;
      }
      out->push_back(new_b);
    }
  } else {
    int inner_size = 1;
    for (uint i = 1; i < w_shape.size(); i++) {
      inner_size *= w_shape[i];
    }
    for (int i = 0; i < b_length; i++) {
      float new_b = b_shape.size() ? bias_data[i] : 0.0;
      for (int j = 0; j < inner_size; j++) {
        int w_index = i * inner_size + j;
        new_b -= (int32_t)weight_data[w_index] * in_zp;
      }
      out->push_back(new_b);
    }
  }

  free(bias_data);
  return out;
}

void CodegenCSINN::Conv1d(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream buf;
  CSINNOP* op = new CSINNOP;
  const auto* attr = call->attrs.as<QnnCSIConv1DAttrs>();
  CHECK(attr);

  CHECK(call->args.size() == 3) << "Conv1d expects 3 args";

  /* Make function call with arguments start */
  decl << "(";

  /* Emit_ input tensor */
  visit(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string complete_name = get_complete_layer_name("conv1d", attr->layer_name.c_str());
  op->set_name(complete_name);
  auto input_qinfo = GetCallOPQuant(call, 0);
  string input_name = CreateInputTensor(op, decl, call, 0, input_qinfo);

  /* Emit output tensor */
  auto output_qinfo = GetCallOPQuant(call, 3);
  string output_name = CreateOutputTensor(op, decl, call, output_qinfo);

  collect_quant_info(complete_name, attr->q_params, 3);

  /* Emit kernel tensor */
  visit(call->args[1]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto kernel = constant_[0];
  auto wshape = call->args[1]->get_shape();
  string kernel_name = "kernel_" + to_string(buf_idx_);
  auto kernel_qinfo = GetCallOPQuant(call, 1);
  if (cfg->quantization_scheme == "CSINN_QUANT_FLOAT16_W_INT8") {
    kernel_qinfo->dtype = "int8_t";
    CreateWeightTensor(op, kernel, kernel_name, wshape, kernel_qinfo);
  } else {
    CreateConstantTensor(op, kernel, kernel_name, wshape, kernel_qinfo);
  }
  decl << ", " << kernel_name;

  /* Emit bias tensor */
  visit(call->args[2]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto bias = constant_[0];
  auto bshape = call->args[2]->get_shape();
  string bias_name = "bias_" + to_string(buf_idx_);
  bool fuse_zp = false;
  CreateBiasTensor(op, call, bias, bias_name, attr->q_params, &fuse_zp, "conv_bias");
  decl << ", " << bias_name;

  string params_name = "params_" + to_string(buf_idx_);

  decl << ", " << params_name << ")";
  push_decl(op);
  SetupConv1dParams<QnnCSIConv1DAttrs>(params_name, attr);

  PushOutput(output_name, call);
  params_common_setup(decl, call, "conv1d", params_name, attr->layer_name.c_str());
  end_stream(decl, "conv1d");
}

void CodegenCSINN::CreateBiasTensor(CSINNOP* op, const CallNode* call, CSIConstant* data,
                                    string name, Array<Array<IndexExpr>> q_params, bool* fuse_zp,
                                    string const_kind) {
  bool depthwise_kernel = const_kind == "depthwise_bias" ? true : false;

  std::shared_ptr<std::vector<int32_t>> new_bias;
  auto call_cfg = call->get_quant_config();
  if ((call_cfg->quantization_scheme == "CSINN_QUANT_INT8_ASYM_W_SYM" ||
       call_cfg->quantization_scheme == "CSINN_QUANT_INT4_ASYM_W_SYM") &&
      call_cfg->fuse_zp2bias) {
    *fuse_zp = true;
    if (depthwise_kernel) {
      const_kind = "input;depthwise_kernel;depthwise_bias;out";
    } else {
      const_kind = "input;conv_kernel;conv_bias;out";
    }
    QuantParams* base_q_params = GetQuantParams(q_params, call_cfg, const_kind);
    new_bias = FuseZpToBias(data, op, call, base_q_params, depthwise_kernel);
  }

  auto bshape = call->args[2]->get_shape();
  if (*fuse_zp) {
    if (bshape.size() == 0) {
      data->realloc_data_buf(new_bias->size() * 4);
      bshape.push_back(new_bias->size());
    }
    int32_t* data_buf = static_cast<int32_t*>(data->get_data_buf());
    std::copy(new_bias->begin(), new_bias->end(), data_buf);
    data->set_dtype("int32_t");
  }

  QuantParams* in_q_params = GetCallOPQuant(call, 0);
  QuantParams* weight_q_params = GetCallOPQuant(call, 1);
  QuantParams* bias_q_params = GetCallOPQuant(call, 2);

  string input_dtype = call_cfg->dtype_weight;
  string weight_dtype = call_cfg->dtype_weight;
  string bias_dtype = call_cfg->dtype_activation;

  // if (input_dtype == "int16_t" && weight_dtype == "int16_t" && bias_dtype == "int32_t") {
  //   CreateConstantTensor(op, data, name, bshape, bias_q_params, false, true);
  // } else {
  CreateConstantTensor(op, data, name, bshape, bias_dtype, in_q_params, weight_q_params,
                       bias_q_params, call_cfg);
  // }
}

void CodegenCSINN::Conv2d(const CallNode* call, string op_name) {
  std::ostringstream decl;
  const auto* attr = call->attrs.as<QnnCSIConv2DAttrs>();
  CHECK(attr);

  CHECK(call->args.size() == 3) << "Conv2d expects 3 args";

  CSINNOP* op = new CSINNOP;

  /* Make function call with arguments start */
  decl << "(";

  // check for depthwise
  auto ishape = call->args[0]->get_shape();
  auto wshape = call->args[1]->get_shape();
  auto bshape = call->args[2]->get_shape();

  bool depthwise_kernel = is_depthwise(ishape, wshape, attr->groups, layout_);

  /* Emit_ input tensor */
  visit(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";

  string params_name = "params_" + to_string(buf_idx_);
  string complete_name = get_complete_layer_name(op_name, attr->layer_name.c_str());
  op->set_name(complete_name);

  auto input_qinfo = GetCallOPQuant(call, 0);
  string input_name = CreateInputTensor(op, decl, call, 0, input_qinfo);

  /* Emit output tensor */
  auto output_qinfo = GetCallOPQuant(call, 3);
  string output_name = CreateOutputTensor(op, decl, call, output_qinfo);

  collect_quant_info(complete_name, attr->q_params, 3);

  /* Emit kernel tensor */
  visit(call->args[1]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";

  auto kernel = constant_[0];
  string kernel_name = "kernel_" + to_string(buf_idx_);

  auto kernel_qinfo = GetCallOPQuant(call, 1);
  if (cfg->quantization_scheme == "CSINN_QUANT_FLOAT16_W_INT8") {
    kernel_qinfo->dtype = "int8_t";
    CreateWeightTensor(op, kernel, kernel_name, wshape, kernel_qinfo);
  } else {
    CreateConstantTensor(op, kernel, kernel_name, wshape, kernel_qinfo, depthwise_kernel, false);
  }

  decl << ", " << kernel_name;

  /* Emit bias tensor */
  visit(call->args[2]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";

  auto bias = constant_[0];
  string bias_name = "bias_" + to_string(buf_idx_);
  bool fuse_zp = false;

  CreateBiasTensor(op, call, bias, bias_name, attr->q_params, &fuse_zp,
                   depthwise_kernel ? "depthwise_bias" : "conv_bias");
  decl << ", " << bias_name;

  output2params[output_name] = complete_name;

  decl << ", " << params_name << ")";
  push_decl(op);
  SetupConv2dParams<QnnCSIConv2DAttrs>(params_name, attr);
  if (fuse_zp) {
    std::ostringstream t0;
    t0 << params_name << "->conv_extra.fuse_zp2bias = true";
    func_def_.PushDecl(t0);
  }

  PushOutput(output_name, call, output_qinfo->dtype);

  params_common_setup(decl, call, op_name, params_name, attr->layer_name.c_str());
  end_stream(decl, op_name);
}

void CodegenCSINN::Conv3d(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream buf;
  CSINNOP* op = new CSINNOP;
  const auto* attr = call->attrs.as<QnnCSIConv3DAttrs>();
  CHECK(attr);

  CHECK(call->args.size() == 3) << "Conv3d expects 3 args";

  /* Make function call with arguments start */
  decl << "(";

  /* Emit_ input tensor */
  visit(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string complete_name = get_complete_layer_name("conv3d", attr->layer_name.c_str());
  op->set_name(complete_name);
  auto input_qinfo = GetCallOPQuant(call, 0);
  string input_name = CreateInputTensor(op, decl, call, 0, input_qinfo);

  /* Emit output tensor */
  auto output_qinfo = GetCallOPQuant(call, 3);
  string output_name = CreateOutputTensor(op, decl, call, output_qinfo);

  collect_quant_info(complete_name, attr->q_params, 3);

  /* Emit kernel tensor */
  visit(call->args[1]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto kernel = constant_[0];
  auto wshape = call->args[1]->get_shape();
  string kernel_name = "kernel_" + to_string(buf_idx_);
  auto kernel_qinfo = GetCallOPQuant(call, 1);
  if (cfg->quantization_scheme == "CSINN_QUANT_FLOAT16_W_INT8") {
    kernel_qinfo->dtype = "int8_t";
    CreateWeightTensor(op, kernel, kernel_name, wshape, kernel_qinfo);
  } else {
    CreateConstantTensor(op, kernel, kernel_name, wshape, kernel_qinfo);
  }
  decl << ", " << kernel_name;

  /* Emit bias tensor */
  visit(call->args[2]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto bias = constant_[0];
  auto bshape = call->args[2]->get_shape();
  string bias_name = "bias_" + to_string(buf_idx_);
  auto bias_qinfo = GetCallOPQuant(call, 2);
  CreateConstantTensor(op, bias, bias_name, bshape, bias_qinfo->dtype, input_qinfo, kernel_qinfo,
                       bias_qinfo);

  decl << ", " << bias_name;

  string params_name = "params_" + to_string(buf_idx_);

  decl << ", " << params_name << ")";
  push_decl(op);
  SetupConv3dParams<QnnCSIConv3DAttrs>(params_name, attr);

  PushOutput(output_name, call);
  params_common_setup(decl, call, "conv3d", params_name, attr->layer_name.c_str());
  end_stream(decl, "conv3d");
}

void CodegenCSINN::Dilation2d(const CallNode* call) {
  std::ostringstream decl;
  CSINNOP* op = new CSINNOP;
  const auto* attr = call->attrs.as<QnnCSIDilation2DAttrs>();
  CHECK(attr);

  CHECK(call->args.size() == 2) << "Dilation2D expects 2 args";

  /* Make function call with arguments start */
  decl << "(";

  /* Emit_ input tensor */
  visit(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string complete_name = get_complete_layer_name("dilation2d", attr->layer_name.c_str());
  op->set_name(complete_name);
  auto input_qinfo = GetCallOPQuant(call, 0);
  string input_name = CreateInputTensor(op, decl, call, 0, input_qinfo);

  /* Emit output tensor */
  auto output_qinfo = GetCallOPQuant(call, 2);
  string output_name = CreateOutputTensor(op, decl, call, output_qinfo);

  collect_quant_info(complete_name, attr->q_params, 2);

  /* Emit kernel tensor */
  visit(call->args[1]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto kernel = constant_[0];
  auto wshape = call->args[1]->get_shape();

  string kernel_name = "kernel_" + to_string(buf_idx_);
  auto kernel_qinfo = GetCallOPQuant(call, 1);
  if (cfg->quantization_scheme == "CSINN_QUANT_FLOAT16_W_INT8") {
    kernel_qinfo->dtype = "int8_t";
    CreateWeightTensor(op, kernel, kernel_name, wshape, kernel_qinfo);
  } else {
    CreateConstantTensor(op, kernel, kernel_name, wshape, kernel_qinfo);
  }

  decl << ", " << kernel_name;

  string params_name = "params_" + to_string(buf_idx_);

  decl << ", " << params_name << ")";
  push_decl(op);
  SetupDilation2dParams<QnnCSIDilation2DAttrs>(params_name, attr);

  PushOutput(output_name, call);
  params_common_setup(decl, call, "dilation2d", params_name, attr->layer_name.c_str());
  end_stream(decl, "dilation2d");
}

void CodegenCSINN::DeConv2d(const CallNode* call) {
  std::ostringstream decl;
  CSINNOP* op = new CSINNOP;
  const auto* attr = call->attrs.as<QnnCSIDeConv2DAttrs>();
  CHECK(attr);
  CHECK(call->args.size() == 3) << "DeConv2d expects 3 args";

  /* Make function call with arguments start */
  decl << "(";

  /* Emit_ input tensor */
  visit(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";

  string complete_name = get_complete_layer_name("deconv2d", attr->layer_name.c_str());
  op->set_name(complete_name);

  auto input_qinfo = GetCallOPQuant(call, 0);
  string input_name = CreateInputTensor(op, decl, call, 0, input_qinfo);

  /* Emit output tensor */
  auto output_qinfo = GetCallOPQuant(call, 3);
  string output_name = CreateOutputTensor(op, decl, call, output_qinfo);

  collect_quant_info(complete_name, attr->q_params, 3);

  output2params[output_name] = complete_name;

  /* Emit kernel tensor */
  visit(call->args[1]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto kernel = constant_[0];
  auto wshape = call->args[1]->get_shape();
  string kernel_name = "kernel_" + to_string(buf_idx_);
  auto kernel_qinfo = GetCallOPQuant(call, 1);
  if (cfg->quantization_scheme == "CSINN_QUANT_FLOAT16_W_INT8") {
    kernel_qinfo->dtype = "int8_t";
    CreateWeightTensor(op, kernel, kernel_name, wshape, kernel_qinfo);
  } else {
    CreateConstantTensor(op, kernel, kernel_name, wshape, kernel_qinfo);
  }
  decl << ", " << kernel_name;
  // fix kernel layout
  auto kernel_tensor = op->get_tensor(kernel_name);
  if (kernel_tensor->tensor->layout == CSINN_LAYOUT_OIHW) {
    kernel_tensor->tensor->layout = CSINN_LAYOUT_IOHW;
  }

  /* Emit bias tensor */
  visit(call->args[2]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto bias = constant_[0];
  auto bshape = call->args[2]->get_shape();
  string bias_name = "bias_" + to_string(buf_idx_);
  bool fuse_zp = false;

  CreateBiasTensor(op, call, bias, bias_name, attr->q_params, &fuse_zp);

  decl << ", " << bias_name;

  string params_name = "params_" + to_string(buf_idx_);

  decl << ", " << params_name << ")";
  push_decl(op);
  SetupConv2dParams<QnnCSIDeConv2DAttrs>(params_name, attr);
  std::ostringstream t0;
  Array<IndexExpr> output_padding = attr->output_padding;
  t0 << params_name
     << "->out_pad_height = " << to_string(output_padding[0].as<IntImmNode>()->value);
  func_def_.PushDecl(t0);
  t0 << params_name << "->out_pad_width = " << to_string(output_padding[1].as<IntImmNode>()->value);
  func_def_.PushDecl(t0);

  PushOutput(output_name, call, output_qinfo->dtype);

  params_common_setup(decl, call, "deconv2d", params_name, attr->layer_name.c_str());
  end_stream(decl, "deconv2d");
}

void CodegenCSINN::DeConv3d(const CallNode* call) {
  std::ostringstream decl;
  CSINNOP* op = new CSINNOP;
  const auto* attr = call->attrs.as<QnnCSIDeConv3DAttrs>();
  CHECK(attr);

  CHECK(call->args.size() == 3) << "DeConv3d expects 3 args";

  /* Make function call with arguments start */
  decl << "(";

  /* Emit_ input tensor */
  visit(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string complete_name = get_complete_layer_name("deconv3d", attr->layer_name.c_str());
  op->set_name(complete_name);
  auto input_qinfo = GetCallOPQuant(call, 0);
  string input_name = CreateInputTensor(op, decl, call, 0, input_qinfo);

  /* Emit output tensor */
  auto output_qinfo = GetCallOPQuant(call, 3);
  string output_name = CreateOutputTensor(op, decl, call, output_qinfo);

  collect_quant_info(complete_name, attr->q_params, 3);

  /* Emit kernel tensor */
  visit(call->args[1]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto kernel = constant_[0];
  auto wshape = call->args[1]->get_shape();

  string kernel_name = "kernel_" + to_string(buf_idx_);
  auto kernel_qinfo = GetCallOPQuant(call, 1);
  if (cfg->quantization_scheme == "CSINN_QUANT_FLOAT16_W_INT8") {
    kernel_qinfo->dtype = "int8_t";
    CreateWeightTensor(op, kernel, kernel_name, wshape, kernel_qinfo);
  } else {
    CreateConstantTensor(op, kernel, kernel_name, wshape, kernel_qinfo);
  }

  decl << ", " << kernel_name;

  /* Emit bias tensor */
  visit(call->args[2]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto bias = constant_[0];
  auto bshape = call->args[2]->get_shape();
  string bias_name = "bias_" + to_string(buf_idx_);
  auto bias_qinfo = GetCallOPQuant(call, 2);
  CreateConstantTensor(op, bias, bias_name, bshape, bias_qinfo->dtype, input_qinfo, kernel_qinfo,
                       bias_qinfo);

  decl << ", " << bias_name;

  string params_name = "params_" + to_string(buf_idx_);

  decl << ", " << params_name << ")";
  push_decl(op);
  SetupConv3dParams<QnnCSIDeConv3DAttrs>(params_name, attr);

  PushOutput(output_name, call);
  params_common_setup(decl, call, "deconv3d", params_name, attr->layer_name.c_str());
  end_stream(decl, "deconv3d");
}

void CodegenCSINN::Dense(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  CSINNOP* op = new CSINNOP;
  const auto* dense_attr = call->attrs.as<QnnCSIDenseAttrs>();
  CHECK(dense_attr);

  CHECK(call->args.size() == 3) << "Dense expects 3 args";

  // Make function call with input buffers when visiting arguments
  decl << "(";

  /* Emit input tensor */
  visit(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";

  string complete_name = get_complete_layer_name("fullyconnected", dense_attr->layer_name.c_str());
  op->set_name(complete_name);

  auto input_qinfo = GetCallOPQuant(call, 0);
  string input_name = CreateInputTensor(op, decl, call, 0, input_qinfo);

  /* Emit output tensor */
  auto output_qinfo = GetCallOPQuant(call, 3);
  string output_name = CreateOutputTensor(op, decl, call, output_qinfo);
  output2params[output_name] = complete_name;

  collect_quant_info(complete_name, dense_attr->q_params, 3);

  /* Emit kernel tensor */
  visit(call->args[1]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto kernel = constant_[0];
  auto wshape = call->args[1]->get_shape();

  string kernel_name = "kernel_" + to_string(buf_idx_);
  auto kernel_qinfo = GetCallOPQuant(call, 1);

  if (cfg->quantization_scheme == "CSINN_QUANT_FLOAT16_W_INT8") {
    kernel_qinfo->dtype = "int8_t";
    CreateWeightTensor(op, kernel, kernel_name, wshape, kernel_qinfo);
  } else {
    CreateConstantTensor(op, kernel, kernel_name, wshape, kernel_qinfo);
  }

  decl << ", " << kernel_name;

  /* Emit bias tensor */
  visit(call->args[2]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto bias = constant_[0];
  auto bshape = call->args[2]->get_shape();
  string bias_name = "bias_" + to_string(buf_idx_);
  bool fuse_zp = false;

  CreateBiasTensor(op, call, bias, bias_name, dense_attr->q_params, &fuse_zp, "dense_bias");

  decl << ", " << bias_name;

  string params_name = "params_" + to_string(buf_idx_);

  decl << ", " << params_name << ")";
  push_decl(op);
  malloc_params("csinn_fc_params", params_name);
  int units;
  if (dense_attr->units.defined()) {
    units = dense_attr->units.as<IntImmNode>()->value;
  } else {
    units = wshape[0];
  }
  t0 << params_name << "->units = " << to_string(units);
  func_def_.PushDecl(t0);
  if (fuse_zp) {
    t0 << params_name << "->fc_extra.fuse_zp2bias = true";
    func_def_.PushDecl(t0);
  }

  PushOutput(output_name, call, output_qinfo->dtype);

  params_common_setup(decl, call, "fullyconnected", params_name, dense_attr->layer_name.c_str());
  end_stream(decl, "fullyconnected");
}

void CodegenCSINN::Softmax(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  CSINNOP* op = new CSINNOP;
  const auto* attr = call->attrs.as<QnnCSIAxisAttrs>();
  SisoOp<QnnCSIAxisAttrs>(op, decl, call, attr, "softmax");

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";
  push_decl(op);
  malloc_params("csinn_softmax_params", params_name);
  int actual_aixs = attr->axis;
  auto ishape = call->args[0]->get_shape();
  if (attr->axis < 0) {
    actual_aixs += ishape.size();
  }
  t0 << params_name << "->axis = " << to_string(actual_aixs);
  func_def_.PushDecl(t0);

  params_common_setup(decl, call, "softmax", params_name, attr->layer_name.c_str());
  end_stream(decl, "softmax");
}

void CodegenCSINN::Reverse(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  CSINNOP* op = new CSINNOP;
  const auto* attr = call->attrs.as<QnnCSIAxisAttrs>();
  SisoOp<QnnCSIAxisAttrs>(op, decl, call, attr, "reverse");
  auto ishape = call->args[0]->get_shape();
  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";
  push_decl(op);
  malloc_params("csinn_reverse_params", params_name);
  int axis = attr->axis < 0 ? attr->axis + ishape.size() : attr->axis;
  t0 << params_name << "->axis = " << axis;
  func_def_.PushDecl(t0);

  params_common_setup(decl, call, "reverse", params_name, attr->layer_name.c_str());
  end_stream(decl, "reverse");
}

void CodegenCSINN::LogSoftmax(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  CSINNOP* op = new CSINNOP;
  const auto* attr = call->attrs.as<QnnCSIAxisAttrs>();
  SisoOp<QnnCSIAxisAttrs>(op, decl, call, attr, "log_softmax");

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";
  push_decl(op);
  malloc_params("csinn_softmax_params", params_name);
  int axis = attr->axis == -1 ? 1 : attr->axis;
  t0 << params_name << "->axis = " << to_string(axis);
  func_def_.PushDecl(t0);

  params_common_setup(decl, call, "log_softmax", params_name, attr->layer_name.c_str());
  end_stream(decl, "log_softmax");
}

void CodegenCSINN::ExpandDims(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  CSINNOP* op = new CSINNOP;
  const auto* attr = call->attrs.as<QnnCSIExpandDimsAttrs>();
  SisoOp<QnnCSIExpandDimsAttrs>(op, decl, call, attr, "expand_dims");

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";
  push_decl(op);
  malloc_params("csinn_expand_dims_params", params_name);
  t0 << params_name << "->axis = " << to_string(attr->axis);
  func_def_.PushDecl(t0);

  params_common_setup(decl, call, "expand_dims", params_name, attr->layer_name.c_str());
  end_stream(decl, "expand_dims");
}

void CodegenCSINN::MaxPool2d(const CallNode* call) {
  std::ostringstream decl;
  CSINNOP* op = new CSINNOP;
  const auto* attr = call->attrs.as<QnnCSIMaxPool2DAttrs>();
  SisoOp<QnnCSIMaxPool2DAttrs>(op, decl, call, attr, "maxpool2d");

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";
  push_decl(op);
  SetupPoolParams(params_name, attr);

  params_common_setup(decl, call, "maxpool2d", params_name, attr->layer_name.c_str());
  end_stream(decl, "maxpool2d");
}

void CodegenCSINN::AvgPool2d(const CallNode* call) {
  std::ostringstream decl;
  CSINNOP* op = new CSINNOP;
  const auto* attr = call->attrs.as<QnnCSIAvgPool2DAttrs>();
  SisoOp<QnnCSIAvgPool2DAttrs>(op, decl, call, attr, "avgpool2d");

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";
  push_decl(op);
  SetupPoolParams(params_name, attr);
  std::ostringstream t0;
  auto count_include_pad = attr->count_include_pad;
  t0 << params_name << "->count_include_pad = " << to_string(count_include_pad);
  func_def_.PushDecl(t0);

  params_common_setup(decl, call, "avgpool2d", params_name, attr->layer_name.c_str());
  end_stream(decl, "avgpool2d");
}

void CodegenCSINN::AvgPool3d(const CallNode* call) {
  std::ostringstream decl;
  CSINNOP* op = new CSINNOP;
  const auto* attr = call->attrs.as<QnnCSIAvgPool3DAttrs>();
  SisoOp<QnnCSIAvgPool3DAttrs>(op, decl, call, attr, "avgpool3d");

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";
  push_decl(op);
  SetupPool3DParams(params_name, attr);

  params_common_setup(decl, call, "avgpool3d", params_name, attr->layer_name.c_str(),
                      "CSINN_NCDHW");
  end_stream(decl, "avgpool3d");
}

void CodegenCSINN::MaxPool3d(const CallNode* call) {
  std::ostringstream decl;
  CSINNOP* op = new CSINNOP;
  const auto* attr = call->attrs.as<QnnCSIMaxPool3DAttrs>();
  SisoOp<QnnCSIMaxPool3DAttrs>(op, decl, call, attr, "maxpool3d");

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";
  push_decl(op);
  SetupPool3DParams(params_name, attr);

  params_common_setup(decl, call, "maxpool3d", params_name, attr->layer_name.c_str(),
                      "CSINN_NCDHW");
  end_stream(decl, "maxpool3d");
}

void CodegenCSINN::GlobalAvgPool2d(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  CSINNOP* op = new CSINNOP;
  const auto* attr = call->attrs.as<QnnCSIGlobalAvgPoolAttrs>();
  SisoOp<QnnCSIGlobalAvgPoolAttrs>(op, decl, call, attr, "global_avgpool2d");

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";
  push_decl(op);
  malloc_params("csinn_pool_params", params_name);

  params_common_setup(decl, call, "global_avgpool2d", params_name, attr->layer_name.c_str());
  end_stream(decl, "global_avgpool2d");
}

void CodegenCSINN::GlobalMaxPool2d(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  CSINNOP* op = new CSINNOP;
  const auto* attr = call->attrs.as<QnnCSIGlobalMaxPoolAttrs>();
  SisoOp<QnnCSIGlobalMaxPoolAttrs>(op, decl, call, attr, "global_maxpool2d");

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";
  push_decl(op);
  malloc_params("csinn_pool_params", params_name);

  params_common_setup(decl, call, "global_maxpool2d", params_name, attr->layer_name.c_str());
  end_stream(decl, "global_maxpool2d");
}

void CodegenCSINN::Maxpool2dWithArgmax(const CallNode* call) {
  std::ostringstream decl;
  CSINNOP* op = new CSINNOP;
  const auto* attr = call->attrs.as<QnnCSIMaxPool2DAttrs>();
  SisoOp<QnnCSIMaxPool2DAttrs>(op, decl, call, attr);

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";
  push_decl(op);
  SetupPoolParams(params_name, attr);

  params_common_setup(decl, call, "maxpool2d", params_name, attr->layer_name.c_str());
  end_stream(decl, "maxpool2d");
}

void CodegenCSINN::MaxPool2dLocat(const CallNode* call) {
  std::ostringstream decl;
  CSINNOP* op = new CSINNOP;
  const auto* attr = call->attrs.as<QnnCSIMaxPool2DLocatAttrs>();
  CHECK(attr);

  CHECK(call->args.size() == 1) << "MaxPool2dLocat expects 1 args";

  /* Make function call with arguments start */
  decl << "(";

  /* Emit_ input tensor */
  visit(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string complete_name = get_complete_layer_name("maxpool_locat", attr->layer_name.c_str());
  op->set_name(complete_name);
  string input_name = CreateInputTensor(op, decl, call, 0, GetCallOPQuant(call, 0));

  /* Emit output tensor */
  auto output_qinfo = GetCallOPQuant(call, 1);
  string output_name = CreateOutputTensor(op, decl, call, output_qinfo);

  collect_quant_info(complete_name, attr->q_params, 1);

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";
  push_decl(op);
  SetupPoolParams(params_name, attr);

  PushOutput(output_name, call, "int32_t");
  params_common_setup(decl, call, "maxpool2d_locat", params_name, attr->layer_name.c_str());
  end_stream(decl, "maxpool2d_locat");
}

void CodegenCSINN::UnPool2d(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  CSINNOP* op = new CSINNOP;
  const auto* attr = call->attrs.as<QnnCSIUnPoolingAttrs>();
  CHECK(attr);
  CHECK(call->args.size() == 2) << "Unpool2d expects 2 args";

  /* Make function call with arguments start */
  decl << "(";

  /* Emit_ input tensor */
  visit(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string input_name = CreateInputTensor(op, decl, call, 0, GetCallOPQuant(call, 0));
  decl << ", ";

  /* Emit_ mask tensor */
  visit(call->args[1]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string complete_name = get_complete_layer_name("unpooling", attr->layer_name.c_str());
  op->set_name(complete_name);
  string mask_name = CreateInputTensor(op, decl, call, 1, GetCallOPQuant(call, 1));

  /* Emit output tensor */
  string output_name = CreateOutputTensor(op, decl, call, GetCallOPQuant(call, 2));

  collect_quant_info(complete_name, attr->q_params, 2);

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";
  push_decl(op);
  malloc_params("csinn_unpooling_params", params_name);

  t0 << params_name
     << "->pad_out_height = " << to_string(attr->out_padding[0].as<IntImmNode>()->value);
  func_def_.PushDecl(t0);
  t0 << params_name
     << "->pad_out_width = " << to_string(attr->out_padding[1].as<IntImmNode>()->value);
  func_def_.PushDecl(t0);
  t0 << params_name << "->scale_height = " << to_string(attr->scales[0].as<IntImmNode>()->value);
  func_def_.PushDecl(t0);
  t0 << params_name << "->scale_width = " << to_string(attr->scales[1].as<IntImmNode>()->value);
  func_def_.PushDecl(t0);

  PushOutput(output_name, call);
  params_common_setup(decl, call, "unpooling", params_name, attr->layer_name.c_str());
  end_stream(decl, "unpooling");
}

void CodegenCSINN::PSROIPool(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  CSINNOP* op = new CSINNOP;
  const auto* attr = call->attrs.as<QnnCSIPSROIPoolingAttrs>();
  CHECK(attr);

  CHECK(call->args.size() == 2) << "PSROIPooling expects 2 args";

  /* Make function call with arguments start */
  decl << "(";

  /* Emit_ input tensor */
  visit(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string input_name = CreateInputTensor(op, decl, call, 0, GetCallOPQuant(call, 0));
  decl << ", ";

  /* Emit_ roi tensor */
  visit(call->args[1]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string complete_name = get_complete_layer_name("psroipooling", attr->layer_name.c_str());
  op->set_name(complete_name);
  string roi_name = CreateInputTensor(op, decl, call, 1, GetCallOPQuant(call, 1));

  /* Emit output tensor */
  string output_name = CreateOutputTensor(op, decl, call, GetCallOPQuant(call, 2));

  collect_quant_info(complete_name, attr->q_params, 2);

  int32_t multiplier;
  int32_t shift;
  GetMultiplierAndShift(attr->spatial_scale, &multiplier, &shift);

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";
  push_decl(op);
  malloc_params("csinn_psroipooling_params", params_name);

  t0 << params_name << "->output_dim = " << to_string(attr->output_dim);
  func_def_.PushDecl(t0);
  t0 << params_name << "->group_size = " << to_string(attr->group_size);
  func_def_.PushDecl(t0);
  t0 << params_name << "->spatial_scale = " << to_string(attr->spatial_scale);
  func_def_.PushDecl(t0);
  t0 << params_name << "->spatial_scale_multiplier = " << to_string(multiplier);
  func_def_.PushDecl(t0);
  t0 << params_name << "->spatial_scale_shift = " << to_string(shift);
  func_def_.PushDecl(t0);

  PushOutput(output_name, call);
  params_common_setup(decl, call, "psroipooling", params_name, attr->layer_name.c_str());
  end_stream(decl, "psroipooling");
}

void CodegenCSINN::ROIPool(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  CSINNOP* op = new CSINNOP;
  const auto* attr = call->attrs.as<QnnCSIROIPoolingAttrs>();
  CHECK(attr);

  CHECK(call->args.size() == 2) << "ROIPooling expects 2 args";

  /* Make function call with arguments start */
  decl << "(";

  /* Emit_ input tensor */
  visit(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string complete_name = get_complete_layer_name("roipool", attr->layer_name.c_str());
  op->set_name(complete_name);
  string input_name = CreateInputTensor(op, decl, call, 0, GetCallOPQuant(call, 0));
  decl << ", ";

  /* Emit_ roi tensor */
  visit(call->args[1]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string roi_name = CreateInputTensor(op, decl, call, 1, GetCallOPQuant(call, 1));

  /* Emit output tensor */
  string output_name = CreateOutputTensor(op, decl, call, GetCallOPQuant(call, 2));

  collect_quant_info(complete_name, attr->q_params, 2);

  int32_t multiplier;
  int32_t shift;
  GetMultiplierAndShift(attr->spatial_scale, &multiplier, &shift);

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";
  push_decl(op);
  Array<IndexExpr> pooled_size = attr->pooled_size;

  malloc_params("csinn_roi_pool_params", params_name);

  t0 << params_name << "->pooled_size_h = " << to_string(pooled_size[0].as<IntImmNode>()->value);
  func_def_.PushDecl(t0);
  t0 << params_name << "->pooled_size_w = " << to_string(pooled_size[1].as<IntImmNode>()->value);
  func_def_.PushDecl(t0);
  t0 << params_name << "->spatial_scale = " << to_string(attr->spatial_scale);
  func_def_.PushDecl(t0);
  t0 << params_name << "->spatial_scale_multiplier = " << to_string(multiplier);
  func_def_.PushDecl(t0);
  t0 << params_name << "->spatial_scale_shift = " << to_string(shift);
  func_def_.PushDecl(t0);

  PushOutput(output_name, call);
  params_common_setup(decl, call, "roipool", params_name, attr->layer_name.c_str());
  end_stream(decl, "roipool");
}

void CodegenCSINN::Proposal(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  std::ostringstream mstream, sstream, fstream;
  CSINNOP* op = new CSINNOP;
  const auto* attr = call->attrs.as<QnnCSIProposalAttrs>();
  CHECK(attr);

  CHECK(call->args.size() == 3) << "Proposal expects 3 args";

  /* Make function call with arguments start */
  decl << "(";

  /* Emit_ cls tensor */
  visit(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string complete_name = get_complete_layer_name("proposal", attr->layer_name.c_str());
  op->set_name(complete_name);
  string cls_name = CreateInputTensor(op, decl, call, 0, GetCallOPQuant(call, 0));
  decl << ", ";

  /* Emit_ bbox tensor */
  visit(call->args[1]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string bbox_name = CreateInputTensor(op, decl, call, 1, GetCallOPQuant(call, 1));

  /* Emit_ im_info tensor */
  visit(call->args[2]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto im_info = constant_[0];
  auto im_info_shape = call->args[2]->get_shape();
  string im_info_name = "im_info_" + to_string(buf_idx_);
  int32_t layout = GetCSINNTensorWeightLayout(im_info_shape);
  CSINNConstantTensor* ret = CreateConstantTensorBase(im_info_name, im_info->byte_size(),
                                                      im_info_shape, "int32_t", layout);
  ret->tensor->dtype = CSINN_DTYPE_FLOAT32;
  ret->set_const(im_info);
  op->push_constant(ret);

  decl << "," << im_info_name;

  /* Emit output tensor */
  string output_name = CreateOutputTensor(op, decl, call, GetCallOPQuant(call, 3));

  collect_quant_info(complete_name, attr->q_params, 3);

  int32_t scales_num = attr->scales.size();
  int32_t ratios_num = attr->ratios.size();

  int32_t multiplier;
  int32_t shift;
  mstream << "int32_t scale_multipliers_" << to_string(buf_idx_) << "[" << scales_num << "] = {";
  sstream << "int32_t scale_shifts_" << to_string(buf_idx_) << "[" << scales_num << "] = {";
  fstream << "float scale_" << to_string(buf_idx_) << "[" << scales_num << "] = {";
  for (int i = 0; i < scales_num; i++) {
    float scale = attr->scales[i].as<FloatImmNode>()->value;
    GetMultiplierAndShift(scale, &multiplier, &shift);
    mstream << to_string(multiplier) << ", ";
    sstream << to_string(shift) << ", ";
    fstream << to_string(scale) << ", ";
  }
  mstream << "}";
  func_def_.PushDecl(mstream);
  sstream << "}";
  func_def_.PushDecl(sstream);
  fstream << "}";
  func_def_.PushDecl(fstream);

  mstream << "int32_t ratio_multipliers_" << to_string(buf_idx_) << "[" << ratios_num << "] = {";
  sstream << "int32_t ratio_shifts_" << to_string(buf_idx_) << "[" << ratios_num << "] = {";
  fstream << "float ratios_" << to_string(buf_idx_) << "[" << scales_num << "] = {";
  for (int i = 0; i < ratios_num; i++) {
    float ratios = attr->ratios[i].as<FloatImmNode>()->value;
    GetMultiplierAndShift(ratios, &multiplier, &shift);
    mstream << to_string(multiplier) << ", ";
    sstream << to_string(shift) << ", ";
    fstream << to_string(ratios) << ", ";
  }
  mstream << "}";
  func_def_.PushDecl(mstream);
  sstream << "}";
  func_def_.PushDecl(sstream);
  fstream << "}";
  func_def_.PushDecl(fstream);

  GetMultiplierAndShift(attr->threshold, &multiplier, &shift);

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";
  push_decl(op);
  malloc_params("csinn_proposal_params", params_name);
  t0 << params_name << "->scales = scale_" << to_string(buf_idx_);
  func_def_.PushDecl(t0);
  t0 << params_name << "->scale_multipliers = scale_multipliers_" << to_string(buf_idx_);
  func_def_.PushDecl(t0);
  t0 << params_name << "->scale_shifts = scale_shifts_" << to_string(buf_idx_);
  func_def_.PushDecl(t0);
  t0 << params_name << "->scales_num = " << to_string(scales_num);
  func_def_.PushDecl(t0);
  t0 << params_name << "->ratios = ratios_" << to_string(buf_idx_);
  func_def_.PushDecl(t0);
  t0 << params_name << "->ratio_multipliers = ratio_multipliers_" << to_string(buf_idx_);
  func_def_.PushDecl(t0);
  t0 << params_name << "->ratio_shifts = ratio_shifts_" << to_string(buf_idx_);
  func_def_.PushDecl(t0);
  t0 << params_name << "->ratios_num = " << to_string(ratios_num);
  func_def_.PushDecl(t0);
  t0 << params_name << "->feature_stride = " << to_string(attr->feature_stride);
  func_def_.PushDecl(t0);
  t0 << params_name << "->threshold = " << to_string(attr->threshold);
  func_def_.PushDecl(t0);
  t0 << params_name << "->threshold_multiplier = " << to_string(multiplier);
  func_def_.PushDecl(t0);
  t0 << params_name << "->threshold_shift = " << to_string(shift);
  func_def_.PushDecl(t0);
  t0 << params_name << "->rpn_pre_nms_top_n = " << to_string(attr->rpn_pre_nms_top_n);
  func_def_.PushDecl(t0);
  t0 << params_name << "->rpn_post_nms_top_n = " << to_string(attr->rpn_post_nms_top_n);
  func_def_.PushDecl(t0);
  t0 << params_name << "->rpn_min_size = " << to_string(attr->rpn_min_size);
  func_def_.PushDecl(t0);
  t0 << params_name << "->iou_loss = " << to_string(attr->iou_loss);
  func_def_.PushDecl(t0);

  PushOutput(output_name, call);
  params_common_setup(decl, call, "proposal", params_name, attr->layer_name.c_str());
  end_stream(decl, "proposal");
}

void CodegenCSINN::UpSampling(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  CSINNOP* op = new CSINNOP;
  const auto* attr = call->attrs.as<QnnCSIUpSamplingAttrs>();
  CHECK(attr);
  CHECK(call->args.size() == 1) << "UpSampling expects 1 args";

  /* Make function call with arguments start */
  decl << "(";

  /* Emit_ input tensor */
  visit(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string complete_name = get_complete_layer_name("resize", attr->layer_name.c_str());
  op->set_name(complete_name);

  auto input_qinfo = GetCallOPQuant(call, 0);
  string input_name = CreateInputTensor(op, decl, call, 0, input_qinfo);

  /* Emit output tensor */
  auto output_qinfo = GetCallOPQuant(call, 1);
  string output_name = CreateOutputTensor(op, decl, call, output_qinfo);
  output2params[output_name] = complete_name;

  collect_quant_info(complete_name, attr->q_params, 1);

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";
  push_decl(op);
  malloc_params("csinn_resize_params", params_name);
  t0 << params_name << "->resize_mode = ";
  if (attr->method == "bilinear") {
    t0 << "CSINN_RESIZE_BILINEAR";
  } else if (attr->method == "linear") {
    t0 << "CSINN_RESIZE_BILINEAR";
  } else if (attr->method == "nearest_neighbor") {
    t0 << "CSINN_RESIZE_NEAREST_NEIGHBOR";
  } else if (attr->method == "nearest_bicubic") {
    t0 << "CSINN_RESIZE_NEAREST_BICUBIC";
  } else if (attr->method == "cubic") {
    t0 << "CSINN_RESIZE_NEAREST_BICUBIC";
  } else {
    CHECK(0);
  }
  func_def_.PushDecl(t0);
  t0 << params_name << "->align_corners = " << to_string(attr->align_corners);
  func_def_.PushDecl(t0);

  PushOutput(output_name, call, output_qinfo->dtype);

  params_common_setup(decl, call, "resize", params_name, attr->layer_name.c_str());
  end_stream(decl, "resize");
}

void CodegenCSINN::Relu(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream buf;
  std::ostringstream t0;
  CSINNOP* op = new CSINNOP;
  const auto* attr = call->attrs.as<QnnCSIUnaryAttrs>();
  SisoOp<QnnCSIUnaryAttrs>(op, decl, call, attr, "relu");

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";
  push_decl(op);
  malloc_params("csinn_relu_params", params_name);
  params_common_setup(decl, call, "relu", params_name, attr->layer_name.c_str());
  end_stream(decl, "relu");
}

void CodegenCSINN::Fsmn(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  CSINNOP* op = new CSINNOP;
  const auto* attr = call->attrs.as<QnnCSIFsmnAttrs>();
  CHECK(attr);

  CHECK(call->args.size() == 5) << "fsmn expects 5 args";

  decl << "(";

  /* Emit_ input tensor */
  visit(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string complete_name = get_complete_layer_name("fsmn", attr->layer_name.c_str());
  op->set_name(complete_name);
  string input_name = CreateInputTensor(op, decl, call, 0, GetCallOPQuant(call, 0));

  /* Emit l_filter tensor */
  visit(call->args[1]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto l_filter = constant_[0];
  auto lshape = call->args[1]->get_shape();
  string l_filter_name = "l_filter_" + to_string(buf_idx_);
  CreateConstantTensor(op, l_filter, l_filter_name, lshape, GetCallOPQuant(call, 1));
  decl << ", " << l_filter_name;

  /* Emit r_filter tensor */
  visit(call->args[2]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto r_filter = constant_[0];
  auto rshape = call->args[2]->get_shape();
  string r_filter_name = "r_filter_" + to_string(buf_idx_);
  CreateConstantTensor(op, r_filter, r_filter_name, rshape, GetCallOPQuant(call, 2));
  decl << ", " << r_filter_name;

  /* Emit frame sequence tensor */
  visit(call->args[3]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto sequence = constant_[0];
  auto seq_shape = call->args[3]->get_shape();
  string sequence_name = "sequence_" + to_string(buf_idx_);
  CreateConstantTensor(op, sequence, sequence_name, seq_shape, GetCallOPQuant(call, 3));
  decl << ", " << sequence_name;

  /* Emit frame counter tensor */
  visit(call->args[4]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto frame_counter = constant_[0];
  auto counter_shape = call->args[4]->get_shape();
  string counter_name = "frame_counter_" + to_string(buf_idx_);
  CreateConstantTensor(op, frame_counter, counter_name, counter_shape, GetCallOPQuant(call, 4));
  decl << ", " << counter_name;

  /* Emit output tensor */
  string output_name = CreateOutputTensor(op, decl, call, GetCallOPQuant(call, 5));

  collect_quant_info(complete_name, attr->q_params, 4);

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";
  push_decl(op);
  malloc_params("csinn_fsmn_params", params_name);

  t0 << params_name << "->l_order = " << to_string(attr->l_order);
  func_def_.PushDecl(t0);
  t0 << params_name << "->r_order = " << to_string(attr->r_order);
  func_def_.PushDecl(t0);
  t0 << params_name << "->l_stride = " << to_string(attr->l_stride);
  func_def_.PushDecl(t0);
  t0 << params_name << "->r_stride = " << to_string(attr->r_stride);
  func_def_.PushDecl(t0);
  t0 << params_name << "->unavailable_frames = " << to_string(attr->unavailable_frames);
  func_def_.PushDecl(t0);

  PushOutput(output_name, call);
  params_common_setup(decl, call, "fsmn", params_name, attr->layer_name.c_str());
  end_stream(decl, "fsmn");
}

void CodegenCSINN::Full(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  CSINNOP* op = new CSINNOP;
  const auto* attr = call->attrs.as<QnnCSIFullAttrs>();
  auto shape = attr->shape;
  SisoOp<QnnCSIFullAttrs>(op, decl, call, attr, "full");

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";
  push_decl(op);
  t0 << "int32_t *shape_" << buf_idx_ << " = malloc(" << shape.size() << " * 4)";
  func_def_.PushDecl(t0);
  for (uint k = 0; k < shape.size(); k++) {
    t0 << "shape_" << buf_idx_ << "[" << k << "] = " << Downcast<IndexExpr>(shape[k]);
    func_def_.PushDecl(t0);
  }

  malloc_params("csinn_full_params", params_name);
  t0 << params_name << "->shape = shape_" << buf_idx_;
  func_def_.PushDecl(t0);

  params_common_setup(decl, call, "full", params_name, attr->layer_name.c_str());
  end_stream(decl, "full");
}

void CodegenCSINN::Take(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  CSINNOP* op = new CSINNOP;
  const auto* attr = call->attrs.as<QnnCSITakeAttrs>();
  CHECK(attr);

  String mode = attr->mode;
  // CHECK(mode == "fast") << "only mode is fast, input indices are in bound.";
  CHECK(call->args.size() == 2) << "take expects 2 args";

  /* Make function call with arguments start */
  decl << "(";

  /* Emit_ input tensor */
  visit(call->args[0]);
  string input_name;
  string complete_name = get_complete_layer_name("gather", attr->layer_name.c_str());
  op->set_name(complete_name);
  auto in_shape = call->args[0]->get_shape();
  int* axis = NULL;
  if (attr->axis.defined()) {
    axis = static_cast<int*>(malloc(4));
    axis[0] = static_cast<int>(attr->axis->value);
    axis[0] = axis[0] < 0 ? axis[0] + in_shape.size() : axis[0];
  }
  if (call->args[0].as<ConstantNode>()) {
    CHECK(constant_.size() == 1) << "Every args expects a single constant_";
    auto constant_input = constant_[0];
    input_name = "gather_input_" + to_string(buf_idx_);
    auto input_qinfo = GetCallOPQuant(call, 0);
    if (cfg->quantization_scheme == "CSINN_QUANT_FLOAT16_W_INT8") {
      input_qinfo->dtype = "int8_t";
      CreateWeightTensor(op, constant_input, input_name, in_shape, input_qinfo);
    } else {
      CreateConstantTensor(op, constant_input, input_name, in_shape, input_qinfo);
    }
    decl << input_name;
  } else {
    CHECK(out_.size() == 1) << "Every args expects a single out_";
    input_name = CreateInputTensor(op, decl, call, 0, GetCallOPQuant(call, 0));
  }

  decl << ", ";

  /* Emit indices tensor */
  visit(call->args[1]);
  string indices_name;
  if (call->args[1].as<ConstantNode>()) {
    CHECK(constant_.size() == 1) << "Every args expects a single constant_";
    auto indices = constant_[0];
    auto indices_shape = call->args[1]->get_shape();
    string indices_name = "indices_" + to_string(buf_idx_);
    CreateConstantTensor(op, indices, indices_name, indices_shape, GetCallOPQuant(call, 1));
    decl << indices_name;
  } else {
    CHECK(out_.size() == 1) << "Every args expects a single out_";
    indices_name = CreateInputTensor(op, decl, call, 1, GetCallOPQuant(call, 1));
  }

  string params_name = "params_" + to_string(buf_idx_);

  /* Emit output tensor */
  auto output_qinfo = GetCallOPQuant(call, 2);
  string output_name = CreateOutputTensor(op, decl, call, output_qinfo);

  collect_quant_info(complete_name, attr->q_params, 2);

  decl << ", " << params_name << ")";
  push_decl(op);
  /* Use gather op */
  malloc_params("csinn_gather_params", params_name);
  if (axis == NULL) {
    t0 << params_name << "->axis = NULL";
  } else {
    t0 << params_name << "->axis = " << axis[0];
  }
  func_def_.PushDecl(t0);

  PushOutput(output_name, call, output_qinfo->dtype);
  params_common_setup(decl, call, "gather", params_name, attr->layer_name.c_str());
  end_stream(decl, "gather");
}

void CodegenCSINN::Clip(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  CSINNOP* op = new CSINNOP;
  const auto* attr = call->attrs.as<QnnCSIClipAttrs>();
  double min = attr->a_min;
  double max = attr->a_max;

  bool use_relu1 = false;
  string op_name = "clip";
  if (min == 0 && max == 1) {
    op_name = "relu1";
    use_relu1 = true;
  }

  SisoOp<QnnCSIClipAttrs>(op, decl, call, attr, op_name);

  if (use_relu1) {
    string params_name = "params_" + to_string(buf_idx_);
    decl << ", " << params_name << ")";
    push_decl(op);
    malloc_params("csinn_relu_params", params_name);
    t0 << params_name << "->n = 1";
    func_def_.PushDecl(t0);

    params_common_setup(decl, call, "relu1", params_name, attr->layer_name.c_str());
    end_stream(decl, "relu1");

  } else {
    string params_name = "params_" + to_string(buf_idx_);
    decl << ", " << params_name << ")";
    push_decl(op);
    malloc_params("csinn_clip_params", params_name);
    t0 << params_name << "->min_value = " << to_string(min);
    func_def_.PushDecl(t0);
    t0 << params_name << "->max_value = " << to_string(max);
    func_def_.PushDecl(t0);

    params_common_setup(decl, call, "clip", params_name, attr->layer_name.c_str());
    end_stream(decl, "clip");
  }
}

void CodegenCSINN::Pad(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  CSINNOP* op = new CSINNOP;
  const auto* attr = call->attrs.as<QnnCSIPadAttrs>();
  auto pad_width = attr->pad_width;
  string pad_mode = attr->pad_mode;

  // Make function call with input buffers when visiting arguments
  decl << "(";

  /* Emit input tensor */
  visit(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string complete_name = get_complete_layer_name("pad", attr->layer_name.c_str());
  op->set_name(complete_name);
  CreateInputTensor(op, decl, call, 0, GetCallOPQuant(call, 0));

  /* Emit output tensor */
  string output_name = CreateOutputTensor(op, decl, call, GetCallOPQuant(call, 2));

  collect_quant_info(complete_name, attr->q_params, 2);

  PushOutput(output_name, call);

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";
  push_decl(op);
  t0 << "int32_t *pad_before_" << buf_idx_ << " = malloc(" << pad_width.size() << " * 4)";
  func_def_.PushDecl(t0);
  for (uint k = 0; k < pad_width.size(); k++) {
    t0 << "pad_before_" << buf_idx_ << "[" << k
       << "] = " << to_string(pad_width[k][0].as<IntImmNode>()->value);
    func_def_.PushDecl(t0);
  }

  t0 << "int32_t *pad_after_" << buf_idx_ << " = malloc(" << pad_width.size() << " * 4)";
  func_def_.PushDecl(t0);
  for (uint k = 0; k < pad_width.size(); k++) {
    t0 << "pad_after_" << buf_idx_ << "[" << k
       << "] = " << to_string(pad_width[k][1].as<IntImmNode>()->value);
    func_def_.PushDecl(t0);
  }

  malloc_params("csinn_pad_params", params_name);
  t0 << params_name << "->pad_before = pad_before_" << buf_idx_;
  func_def_.PushDecl(t0);
  t0 << params_name << "->pad_after = pad_after_" << buf_idx_;
  func_def_.PushDecl(t0);
  if (pad_mode == "constant") {
    t0 << params_name << "->pad_mode = CSINN_PAD_CONSTANT";
  } else {
    t0 << params_name << "->pad_mode = CSINN_PAD_EDGE";
  }

  func_def_.PushDecl(t0);
  visit(call->args[1]);
  auto pad_value = constant_[0];
  float* pad_value_ = reinterpret_cast<float*>(pad_value->get_data_buf());
  /* FIXME: real pad_value in arg[1] */
  t0 << params_name << "->pad_value = " << *pad_value_;
  func_def_.PushDecl(t0);
  t0 << params_name << "->pad_num = " << to_string(pad_width.size());
  func_def_.PushDecl(t0);

  params_common_setup(decl, call, "pad", params_name, attr->layer_name.c_str());
  end_stream(decl, "pad");
}

void CodegenCSINN::Tile(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  CSINNOP* op = new CSINNOP;
  const auto* attr = call->attrs.as<QnnCSITileAttrs>();
  auto reps = attr->reps;
  SisoOp<QnnCSITileAttrs>(op, decl, call, attr, "tile");

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";
  push_decl(op);
  t0 << "int32_t *reps_" << buf_idx_ << " = malloc(" << reps.size() << " * 4)";
  func_def_.PushDecl(t0);
  for (uint k = 0; k < reps.size(); k++) {
    t0 << "reps_" << buf_idx_ << "[" << k << "] = " << reps[k].as<IntImmNode>()->value;
    func_def_.PushDecl(t0);
  }

  malloc_params("csinn_tile_params", params_name);
  t0 << params_name << "->reps = reps_" << buf_idx_;
  func_def_.PushDecl(t0);
  t0 << params_name << "->reps_num = " << reps.size();
  func_def_.PushDecl(t0);

  params_common_setup(decl, call, "tile", params_name, attr->layer_name.c_str());
  end_stream(decl, "tile");
}

void CodegenCSINN::DepthToSpace(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  CSINNOP* op = new CSINNOP;
  const auto* attr = call->attrs.as<QnnCSISubPixelAttrs>();
  int block_size = attr->block_size;
  string mode = attr->mode;
  SisoOp<QnnCSISubPixelAttrs>(op, decl, call, attr, "depth_to_space");

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";
  push_decl(op);
  malloc_params("csinn_depth_to_space_params", params_name);
  t0 << params_name << "->block_size = " << to_string(block_size);
  func_def_.PushDecl(t0);
  if (mode == "DCR") {
    t0 << params_name << "->mode = CSINN_DEPTHTOSPACE_DCR";
  } else if (mode == "CDR") {
    t0 << params_name << "->mode = CSINN_DEPTHTOSPACE_CRD";
  }
  func_def_.PushDecl(t0);

  params_common_setup(decl, call, "depth_to_space", params_name, attr->layer_name.c_str());
  end_stream(decl, "depth_to_space");
}

void CodegenCSINN::SpaceToDepth(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  CSINNOP* op = new CSINNOP;
  const auto* attr = call->attrs.as<QnnCSISubPixelAttrs>();
  int block_size = attr->block_size;
  SisoOp<QnnCSISubPixelAttrs>(op, decl, call, attr, "space_to_depth");

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";
  push_decl(op);
  malloc_params("csinn_space_to_depth_params", params_name);
  t0 << params_name << "->block_size = " << to_string(block_size);
  func_def_.PushDecl(t0);

  params_common_setup(decl, call, "space_to_depth", params_name, attr->layer_name.c_str());
  end_stream(decl, "space_to_depth");
}

void CodegenCSINN::Relu6(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  CSINNOP* op = new CSINNOP;
  const auto* attr = call->attrs.as<QnnCSIUnaryAttrs>();
  SisoOp<QnnCSIUnaryAttrs>(op, decl, call, attr, "relu6");

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";
  push_decl(op);
  malloc_params("csinn_relu_params", params_name);
  t0 << params_name << "->n = 6";
  func_def_.PushDecl(t0);

  params_common_setup(decl, call, "relu6", params_name, attr->layer_name.c_str());
  end_stream(decl, "relu6");
}

void CodegenCSINN::PRelu(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  CSINNOP* op = new CSINNOP;
  const auto* attr = call->attrs.as<QnnCSIPReluAttrs>();
  CHECK(attr);
  CHECK(call->args.size() == 2) << "PRelu expects 2 args";

  /* Make function call with arguments start */
  decl << "(";

  /* Emit_ input tensor */
  visit(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string complete_name = get_complete_layer_name("prelu", attr->layer_name.c_str());
  op->set_name(complete_name);

  auto input_qinfo = GetCallOPQuant(call, 0);
  string input_name = CreateInputTensor(op, decl, call, 0, input_qinfo);

  /* Emit kernel tensor */
  visit(call->args[1]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto alpha = constant_[0];
  auto wshape = call->args[1]->get_shape();
  string alpha_name = "alpha_" + to_string(buf_idx_);
  auto alpha_qinfo = GetCallOPQuant(call, 1);
  CreateConstantTensor(op, alpha, alpha_name, wshape, alpha_qinfo);
  decl << ", " << alpha_name;

  /* Emit output tensor */
  auto output_qinfo = GetCallOPQuant(call, 2);
  string output_name = CreateOutputTensor(op, decl, call, output_qinfo);
  output2params[output_name] = complete_name;

  collect_quant_info(complete_name, attr->q_params, 2);

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";
  push_decl(op);
  malloc_params("csinn_prelu_params", params_name);
  t0 << params_name << "->axis = " << to_string(attr->axis);
  func_def_.PushDecl(t0);

  PushOutput(output_name, call, output_qinfo->dtype);

  params_common_setup(decl, call, "prelu", params_name, attr->layer_name.c_str());
  end_stream(decl, "prelu");
}

void CodegenCSINN::LeakyRelu(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  CSINNOP* op = new CSINNOP;
  const auto* attr = call->attrs.as<QnnCSILeakyReluAttrs>();
  CHECK(attr);
  double alpha = attr->alpha;
  CHECK(call->args.size() == 1) << "LeakyRelu expects 1 args";

  /* Make function call with arguments start */
  decl << "(";

  /* Emit_ input tensor */
  visit(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string complete_name = get_complete_layer_name("leaky_relu", attr->layer_name.c_str());
  op->set_name(complete_name);
  auto input_qinfo = GetCallOPQuant(call, 0);
  string input_name = CreateInputTensor(op, decl, call, 0, input_qinfo);

  /* Emit output tensor */
  auto output_qinfo = GetCallOPQuant(call, 1);
  string output_name = CreateOutputTensor(op, decl, call, output_qinfo);
  output2params[output_name] = complete_name;

  collect_quant_info(complete_name, attr->q_params, 1);

  buf_idx_++;

  int32_t alpha_multiplier;
  int32_t alpha_shift;
  GetMultiplierAndShift(alpha, &alpha_multiplier, &alpha_shift);

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";
  push_decl(op);
  malloc_params("csinn_relu_params", params_name);

  t0 << params_name << "->n = " << to_string(attr->alpha);
  func_def_.PushDecl(t0);
  t0 << params_name << "->n_multiplier = " << to_string(alpha_multiplier);
  func_def_.PushDecl(t0);
  t0 << params_name << "->n_shift = " << to_string(alpha_shift);
  func_def_.PushDecl(t0);

  PushOutput(output_name, call, output_qinfo->dtype);

  params_common_setup(decl, call, "leaky_relu", params_name, attr->layer_name.c_str());
  end_stream(decl, "leaky_relu");
}

void CodegenCSINN::Concat(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  CSINNOP* op = new CSINNOP;
  const auto* attr = call->attrs.as<QnnConcatenateAttrs>();
  CHECK(attr);
  /* Make function call with input buffers when visiting arguments */
  decl << "(";

  /* Emit input tensor */
  visit(call->args[0]);
  auto tuple = call->args[0].as<TupleNode>();
  CHECK(tuple);
  int32_t input_num = tuple->fields.size();

  string input_name = "input_" + to_string(buf_idx_) + "_concat";
  t0 << "struct csinn_tensor *" << input_name << "[" << input_num << "]";
  func_def_.PushDecl(t0);
  string complete_name = get_complete_layer_name("concat", attr->layer_name.c_str());
  op->set_name(complete_name);

  for (int i = 0; i < input_num; i++) {
    std::ostringstream mem_stream;
    if (auto sub_input_node = tuple->fields[i].as<CallNode>()) {
      auto sub_input = GetRealInput(sub_input_node);
      CHECK(sub_input.need_copy == true);
      mem_stream << input_name << "[" << i << "] = " << sub_input.name;
    } else if (auto sub_input_var_node = tuple->fields[i].as<VarNode>()) {
      string var_name =
          InputTensorVar(op, sub_input_var_node, i, *GetCallOPQuant(call, i), cfg->dtype_weight);
      mem_stream << input_name << "[" << i << "] = " << var_name;
    } else if (auto sub_input_item_node = tuple->fields[i].as<TupleGetItemNode>()) {
      string item_name =
          InputTensorTupleItem(sub_input_item_node, *GetCallOPQuant(call, i), cfg->dtype_weight);
      mem_stream << input_name << "[" << i << "] = " << item_name;
    } else {
      auto sub_input_const_node = tuple->fields[i].as<ConstantNode>();
      CHECK(sub_input_const_node);
      CSIConstant* const_out = new CSIConstant(
          GetDtypeString(sub_input_const_node->data.DataType()), sub_input_const_node->get_shape());
      const_out->set_name("constant_" + to_string(const_idx_++));
      sub_input_const_node->data.CopyToBytes(const_out->get_data_buf(), const_out->byte_size());
      auto const_name = const_out->get_name() + "_" + to_string(i);
      auto const_shape = sub_input_const_node->get_shape();
      CreateConstantTensor(op, const_out, const_name, const_shape, GetCallOPQuant(call, i));
      mem_stream << input_name << "[" << i << "] = " << const_name;
    }
    func_def_.PushCall(mem_stream);
  }
  decl << input_name;

  /* Emit output tensor */
  auto output_qinfo = GetCallOPQuant(call, attr->q_params.size() - 1);
  string output_name = CreateOutputTensor(op, decl, call, output_qinfo);

  collect_quant_info(complete_name, attr->q_params, attr->q_params.size() - 1);

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";
  push_decl(op);
  malloc_params("csinn_concat_params", params_name);

  t0 << params_name << "->inputs_count = " << to_string(input_num);
  func_def_.PushDecl(t0);
  t0 << params_name << "->axis = " << to_string(attr->axis);
  func_def_.PushDecl(t0);
  PushOutput(output_name, call, output_qinfo->dtype);

  params_common_setup(decl, call, "concat", params_name, attr->layer_name.c_str());
  end_stream(decl, "concat");
}

void CodegenCSINN::LRN(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  CSINNOP* op = new CSINNOP;
  const auto* attr = call->attrs.as<QnnCSILRNAttrs>();
  SisoOp<QnnCSILRNAttrs>(op, decl, call, attr, "lrn");
  int32_t multiplier, shift;

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";
  push_decl(op);
  malloc_params("csinn_lrn_params", params_name);

  /* range */
  t0 << params_name << "->range = " << to_string(attr->size);
  func_def_.PushDecl(t0);
  t0 << params_name << "->norm_region = CSINN_LRN_" << attr->norm_region;
  func_def_.PushDecl(t0);
  /* bias */
  GetMultiplierAndShift(attr->bias, &multiplier, &shift);
  t0 << params_name << "->bias = " << to_string(attr->bias);
  func_def_.PushDecl(t0);
  t0 << params_name << "->bias_multiplier = " << to_string(multiplier);
  func_def_.PushDecl(t0);
  t0 << params_name << "->bias_shift = " << to_string(shift);
  func_def_.PushDecl(t0);

  /* alpha */
  GetMultiplierAndShift(attr->alpha, &multiplier, &shift);
  t0 << params_name << "->alpha = " << to_string(attr->alpha);
  func_def_.PushDecl(t0);
  t0 << params_name << "->alpha_multiplier = " << to_string(multiplier);
  func_def_.PushDecl(t0);
  t0 << params_name << "->alpha_shift = " << to_string(shift);
  func_def_.PushDecl(t0);

  /* beta */
  GetMultiplierAndShift(attr->beta, &multiplier, &shift);
  t0 << params_name << "->beta = " << to_string(attr->beta);
  func_def_.PushDecl(t0);
  t0 << params_name << "->beta_multiplier = " << to_string(multiplier);
  func_def_.PushDecl(t0);
  t0 << params_name << "->beta_shift = " << to_string(shift);
  func_def_.PushDecl(t0);

  params_common_setup(decl, call, "lrn", params_name, attr->layer_name.c_str());
  end_stream(decl, "lrn");
}

void CodegenCSINN::Flatten(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  CSINNOP* op = new CSINNOP;
  const auto* attr = call->attrs.as<QnnCSIUnaryAttrs>();
  CHECK(attr);
  SisoOp<QnnCSIUnaryAttrs>(op, decl, call, attr, "flatten");

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";
  push_decl(op);
  malloc_params("csinn_flatten_params", params_name);

  params_common_setup(decl, call, "flatten", params_name, attr->layer_name.c_str());
  end_stream(decl, "flatten");
}

void CodegenCSINN::Sigmoid(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  CSINNOP* op = new CSINNOP;
  const auto* attr = call->attrs.as<QnnCSIUnaryAttrs>();
  CHECK(attr);
  SisoOp<QnnCSIUnaryAttrs>(op, decl, call, attr, "sigmoid");

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";
  push_decl(op);
  malloc_params("csinn_sigmoid_params", params_name);

  params_common_setup(decl, call, "sigmoid", params_name, attr->layer_name.c_str());
  end_stream(decl, "sigmoid");
}

void CodegenCSINN::Transpose(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream pstream;
  std::ostringstream t0;
  CSINNOP* op = new CSINNOP;
  const auto* attrs = call->attrs.as<QnnCSITransposeAttrs>();
  CHECK(attrs);

  CHECK(call->args.size() == 1) << "Transpose expects 1 args";

  decl << "(";

  /* Emit_ input tensor */
  visit(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string complete_name = get_complete_layer_name("transpose", attrs->layer_name.c_str());
  op->set_name(complete_name);
  string input_name = CreateInputTensor(op, decl, call, 0, GetCallOPQuant(call, 0));

  string perm_name = "permute_" + to_string(buf_idx_);
  int32_t perm_size = attrs->axes.size();

  t0 << "int32_t *" << perm_name << " = malloc(" << perm_size << " * 4)";
  func_def_.PushDecl(t0);
  for (int i = 0; i < perm_size; i++) {
    t0 << perm_name << "[" << i << "] = " << to_string(attrs->axes[i].as<IntImmNode>()->value);
    func_def_.PushDecl(t0);
  }

  auto output_qinfo = GetCallOPQuant(call, 1);
  string output_name = CreateOutputTensor(op, decl, call, output_qinfo);

  collect_quant_info(complete_name, attrs->q_params, 1);

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";
  push_decl(op);
  malloc_params("csinn_transpose_params", params_name);
  t0 << params_name << "->permute = " << perm_name;
  func_def_.PushDecl(t0);
  t0 << params_name << "->permute_num = " << to_string(perm_size);
  func_def_.PushDecl(t0);

  PushOutput(output_name, call, output_qinfo->dtype);
  params_common_setup(decl, call, "transpose", params_name, attrs->layer_name.c_str());
  end_stream(decl, "transpose");
}

void CodegenCSINN::Reshape(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  CSINNOP* op = new CSINNOP;
  const auto* attr = call->attrs.as<QnnCSIReshapeAttrs>();
  SisoOp<QnnCSIReshapeAttrs>(op, decl, call, attr, "reshape");

  auto out_shape = attr->newshape;
  string new_shape_name = "shape_" + to_string(buf_idx_);
  int32_t new_shape_dim_num = out_shape.size();
  t0 << "int32_t *" << new_shape_name << " = malloc(" << new_shape_dim_num << " * 4)";
  func_def_.PushDecl(t0);
  for (int i = 0; i < new_shape_dim_num; i++) {
    t0 << new_shape_name << "[" << i << "] = " << to_string(out_shape[i]);
    func_def_.PushDecl(t0);
  }

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";
  push_decl(op);
  malloc_params("csinn_reshape_params", params_name);

  t0 << params_name << "->shape = " << new_shape_name;
  func_def_.PushDecl(t0);
  t0 << params_name << "->shape_num = " << new_shape_dim_num;
  func_def_.PushDecl(t0);
  params_common_setup(decl, call, "reshape", params_name, attr->layer_name.c_str());

  end_stream(decl, "reshape");
}

void CodegenCSINN::BroadCastTo(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  CSINNOP* op = new CSINNOP;
  const auto* attr = call->attrs.as<QnnCSIBroadCastToAttrs>();
  SisoOp<QnnCSIBroadCastToAttrs>(op, decl, call, attr, "broadcast_to");

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";
  push_decl(op);
  malloc_params("csinn_broadcast_to_params", params_name);
  t0 << params_name << "->shape = malloc(" << attr->shape.size() << " * 4)";
  func_def_.PushDecl(t0);
  for (uint i = 0; i < attr->shape.size(); i++) {
    t0 << params_name << "->shape[" << i
       << "] = " << to_string(attr->shape[i].as<IntImmNode>()->value);
    func_def_.PushDecl(t0);
  }
  t0 << params_name << "->shape_count = " << attr->shape.size();
  func_def_.PushDecl(t0);

  params_common_setup(decl, call, "broadcast_to", params_name, attr->layer_name.c_str());
  end_stream(decl, "broadcast_to");
}

void CodegenCSINN::Squeeze(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  CSINNOP* op = new CSINNOP;
  const auto* attr = call->attrs.as<QnnCSISqueezeAttrs>();
  SisoOp<QnnCSISqueezeAttrs>(op, decl, call, attr, "squeeze");

  string squeeze_axis_name = "squeeze_aixs_" + to_string(buf_idx_);
  int32_t squeeze_axis_dim_num = attr->axis.size();
  t0 << "int32_t " << squeeze_axis_name << "[" << squeeze_axis_dim_num << "] = {";
  for (int i = 0; i < squeeze_axis_dim_num; i++) {
    t0 << to_string(attr->axis[i].as<IntImmNode>()->value) << ", ";
  }
  t0 << "}";
  func_def_.PushDecl(t0);

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";
  push_decl(op);
  malloc_params("csinn_squeeze_params", params_name);
  t0 << params_name << "->axis = " << squeeze_axis_name;
  func_def_.PushDecl(t0);
  t0 << params_name << "->axis_num = " << squeeze_axis_dim_num;
  func_def_.PushDecl(t0);
  params_common_setup(decl, call, "squeeze", params_name, attr->layer_name.c_str());
  end_stream(decl, "squeeze");
}

void CodegenCSINN::Segment(const CallNode* call, string name) {
  const auto* attr = call->attrs.as<QnnCSISegmentAttrs>();
  CHECK(attr);

  std::ostringstream decl;
  std::ostringstream t0;
  CSINNOP* op = new CSINNOP;

  CHECK(call->args.size() == 2) << "op expects 2 args";

  // Make function call with input buffers when visiting arguments
  decl << "(";

  /* Emit input tensor */
  visit(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string complete_name = get_complete_layer_name("segment_" + name, attr->layer_name.c_str());
  op->set_name(complete_name);

  auto input_qinfo = GetCallOPQuant(call, 0);
  string input_name = CreateInputTensor(op, decl, call, 0, input_qinfo);

  /* Emit idx tensor */
  visit(call->args[1]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto idx = constant_[0];
  auto ishape = call->args[1]->get_shape();
  string idx_name = "idx_" + to_string(buf_idx_);
  CreateConstantTensor(op, idx, idx_name, ishape, GetCallOPQuant(call, 1));
  decl << ", " << idx_name;

  /* Emit output tensor */
  auto output_qinfo = GetCallOPQuant(call, 2);
  string output_name = CreateOutputTensor(op, decl, call, output_qinfo);
  output2params[output_name] = complete_name;

  collect_quant_info(complete_name, attr->q_params, 2);

  PushOutput(output_name, call, output_qinfo->dtype);

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";
  push_decl(op);
  malloc_params("csinn_segment_params", params_name);
  t0 << params_name << "->axis = " << attr->num_segments;
  func_def_.PushDecl(t0);

  params_common_setup(decl, call, "segment_" + name, params_name, attr->layer_name.c_str());
  end_stream(decl, "segment_" + name);
}

void CodegenCSINN::ScatterND(const CallNode* call) {
  const auto* attr = call->attrs.as<QnnCSIUnaryAttrs>();
  CHECK(attr);

  std::ostringstream decl;
  std::ostringstream t0;
  CSINNOP* op = new CSINNOP;

  CHECK(call->args.size() == 3) << "op expects 3 args";

  // Make function call with input buffers when visiting arguments
  decl << "(";

  /* Emit input tensor */
  visit(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string complete_name = get_complete_layer_name("scatter_nd", attr->layer_name.c_str());
  op->set_name(complete_name);
  string input_name = CreateInputTensor(op, decl, call, 0, GetCallOPQuant(call, 0));

  /* Emit idx tensor */
  visit(call->args[1]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto idx = constant_[0];
  auto ishape = call->args[1]->get_shape();
  string idx_name = "idx_" + to_string(buf_idx_);
  CreateConstantTensor(op, idx, idx_name, ishape, GetCallOPQuant(call, 1));
  decl << ", " << idx_name << ", ";

  visit(call->args[2]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string updates_name = CreateInputTensor(op, decl, call, 2, GetCallOPQuant(call, 2));

  /* Emit output tensor */
  string output_name = CreateOutputTensor(op, decl, call, GetCallOPQuant(call, 3));

  collect_quant_info(complete_name, attr->q_params, 3);

  PushOutput(output_name, call);

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";
  push_decl(op);
  malloc_params("csinn_scatter_nd_params", params_name);

  params_common_setup(decl, call, "scatter_nd", params_name, attr->layer_name.c_str());
  end_stream(decl, "scatter_nd");
}

void CodegenCSINN::Reduce(const CallNode* call, string name, string out_dtype) {
  std::ostringstream decl;
  std::ostringstream t0;
  CSINNOP* op = new CSINNOP;

  const auto* attr = call->attrs.as<QnnCSIReduceAttrs>();
  CHECK(attr);
  // x86 reference
  auto axis = attr->axis;

  auto input_shape = call->args[0]->get_shape();

  CHECK(call->args.size() == 1) << name << " expects 1 args";
  auto out_shape = call->get_shape();
  visit(call->args[0]);

  string complete_name = get_complete_layer_name(name, attr->layer_name.c_str());
  op->set_name(complete_name);

  CHECK(out_.size() == 1) << "Every args expects a single out_";
  decl << "(";
  auto input_qinfo = GetCallOPQuant(call, 0);
  string input_name = CreateInputTensor(op, decl, call, 0, input_qinfo);
  auto output_qinfo = GetCallOPQuant(call, 1);
  string output_name = CreateOutputTensor(op, decl, call, output_qinfo);

  collect_quant_info(complete_name, attr->q_params, 1);

  output2params[output_name] = complete_name;

  std::vector<int> out_extents;
  std::vector<int> out_strides;
  std::vector<int> inner_extents;
  std::vector<int> inner_strides;

  auto reduce_axes = __get_real_axis(input_shape.size(), axis);
  for (uint i = 0; i < input_shape.size(); i++) {
    int flag = 0;
    for (uint j = 0; j < reduce_axes.size(); j++) {
      uint tmp = reduce_axes[j];
      if (i == tmp) {
        flag = 1;
      }
    }
    if (flag) {
      inner_extents.push_back(input_shape[i]);
      int stride = __get_stride(i, input_shape);
      inner_strides.push_back(stride);
    } else {
      out_extents.push_back(input_shape[i]);
      int stride = __get_stride(i, input_shape);
      out_strides.push_back(stride);
    }
  }

  t0 << "int32_t *out_strides_" << buf_idx_ << " = malloc(" << out_strides.size() << " * 4)";
  func_def_.PushDecl(t0);
  t0 << "int32_t *out_extents_" << buf_idx_ << " = malloc(" << out_extents.size() << " * 4)";
  func_def_.PushDecl(t0);
  for (uint i = 0; i < out_strides.size(); i++) {
    t0 << "out_strides_" << buf_idx_ << "[" << i << "] = " << to_string(out_strides[i]);
    func_def_.PushDecl(t0);
  }
  for (uint i = 0; i < out_extents.size(); i++) {
    t0 << "out_extents_" << buf_idx_ << "[" << i << "] = " << to_string(out_extents[i]);
    func_def_.PushDecl(t0);
  }

  t0 << "int32_t *inner_strides_" << buf_idx_ << " = malloc(" << inner_strides.size() << " * 4)";
  func_def_.PushDecl(t0);
  t0 << "int32_t *inner_extents_" << buf_idx_ << " = malloc(" << inner_extents.size() << " * 4)";
  func_def_.PushDecl(t0);
  for (uint i = 0; i < inner_strides.size(); i++) {
    t0 << "inner_strides_" << buf_idx_ << "[" << i << "] = " << to_string(inner_strides[i]);
    func_def_.PushDecl(t0);
  }
  for (uint i = 0; i < inner_extents.size(); i++) {
    t0 << "inner_extents_" << buf_idx_ << "[" << i << "] = " << to_string(inner_extents[i]);
    func_def_.PushDecl(t0);
  }

  t0 << "int32_t *aixs_" << buf_idx_ << " = malloc(" << axis.size() << " * 4)";
  func_def_.PushDecl(t0);
  for (uint i = 0; i < axis.size(); i++) {
    t0 << "aixs_" << buf_idx_ << "[" << i << "] = " << to_string(axis[i].as<IntImmNode>()->value);
    func_def_.PushDecl(t0);
  }

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";
  if (name == "argmax" || name == "argmin") {
    CSINNTensor* i32out = op->get_tensor(output_name);
    i32out->tensor->dtype = CSINN_DTYPE_INT32;
  }
  push_decl(op);
  malloc_params("csinn_reduce_params", params_name);
  t0 << params_name << "->out_strides = out_strides_" << buf_idx_;
  func_def_.PushDecl(t0);
  t0 << params_name << "->out_extents = out_extents_" << buf_idx_;
  func_def_.PushDecl(t0);
  t0 << params_name << "->n = " << to_string(out_extents.size());
  func_def_.PushDecl(t0);
  t0 << params_name << "->inner_strides = inner_strides_" << buf_idx_;
  func_def_.PushDecl(t0);
  t0 << params_name << "->inner_extents = inner_extents_" << buf_idx_;
  func_def_.PushDecl(t0);
  t0 << params_name << "->m = " << to_string(inner_extents.size());
  func_def_.PushDecl(t0);
  t0 << params_name << "->axis = aixs_" << buf_idx_;
  func_def_.PushDecl(t0);
  t0 << params_name << "->axis_count = " << axis.size();
  func_def_.PushDecl(t0);
  auto str_keepdims = attr->keepdims ? "true" : "false";
  t0 << params_name << "->keepdims = " << str_keepdims;
  func_def_.PushDecl(t0);
  if (name == "argmax" || name == "argmin") {
    PushOutput(output_name, call, "int32_t");
  } else {
    PushOutput(output_name, call, output_qinfo->dtype);
  }

  params_common_setup(decl, call, name, params_name, attr->layer_name.c_str());
  end_stream(decl, name);
}

void CodegenCSINN::CropResize(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  CSINNOP* op = new CSINNOP;
  const auto* attr = call->attrs.as<QnnCSICropResizeAttrs>();
  CHECK(attr);

  auto crop_size = attr->crop_size;
  auto method = attr->method;
  auto extrapolation_value = attr->extrapolation_value;
  CHECK(call->args.size() == 3) << "CropResize expects 3 args";
  /* Make function call with arguments start */
  decl << "(";

  /* Emit_ input tensor */
  visit(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string complete_name = get_complete_layer_name("crop_resize", attr->layer_name.c_str());
  op->set_name(complete_name);
  string input_name = CreateInputTensor(op, decl, call, 0, GetCallOPQuant(call, 0));

  /* Emit output tensor */
  string output_name = CreateOutputTensor(op, decl, call, GetCallOPQuant(call, 3));

  collect_quant_info(complete_name, attr->q_params, 3);

  /* Emit boxes tensor */
  visit(call->args[1]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto boxes = constant_[0];
  auto bshape = call->args[1]->get_shape();
  string boxes_name = "boxes_" + to_string(buf_idx_);
  CreateConstantTensor(op, boxes, boxes_name, bshape, GetCallOPQuant(call, 1));
  decl << ", " << boxes_name;

  /* Emit bias tensor */
  visit(call->args[2]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto box_indices = constant_[0];
  auto index_shape = call->args[2]->get_shape();
  string index_name = "index_" + to_string(buf_idx_);
  CreateConstantTensor(op, box_indices, index_name, index_shape, GetCallOPQuant(call, 2));
  decl << ", " << index_name;

  string params_name = "params_" + to_string(buf_idx_);

  decl << ", " << params_name << ")";
  push_decl(op);
  t0 << "  int32_t crop_size_" << buf_idx_ << "[" << crop_size.size() << "] = {";
  for (uint i = 0; i < crop_size.size(); i++) {
    t0 << Downcast<IndexExpr>(crop_size[i]) << ", ";
  }
  t0 << "}";
  func_def_.PushDecl(t0);
  malloc_params("csinn_crop_resize_params", params_name);

  t0 << params_name << "->method = " << method;
  func_def_.PushDecl(t0);
  t0 << params_name << "->extrapolation_value = " << extrapolation_value;
  func_def_.PushDecl(t0);
  t0 << params_name << "->crop_size = crop_size_" << buf_idx_;
  func_def_.PushDecl(t0);

  PushOutput(output_name, call);
  params_common_setup(decl, call, "crop_resize", params_name, attr->layer_name.c_str());
  end_stream(decl, "crop_resize");
}

void CodegenCSINN::StridedSlice(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0, t1;
  CSINNOP* op = new CSINNOP;
  const auto* attr = call->attrs.as<QnnCSIStridedSliceAttrs>();
  CHECK(attr);
  // x86 reference
  auto begin = attr->begin;
  auto end = attr->end;
  auto strides = attr->strides;

  CHECK(call->args.size() == 1) << "strided slic expects 1 args";
  visit(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";

  decl << "(";
  string complete_name = get_complete_layer_name("strided_slice", attr->layer_name.c_str());
  op->set_name(complete_name);
  string input_name = CreateInputTensor(op, decl, call, 0, GetCallOPQuant(call, 0));

  string output_name = CreateOutputTensor(op, decl, call, GetCallOPQuant(call, 1));

  collect_quant_info(complete_name, attr->q_params, 1);

  t0 << "int32_t *begin_" << buf_idx_ << " = malloc(" << begin.size() << " * 4)";
  func_def_.PushDecl(t0);
  t0 << "int32_t *end_" << buf_idx_ << " = malloc(" << end.size() << " * 4)";
  func_def_.PushDecl(t0);
  for (uint i = 0; i < begin.size(); i++) {
    t0 << "begin_" << buf_idx_ << "[" << i << "] = " << to_string(begin[i]);
    func_def_.PushDecl(t0);
  }
  auto ishape = call->args[0]->get_shape();
  for (uint i = 0; i < end.size(); i++) {
    int end_ =
        end[i].as<IntImmNode>()->value > ishape[i] ? ishape[i] : end[i].as<IntImmNode>()->value;
    t0 << "end_" << buf_idx_ << "[" << i << "] = " << to_string(end_);
    func_def_.PushDecl(t0);
  }

  uint stride_size = strides.size();
  if (stride_size == 1) {
    stride_size = begin.size();
  }

  t0 << "int32_t *strides_" << buf_idx_ << " = malloc(" << stride_size << " * 4)";
  func_def_.PushDecl(t0);

  for (uint i = 0; i < stride_size; i++) {
    if (i < strides.size()) {
      t0 << "strides_" << buf_idx_ << "[" << i << "] = " << to_string(strides[i]);
      func_def_.PushDecl(t0);
    } else {
      t0 << "strides_" << buf_idx_ << "[" << i << "] = " << to_string(strides[0]);
      func_def_.PushDecl(t0);
    }
  }

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";
  push_decl(op);
  malloc_params("csinn_strided_slice_params", params_name);
  t0 << params_name << "->begin = begin_" << buf_idx_;
  func_def_.PushDecl(t0);
  t0 << params_name << "->end = end_" << buf_idx_;
  func_def_.PushDecl(t0);
  t0 << params_name << "->stride = strides_" << buf_idx_;
  func_def_.PushDecl(t0);
  t0 << params_name << "->slice_count = " << begin.size();
  func_def_.PushDecl(t0);

  PushOutput(output_name, call);
  params_common_setup(decl, call, "strided_slice", params_name, attr->layer_name.c_str());
  end_stream(decl, "strided_slice");
}

void CodegenCSINN::Split(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  CSINNOP* op = new CSINNOP;
  const auto* attr = call->attrs.as<QnnCSISplitAttrs>();
  CHECK(attr);

  // x86 reference
  auto axis = attr->axis;

  CHECK(call->args.size() == 1) << "strided slic expects 1 args";
  visit(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";

  string out_name = "output_" + to_string(buf_idx_);
  t0 << "struct csinn_tensor *" << out_name << "[" << attr->q_params.size() - 1 << "]";
  func_def_.PushDecl(t0);
  decl << "(";
  string complete_name = get_complete_layer_name("split", attr->layer_name.c_str());
  op->set_name(complete_name);
  string input_name = CreateInputTensor(op, decl, call, 0, GetCallOPQuant(call, 0));
  auto in_shape = call->args[0]->get_shape();

  decl << ", " << out_name << ", ";
  string params_name = "params_" + to_string(buf_idx_);
  decl << params_name << ")";

  std::vector<string> out_names;
  for (uint i = 0; i < attr->q_params.size() - 1; i++) {
    std::vector<int> out_shape;
    if (call->get_next_expr(i)) {
      out_shape = call->get_next_expr(i)->get_shape();
    } else {
      LOG(WARNING) << "split out unused";
      out_shape = in_shape;
    }

    string out = "alloc_" + to_string(alloc_idx_);

    int out_size = 1;
    for (size_t j = 0; j < out_shape.size(); ++j) {
      out_size *= out_shape[j];
    }
    if (cfg->dtype_weight == "float16" || cfg->dtype_weight == "bfloat16" ||
        cfg->dtype_weight == "int16_t") {
      out_size = out_size * 2;
    } else if (cfg->dtype_weight == "float" || cfg->dtype_weight == "float32") {
      out_size *= 4;
    }
    alloc_idx_++;
    string output_name = "output_" + to_string(buf_idx_) + "_" + to_string(i);
    CSINNVarTensor* ret =
        CreateTensor(output_name, out, out_shape, *GetCallOPQuant(call, i + 1), cfg->dtype_weight);

    collect_quant_info(complete_name, attr->q_params, i + 1);

    t0 << out_name << "[" << to_string(i) << "] = " << output_name;
    ret->append_str(t0);
    op->push_output(ret);
    out_names.push_back(output_name);
  }
  push_decl(op);
  Array<Integer> indices_or_sections;
  if (const IntImmNode* sections = attr->indices_or_sections.as<IntImmNode>()) {
    axis = axis == -1 ? in_shape.size() - 1 : axis;
    t0 << "int32_t axis_len" << buf_idx_ << " = ";
    t0 << input_name << "->dim[" << axis << "]";
    func_def_.PushDecl(t0);

    t0 << "int32_t index_" << buf_idx_ << " = ";
    t0 << "axis_len" << buf_idx_ << " / " << sections->value;
    func_def_.PushDecl(t0);

    t0 << "int32_t *indices_or_sections_" << buf_idx_ << " = malloc(sizeof(int32_t) * "
       << sections->value - 1 << ")";
    func_def_.PushDecl(t0);
    for (int x = 1; x < sections->value; x++) {
      t0 << "indices_or_sections_" << buf_idx_ << "[" << x - 1 << "] = index_" << buf_idx_ << " * "
         << x;
      func_def_.PushDecl(t0);
    }
  } else {
    auto indices_ = Downcast<Array<ObjectRef>>(attr->indices_or_sections);
    t0 << "int32_t *indices_or_sections_" << buf_idx_ << " = malloc(sizeof(int32_t) * "
       << indices_.size() << ")";
    func_def_.PushDecl(t0);
    for (uint32_t k = 0; k < indices_.size(); k++) {
      auto idx = Downcast<IndexExpr>(indices_[k]);
      t0 << "indices_or_sections_" << buf_idx_ << "[" << k
         << "] = " << to_string(*tir::as_const_int(idx)) << ";";
      func_def_.PushDecl(t0);
    }
  }

  malloc_params("csinn_split_params", params_name);
  t0 << params_name << "->split_index = indices_or_sections_" << buf_idx_;
  func_def_.PushDecl(t0);
  t0 << params_name << "->output_num = " << attr->q_params.size() - 1;
  func_def_.PushDecl(t0);
  t0 << params_name << "->axis = " << to_string(axis);
  func_def_.PushDecl(t0);

  PushOutput(out_names, call);
  params_common_setup(decl, call, "split", params_name, attr->layer_name.c_str());
  end_stream(decl, "split");
}

void CodegenCSINN::BatchToSpaceND(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream pstream;
  std::ostringstream t0;
  CSINNOP* op = new CSINNOP;
  const auto* attrs = call->attrs.as<QnnCSIBatchToSpaceNDAttrs>();
  CHECK(attrs);

  CHECK(call->args.size() == 1) << "BatchToSpaceND expects 1 args";

  decl << "(";

  /* Emit_ input tensor */
  visit(call->args[0]);

  string complete_name = get_complete_layer_name("batch_to_space_nd", attrs->layer_name.c_str());
  op->set_name(complete_name);

  CHECK(out_.size() == 1) << "Every args expects a single out_";
  auto input_qinfo = GetCallOPQuant(call, 0);
  string input_name = CreateInputTensor(op, decl, call, 0, input_qinfo);

  string block_shape_name = "block_shape_" + to_string(buf_idx_);
  string crops_name = "crops_" + to_string(buf_idx_);
  int32_t spatial_dim_cnt = attrs->block_shape.size();

  // Emit block shape
  t0 << "int32_t " << block_shape_name << "[" << spatial_dim_cnt << "] = {";
  for (int i = 0; i < spatial_dim_cnt; i++) {
    t0 << to_string(attrs->block_shape[i].as<IntImmNode>()->value) << ", ";
  }
  t0 << "}";
  func_def_.PushDecl(t0);

  // Emit crops
  t0 << "int32_t " << crops_name << "[" << spatial_dim_cnt * 2 << "] = {";
  for (int i = 0; i < spatial_dim_cnt; i++) {
    t0 << to_string(attrs->crops[i][0].as<IntImmNode>()->value) << ", ";
    t0 << to_string(attrs->crops[i][1].as<IntImmNode>()->value) << ", ";
  }
  t0 << "}";
  func_def_.PushDecl(t0);
  auto output_qinfo = GetCallOPQuant(call, 1);
  string output_name = CreateOutputTensor(op, decl, call, output_qinfo);

  collect_quant_info(complete_name, attrs->q_params, 1);

  string params_name = "params_" + to_string(buf_idx_);
  output2params[output_name] = complete_name;
  decl << ", " << params_name << ")";
  push_decl(op);
  malloc_params("csinn_batch_to_space_nd_params", params_name);

  t0 << params_name << "->crops = " << crops_name;
  func_def_.PushDecl(t0);
  t0 << params_name << "->block_shape = " << block_shape_name;
  func_def_.PushDecl(t0);
  t0 << params_name << "->spatial_dim_cnt = " << to_string(spatial_dim_cnt);
  func_def_.PushDecl(t0);

  PushOutput(output_name, call, output_qinfo->dtype);

  params_common_setup(decl, call, "batch_to_space_nd", params_name, attrs->layer_name.c_str());
  end_stream(decl, "batch_to_space_nd");
}

void CodegenCSINN::SpaceToBatchND(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream pstream;
  std::ostringstream t0;
  CSINNOP* op = new CSINNOP;
  const auto* attrs = call->attrs.as<QnnCSISpaceToBatchNDAttrs>();
  CHECK(attrs);

  CHECK(call->args.size() == 1) << "SpaceToBatchND expects 1 args";

  decl << "(";

  /* Emit_ input tensor */
  visit(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string complete_name = get_complete_layer_name("space_to_batch_nd", attrs->layer_name.c_str());
  op->set_name(complete_name);
  string input_name = CreateInputTensor(op, decl, call, 0, GetCallOPQuant(call, 0));

  string block_shape_name = "block_shape_" + to_string(buf_idx_);
  string paddings_name = "paddings_" + to_string(buf_idx_);
  int32_t spatial_dim_cnt = attrs->block_shape.size();

  // Emit block shape
  t0 << "int32_t " << block_shape_name << "[" << spatial_dim_cnt << "] = {";
  for (int i = 0; i < spatial_dim_cnt; i++) {
    t0 << to_string(attrs->block_shape[i].as<IntImmNode>()->value) << ", ";
  }
  t0 << "}";
  func_def_.PushDecl(t0);

  // Emit paddings
  t0 << "int32_t " << paddings_name << "[" << spatial_dim_cnt * 2 << "] = {";
  for (int i = 0; i < spatial_dim_cnt; i++) {
    t0 << to_string(attrs->paddings[i][0].as<IntImmNode>()->value) << ", ";
    t0 << to_string(attrs->paddings[i][1].as<IntImmNode>()->value) << ", ";
  }
  t0 << "}";
  func_def_.PushDecl(t0);

  string output_name = CreateOutputTensor(op, decl, call, GetCallOPQuant(call, 1));

  collect_quant_info(complete_name, attrs->q_params, 1);

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";
  push_decl(op);
  malloc_params("csinn_space_to_batch_nd_params", params_name);

  t0 << params_name << "->paddings = " << paddings_name;
  func_def_.PushDecl(t0);
  t0 << params_name << "->block_shape = " << block_shape_name;
  func_def_.PushDecl(t0);
  t0 << params_name << "->spatial_dim_cnt = " << to_string(spatial_dim_cnt);
  func_def_.PushDecl(t0);

  PushOutput(output_name, call);
  params_common_setup(decl, call, "space_to_batch_nd", params_name, attrs->layer_name.c_str());
  end_stream(decl, "space_to_batch_nd");
}

void CodegenCSINN::MatMul(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  CSINNOP* op = new CSINNOP;
  const auto* attrs = call->attrs.as<QnnCSIMatMulAttrs>();
  CHECK(attrs);

  CHECK(call->args.size() == 3) << "Dense expects 2 args";

  // Make function call with input buffers when visiting arguments
  decl << "(";

  /* Emit input tensor */
  visit(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string complete_name = get_complete_layer_name("matmul", attrs->layer_name.c_str());
  op->set_name(complete_name);

  auto input_qinfo = GetCallOPQuant(call, 0);
  string input1_name = CreateInputTensor(op, decl, call, 0, input_qinfo);
  buf_idx_++;
  decl << ", ";
  string input2_name;

  /* Emit input tensor */
  visit(call->args[1]);
  auto input2_qinfo = GetCallOPQuant(call, 1);
  if (call->args[1].as<CallNode>() || call->args[1].as<VarNode>() ||
      call->args[1].as<TupleGetItemNode>()) {
    CHECK(out_.size() == 1) << "Every args expects a single out_";
    input2_name = CreateInputTensor(op, decl, call, 1, input2_qinfo);
  } else {
    // add constant arg
    CHECK(constant_.size() == 1) << "Every args expects a single constant_";
    auto data_b = constant_[0];
    auto b_shape = call->args[1]->get_shape();
    input2_name = "data_b_" + to_string(buf_idx_);
    auto input2_qinfo = GetCallOPQuant(call, 1);
    if (cfg->quantization_scheme == "CSINN_QUANT_FLOAT16_W_INT8") {
      input2_qinfo->dtype = "int8_t";
      CreateWeightTensor(op, data_b, input2_name, b_shape, input2_qinfo);
    } else {
      CreateConstantTensor(op, data_b, input2_name, b_shape, input2_qinfo);
    }
    decl << input2_name;
  }

  /* Emit output tensor */
  auto output_qinfo = GetCallOPQuant(call, 3);
  string output_name = CreateOutputTensor(op, decl, call, output_qinfo);
  output2params[output_name] = complete_name;

  collect_quant_info(complete_name, attrs->q_params, 2);

  string params_name = "params_" + to_string(buf_idx_);

  decl << ", " << params_name << ")";
  push_decl(op);
  malloc_params("csinn_matmul_params", params_name);
  string transpose_a = attrs->transpose_a ? "true" : "false";
  string transpose_b = attrs->transpose_b ? "true" : "false";
  t0 << params_name << "->trans_a = " << transpose_a;
  func_def_.PushDecl(t0);
  t0 << params_name << "->trans_b = " << transpose_b;
  func_def_.PushDecl(t0);

  PushOutput(output_name, call, output_qinfo->dtype);

  params_common_setup(decl, call, "matmul", params_name, attrs->layer_name.c_str());
  end_stream(decl, "matmul");
}

void CodegenCSINN::CacheMatMul(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  CSINNOP* op = new CSINNOP;
  const auto* attrs = call->attrs.as<QnnCSICacheMatMulAttrs>();
  CHECK(attrs);

  CHECK(call->args.size() == 3) << "CacheMatMul expects 3 args";

  // Make function call with input buffers when visiting arguments
  decl << "(";

  /* Emit input tensor */
  visit(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string complete_name = get_complete_layer_name("cache_matmul", attrs->layer_name.c_str());
  op->set_name(complete_name);
  string input_name = CreateInputTensor(op, decl, call, 0, GetCallOPQuant(call, 0));

  /* Emit output tensor */
  string output_name = CreateOutputTensor(op, decl, call, GetCallOPQuant(call, 3));

  collect_quant_info(complete_name, attrs->q_params, 3);

  /* Emit weight tensor */
  visit(call->args[1]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto weight = constant_[0];
  auto weight_shape = call->args[1]->get_shape();

  string weight_name = "weight_" + to_string(buf_idx_);
  CreateConstantTensor(op, weight, weight_name, weight_shape, GetCallOPQuant(call, 1));

  decl << ", " << weight_name;

  /* Emit bias tensor */
  visit(call->args[2]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto bias = constant_[0];
  auto bshape = call->args[2]->get_shape();
  string bias_name = "bias_" + to_string(buf_idx_);
  CreateConstantTensor(op, bias, bias_name, bshape, cfg->dtype_activation, GetCallOPQuant(call, 0),
                       GetCallOPQuant(call, 1), GetCallOPQuant(call, 2));

  decl << ", " << bias_name;

  string params_name = "params_" + to_string(buf_idx_);

  decl << ", " << params_name << ")";
  push_decl(op);
  malloc_params("csinn_cache_matmul_params", params_name);

  string cache_shape_name = "cache_shape_" + to_string(buf_idx_);
  t0 << "int32_t *" << cache_shape_name << " = malloc(" << attrs->cache_shape.size() << " * 4)";

  func_def_.PushDecl(t0);
  for (uint i = 0; i < attrs->cache_shape.size(); i++) {
    t0 << cache_shape_name << "[" << i << "] = " << to_string(attrs->cache_shape[i]);
    func_def_.PushDecl(t0);
  }

  string shape_name = "shape_" + to_string(buf_idx_);
  t0 << "int32_t *" << shape_name << " = malloc(" << attrs->shape.size() << " * 4)";

  func_def_.PushDecl(t0);
  for (uint i = 0; i < attrs->shape.size(); i++) {
    t0 << shape_name << "[" << i << "] = " << to_string(attrs->shape[i]);
    func_def_.PushDecl(t0);
  }

  string axes_name = "axes_" + to_string(buf_idx_);
  t0 << "int32_t *" << axes_name << " = malloc(" << attrs->axes.size() << " * 4)";
  func_def_.PushDecl(t0);
  for (uint i = 0; i < attrs->axes.size(); i++) {
    t0 << axes_name << "[" << i << "] = " << to_string(attrs->axes[i]);
    func_def_.PushDecl(t0);
  }

  t0 << params_name << "->cache_shape = " << cache_shape_name;
  func_def_.PushDecl(t0);
  t0 << params_name << "->shape = " << shape_name;
  func_def_.PushDecl(t0);
  t0 << params_name << "->axes = " << axes_name;
  func_def_.PushDecl(t0);
  PushOutput(output_name, call);

  params_common_setup(decl, call, "cache_matmul", params_name, attrs->layer_name.c_str());
  end_stream(decl, "cache_matmul");
}

void CodegenCSINN::CacheConv1d(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  CSINNOP* op = new CSINNOP;
  const auto* attr = call->attrs.as<QnnCSICacheConv1DAttrs>();
  CHECK(attr);

  CHECK(call->args.size() == 3) << "Conv1d expects 3 args";

  /* Make function call with arguments start */
  decl << "(";

  /* Emit_ input tensor */
  visit(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string complete_name = get_complete_layer_name("cache_conv1d", attr->layer_name.c_str());
  op->set_name(complete_name);
  string input_name = CreateInputTensor(op, decl, call, 0, GetCallOPQuant(call, 0));

  /* Emit output tensor */
  string output_name = CreateOutputTensor(op, decl, call, GetCallOPQuant(call, 3));

  collect_quant_info(complete_name, attr->q_params, 3);

  /* Emit kernel tensor */
  visit(call->args[1]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto kernel = constant_[0];
  auto wshape = call->args[1]->get_shape();
  string kernel_name = "kernel_" + to_string(buf_idx_);
  CreateConstantTensor(op, kernel, kernel_name, wshape, GetCallOPQuant(call, 1));
  decl << ", " << kernel_name;

  /* Emit bias tensor */
  visit(call->args[2]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto bias = constant_[0];
  auto bshape = call->args[2]->get_shape();
  string bias_name = "bias_" + to_string(buf_idx_);
  CreateConstantTensor(op, bias, bias_name, bshape, cfg->dtype_activation, GetCallOPQuant(call, 0),
                       GetCallOPQuant(call, 1), GetCallOPQuant(call, 2));

  decl << ", " << bias_name;

  string params_name = "params_" + to_string(buf_idx_);

  decl << ", " << params_name << ")";
  push_decl(op);
  malloc_params("csinn_cache_conv1d_params", params_name);
  string shape_name = "cache_shape_" + to_string(buf_idx_);
  t0 << "int32_t *" << shape_name << " = malloc(" << attr->cache_shape.size() << " * 4)";

  func_def_.PushDecl(t0);
  for (uint i = 0; i < attr->cache_shape.size(); i++) {
    t0 << shape_name << "[" << i << "] = " << to_string(attr->cache_shape[i]);
    func_def_.PushDecl(t0);
  }

  t0 << params_name << "->cache_shape = " << shape_name;
  func_def_.PushDecl(t0);

  t0 << params_name << "->group = " << to_string(attr->groups);
  func_def_.PushDecl(t0);
  Array<IndexExpr> strides = attr->strides;
  t0 << params_name << "->stride_width = " << to_string(strides[0].as<IntImmNode>()->value);
  func_def_.PushDecl(t0);
  Array<IndexExpr> dilation = attr->dilation;
  t0 << params_name << "->dilation_width = " << to_string(dilation[0].as<IntImmNode>()->value);
  func_def_.PushDecl(t0);
  Setup1dPadding<QnnCSICacheConv1DAttrs>(params_name, attr);

  PushOutput(output_name, call);
  params_common_setup(decl, call, "cache_conv1d", params_name, attr->layer_name.c_str());
  end_stream(decl, "cache_conv1d");
}

void CodegenCSINN::LayerNorm(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  CSINNOP* op = new CSINNOP;
  const auto* attrs = call->attrs.as<QnnCSILayerNormAttrs>();
  CHECK(attrs);

  CHECK(call->args.size() == 3) << "LayerNorm expects 3 args";

  // Make function call with input buffers when visiting arguments
  decl << "(";

  /* Emit input tensor */
  visit(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string complete_name = get_complete_layer_name("layer_norm", attrs->layer_name.c_str());
  op->set_name(complete_name);
  string input_name = CreateInputTensor(op, decl, call, 0, GetCallOPQuant(call, 0));

  /* Emit output tensor */
  auto output_qinfo = GetCallOPQuant(call, 3);
  string output_name = CreateOutputTensor(op, decl, call, output_qinfo);

  collect_quant_info(complete_name, attrs->q_params, 3);

  /* Emit gamma tensor */
  visit(call->args[1]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto gamma = constant_[0];
  auto gamma_shape = call->args[1]->get_shape();

  string gamma_name = "gamma_" + to_string(buf_idx_);
  CreateConstantTensor(op, gamma, gamma_name, gamma_shape, GetCallOPQuant(call, 1));

  decl << ", " << gamma_name;

  /* Emit bias tensor */
  visit(call->args[2]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto beta = constant_[0];
  auto bshape = call->args[2]->get_shape();
  string beta_name = "beta_" + to_string(buf_idx_);
  CreateConstantTensor(op, beta, beta_name, bshape, GetCallOPQuant(call, 2));

  decl << ", " << beta_name;

  string params_name = "params_" + to_string(buf_idx_);

  decl << ", " << params_name << ")";
  push_decl(op);
  malloc_params("csinn_layer_norm_params", params_name);

  t0 << params_name << "->epsilon = " << attrs->epsilon;
  func_def_.PushDecl(t0);
  t0 << params_name << "->axis = " << attrs->axis;
  func_def_.PushDecl(t0);
  string center = attrs->center ? "true" : "false";
  t0 << params_name << "->center = " << center;
  func_def_.PushDecl(t0);
  string scale = attrs->scale ? "true" : "false";
  t0 << params_name << "->scale = " << scale;
  func_def_.PushDecl(t0);
  PushOutput(output_name, call, output_qinfo->dtype);

  params_common_setup(decl, call, "layer_norm", params_name, attrs->layer_name.c_str());
  end_stream(decl, "layer_norm");
}

void CodegenCSINN::DataConvert(const CallNode* call) {
  std::ostringstream decl;
  CSINNOP* op = new CSINNOP;
  const auto* attr = call->attrs.as<QnnCSIDataConvertAttrs>();

  // Make function call with input buffers when visiting arguments
  decl << "(";

  /* Emit input tensor */
  visit(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string complete_name = attr->layer_name.c_str();
  op->set_name(complete_name);

  auto input_qinfo = GetCallOPQuant(call, 0);
  string input_name = CreateInputTensor(op, decl, call, 0, input_qinfo);

  /* Emit output tensor */
  auto output_qinfo = GetCallOPQuant(call, 1);
  string output_name = CreateOutputTensor(op, decl, call, output_qinfo);
  PushOutput(output_name, call, output_qinfo->dtype);

  push_decl(op);
  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";

  malloc_params("csinn_siso_params", params_name);
  string op_name = "data_convert";
  params_common_setup(decl, call, op_name, params_name, attr->layer_name.c_str());
  end_stream(decl, op_name);
}

void CodegenCSINN::OneHot(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  CSINNOP* op = new CSINNOP;
  const auto* attr = call->attrs.as<QnnCSIOneHotAttrs>();
  CHECK(attr);
  SisoOp<QnnCSIOneHotAttrs>(op, decl, call, attr, "one_hot");

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";
  push_decl(op);
  malloc_params("csinn_one_hot_params", params_name);

  t0 << params_name << "->depth = " << to_string(attr->depth);
  func_def_.PushDecl(t0);
  t0 << params_name << "->axis = " << to_string(attr->axis);
  func_def_.PushDecl(t0);

  params_common_setup(decl, call, "one_hot", params_name, attr->layer_name.c_str());
  end_stream(decl, "one_hot");
}

void CodegenCSINN::Where(const CallNode* call) {
  std::ostringstream decl;
  CSINNOP* op = new CSINNOP;
  const auto* attr = call->attrs.as<QnnCSIUnaryAttrs>();

  // Make function call with input buffers when visiting arguments
  decl << "(";

  /* Emit input0 tensor */
  visit(call->args[0]);
  string complete0_name = attr->layer_name.c_str();
  op->set_name(complete0_name);
  buf_idx_++;
  auto input0_qinfo = GetCallOPQuant(call, 0);
  string input0_name;
  if (call->args[0].as<ConstantNode>()) {
    CHECK(constant_.size() == 1) << "Every args expects a single constant_";
    auto constant_input = constant_[0];
    input0_name = "where_condition_" + to_string(buf_idx_);
    auto in_shape0 = call->args[0]->get_shape();
    CreateConstantTensor(op, constant_input, input0_name, in_shape0, input0_qinfo);
    decl << input0_name;
  } else {
    CHECK(out_.size() == 1) << "Every args expects a single out_";
    input0_name = CreateInputTensor(op, decl, call, 0, input0_qinfo);
  }

  decl << ", ";

  /* Emit input1 tensor */
  visit(call->args[1]);
  string complete1_name = attr->layer_name.c_str();
  op->set_name(complete1_name);

  auto input1_qinfo = GetCallOPQuant(call, 1);
  string input1_name;
  if (call->args[1].as<ConstantNode>()) {
    CHECK(constant_.size() == 1) << "Every args expects a single constant_";
    auto constant_input = constant_[0];
    input1_name = "where_x_" + to_string(buf_idx_);
    auto in_shape1 = call->args[1]->get_shape();
    CreateConstantTensor(op, constant_input, input1_name, in_shape1, input1_qinfo);
    decl << input1_name;
  } else {
    CHECK(out_.size() == 1) << "Every args expects a single out_";
    input1_name = CreateInputTensor(op, decl, call, 1, input1_qinfo);
  }
  decl << ", ";

  /* Emit input2 tensor */
  visit(call->args[2]);
  string complete2_name = attr->layer_name.c_str();
  op->set_name(complete2_name);

  auto input2_qinfo = GetCallOPQuant(call, 2);
  string input2_name;
  if (call->args[2].as<ConstantNode>()) {
    CHECK(constant_.size() == 1) << "Every args expects a single constant_";
    auto constant_input = constant_[0];
    input2_name = "where_y_" + to_string(buf_idx_);
    auto in_shape2 = call->args[2]->get_shape();
    CreateConstantTensor(op, constant_input, input2_name, in_shape2, input2_qinfo);
    decl << input2_name;
  } else {
    CHECK(out_.size() == 1) << "Every args expects a single out_";
    input2_name = CreateInputTensor(op, decl, call, 2, input2_qinfo);
  }

  /* Emit output tensor */
  auto output_qinfo = GetCallOPQuant(call, 3);
  string output_name = CreateOutputTensor(op, decl, call, output_qinfo);
  PushOutput(output_name, call, output_qinfo->dtype);

  push_decl(op);
  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";

  malloc_params("csinn_where_params", params_name);
  string op_name = "where";
  params_common_setup(decl, call, op_name, params_name, attr->layer_name.c_str());
  end_stream(decl, op_name);
}

void CodegenCSINN::WhereSoftmax(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  CSINNOP* op = new CSINNOP;
  const auto* attr = call->attrs.as<QnnCSIWhereSoftmaxAttrs>();

  // Make function call with input buffers when visiting arguments
  decl << "(";

  /* Emit input0 tensor */
  visit(call->args[0]);
  string complete0_name = attr->layer_name.c_str();
  op->set_name(complete0_name);
  buf_idx_++;
  auto input0_qinfo = GetCallOPQuant(call, 0);
  string input0_name;
  if (call->args[0].as<ConstantNode>()) {
    CHECK(constant_.size() == 1) << "Every args expects a single constant_";
    auto constant_input = constant_[0];
    input0_name = "where_condition_" + to_string(buf_idx_);
    auto in_shape0 = call->args[0]->get_shape();
    CreateConstantTensor(op, constant_input, input0_name, in_shape0, input0_qinfo);
    decl << input0_name;
  } else {
    CHECK(out_.size() == 1) << "Every args expects a single out_";
    input0_name = CreateInputTensor(op, decl, call, 0, input0_qinfo);
  }

  decl << ", ";

  /* Emit input1 tensor */
  if (!call->args[1].as<ConstantNode>()) {
    CHECK(0) << "The first input of where_oftmax must be";
  }

  /* Emit input2 tensor */
  visit(call->args[2]);
  string complete2_name = attr->layer_name.c_str();
  op->set_name(complete2_name);

  auto input2_qinfo = GetCallOPQuant(call, 2);
  string input2_name;
  if (call->args[2].as<ConstantNode>()) {
    CHECK(constant_.size() == 1) << "Every args expects a single constant_";
    auto constant_input = constant_[0];
    input2_name = "where_y_" + to_string(buf_idx_);
    auto in_shape2 = call->args[2]->get_shape();
    CreateConstantTensor(op, constant_input, input2_name, in_shape2, input2_qinfo);
    decl << input2_name;
  } else {
    CHECK(out_.size() == 1) << "Every args expects a single out_";
    input2_name = CreateInputTensor(op, decl, call, 2, input2_qinfo);
  }

  /* Emit output tensor */
  auto output_qinfo = GetCallOPQuant(call, 3);
  string output_name = CreateOutputTensor(op, decl, call, output_qinfo);
  PushOutput(output_name, call, output_qinfo->dtype);

  push_decl(op);
  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";

  malloc_params("csinn_where_softmax_params", params_name);

  t0 << params_name << "->minus_inf = -(float)HUGE_VALF";
  func_def_.PushDecl(t0);
  t0 << params_name << "->axis = " << to_string(attr->axis);
  func_def_.PushDecl(t0);
  string op_name = "where_softmax";
  params_common_setup(decl, call, op_name, params_name, attr->layer_name.c_str());
  end_stream(decl, op_name);
}

}  // namespace csinn
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
