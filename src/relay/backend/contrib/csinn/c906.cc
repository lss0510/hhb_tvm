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
 * \file src/relay/backend/contrib/csinn/c906.cc
 * \brief Implementation of CSINN c906 codegen APIs.
 */

#include "c906.h"

#include <string>

using namespace tvm::relay::qnn;
namespace tvm {
namespace relay {
namespace contrib {
namespace csinn {

CSIConstant* CodegenC906::CastParams(CSIConstant* data, string target_dtype,
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
    if (target_dtype == "int8_t") {
      if ((layout_ == "NHWC") && depthwise_kernel) {
        Axis3Cast(data, output, q_infos, target_dtype, q_size, inner_size);
      } else {
        Axis0Cast(data, output, q_infos, target_dtype, q_size, inner_size);
      }
    } else if (target_dtype == "float16") {
      int16_t* out = reinterpret_cast<int16_t*>(output->get_data_buf());
      output->mtype = CSINN_MEM_TYPE_CPU_ALIGNED;
      output->set_align(32);
      output->set_offset(8);
      for (int i = 0; i < inner_size; i++) {
        for (int c = 0; c < q_size; c++) {
          int index = i * q_size + c;

          float q_value = input_data[index] / q_infos[c].scale;
          int16_t out_ = float32_to_float16(q_value);
          out[i] = out_;
        }
      }
    } else {
      LOG(ERROR) << "C906 get error dtype:" << target_dtype;
    }
    free(input_data);
  }
  return output;
}

void CodegenC906::Conv2d(const CallNode* call, string op_name) {
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
    CreateConstantTensor(op, kernel, kernel_name, wshape, kernel_qinfo, depthwise_kernel);
  }

  bool gemm_kernel = is_gemm_kernel(depthwise_kernel, op);

  if (gemm_kernel) {
    CSINNConstantTensor* ct = op->get_constant(0);
    reorder_kernel(ct, attr->groups);
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
  if (gemm_kernel) {
    std::ostringstream t0;
    malloc_params("csinn_conv2d_params", params_name);
    t0 << params_name << "->group = " << to_string(attr->groups);
    func_def_.PushDecl(t0);
    Array<IndexExpr> strides = attr->strides;
    t0 << params_name << "->stride_height = " << to_string(strides[0].as<IntImmNode>()->value);
    func_def_.PushDecl(t0);
    t0 << params_name << "->stride_width = " << to_string(strides[1].as<IntImmNode>()->value);
    func_def_.PushDecl(t0);
    Array<IndexExpr> dilation = attr->dilation;
    t0 << params_name << "->dilation_height = " << to_string(dilation[0].as<IntImmNode>()->value);
    func_def_.PushDecl(t0);
    t0 << params_name << "->dilation_width = " << to_string(dilation[1].as<IntImmNode>()->value);
    func_def_.PushDecl(t0);
    t0 << params_name << "->conv_extra.conv_mode = CSINN_GEMM";
    func_def_.PushDecl(t0);
    SetupPadding(params_name, attr);
  } else {
    SetupConv2dParams<QnnCSIConv2DAttrs>(params_name, attr);
  }
  if (fuse_zp) {
    std::ostringstream t0;
    t0 << params_name << "->conv_extra.fuse_zp2bias = true";
    func_def_.PushDecl(t0);
  }

  PushOutput(output_name, call, cfg->dtype_weight);

  params_common_setup(decl, call, op_name, params_name, attr->layer_name.c_str(),
                      "CSINN_LAYOUT_NCHW");
  end_stream(decl, op_name);
}

void CodegenC906::Dense(const CallNode* call) {
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
  // CreateConstantTensor(op, kernel, kernel_name, wshape, kernel_qinfo);
  if (cfg->quantization_scheme == "CSINN_QUANT_FLOAT16_W_INT8") {
    kernel_qinfo->dtype = "int8_t";
    CreateWeightTensor(op, kernel, kernel_name, wshape, kernel_qinfo);
  } else {
    CreateConstantTensor(op, kernel, kernel_name, wshape, kernel_qinfo);
  }

  CSINNConstantTensor* ct = op->get_constant(0);
  if (ct->tensor->dtype == CSINN_DTYPE_FLOAT16) {
    reorder_fcl(ct);
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

}  // namespace csinn
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
