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
 * \file src/relay/backend/contrib/csinn/hth1520.cc
 * \brief Implementation of CSINN hth1520 codegen APIs.
 */

#include "hth1520.h"

#include <algorithm>
#include <string>
#include <vector>

using namespace tvm::relay::qnn;
namespace tvm {
namespace relay {
namespace contrib {
namespace csinn {

void CodegenHTH1520::params_common_setup(std::ostringstream& decl, const CallNode* call,
                                         string op_name, string params_name, string layer_name,
                                         string layout = "CSINN_LAYOUT_NCHW") {
  std::ostringstream t0;
  if (!(layout_ == "NCHW" && layout == "CSINN_LAYOUT_NCHW")) {
    t0 << params_name << "->base.layout = CSINN_LAYOUT_" << layout_;
    func_def_.PushDecl(t0);
  }

  string complete_name = get_complete_layer_name(op_name, layer_name);
  t0 << params_name << "->base.name = "
     << "\"" << complete_name << "\"";
  func_def_.PushDecl(t0);
  params_idx_++;
  if (InOpList(call) || (target_ == "th1520" && auto_hybrid_quantization &&
                         !IsOp(call, "qnn.csi.softmax") && !IsOp(call, "qnn.csi.data_convert"))) {
    t0 << params_name << "->base.api = CSINN_TH1520";
    func_def_.PushDecl(t0);
  }
  auto call_cfg = call->get_quant_config();
  if (call_cfg && call_cfg->quantization_scheme != "unset" &&
      call_cfg->quantization_scheme != "CSINN_QUANT_INT4_ASYM_W_SYM" &&
      call_cfg->quantization_scheme != "CSINN_QUANT_INT8_ASYM_W_SYM" &&
      !hybrid_layer_name.empty()) {
    t0 << params_name << "->base.quant_type = " << call_cfg->quantization_scheme;
    func_def_.PushDecl(t0);
  }

  t0 << "csinn_" << op_name << "_init" << decl.str();
  func_def_.PushDecl(t0);
}

void CodegenHTH1520::EmitSessionSetup(void) {
  std::ostringstream t0;
  t0 << "void *" << ext_func_id_ << "_(";
  t0 << "char *params_base) {";
  func_def_.OneLine(t0);
  func_def_.EnterScope();

  func_def_.OneLine("struct csinn_session *sess = csinn_alloc_session();");
  SessionRunMode();
  ModelBinarySave();
  t0 << "sess->base_api = " << target_name_ << ";";
  func_def_.OneLine(t0);
  t0 << "sess->base_dtype = " << base_dtype_ << ";";
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

  auto ctx = transform::PassContext::Current();
  auto opt = ctx->GetConfig<CSINNConfig>("relay.ext.csinn.options");
  auto opt_cfg = opt.value();

  double fix_height = opt_cfg->th1520_input_fix_height;
  double fix_width = opt_cfg->th1520_input_fix_width;
  if (fix_height != 0) {
    t0 << "shl_pnna_set_input_strides(sess, 1, " << fix_height << " ," << fix_width << ");";
    func_def_.OneLine(t0);
  }

  func_def_.NewLine();
  func_def_.OneLine("csinn_session_setup(sess);");
  func_def_.OneLine("return sess;");
  func_def_.ExitScope();
  func_def_.OneLine("}");
}

void CodegenHTH1520::ModelBinarySave() {
  std::ostringstream t0;
  std::string quantization_scheme = cfg->quantization_scheme == "CSINN_QUANT_INT8_ASYM_W_SYM"
                                        ? "CSINN_QUANT_INT8_ASYM"
                                        : cfg->quantization_scheme;
  t0 << "sess->base_quant_type = " << quantization_scheme << ";";
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
}

string CodegenHTH1520::get_ccode(void) {
  EmitVersion();
  EmitHeader();
  EmitSessionSetup();
  EmitSessionRun();
  EmitNBGSetup();
  DumpConstant();
  DumpGraphInfo();
  return func_def_.str();
}

void CodegenHTH1520::phase1() {
  QuantConfig* set_config = new QuantConfig(expr_);
  set_config->set_quant_config(cfg);
  set_config->set_hybrid_quant_config(hybrid_cfg, hybrid_layer_name);
  expr_ = set_config->get_expr();

  QuantInfo* qinfo = new QuantInfo(expr_);
  TH1520QuantCalculator* lqc = new TH1520QuantCalculator;
  qinfo->set_quant_calulator(lqc);
  qinfo->calculate_quant_info();
  expr_ = qinfo->get_expr();

  DataConvertInserter* insert_dc = new DataConvertInserter(expr_);
  insert_dc->set_quant_calulator(lqc);
  insert_dc->Insert(cfg);
  expr_ = insert_dc->get_expr();
}

}  // namespace csinn
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
