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
 * \file src/relay/backend/contrib/csinn/hth1520.h
 * \brief The base class for hth1520.
 */
#ifndef TVM_RELAY_BACKEND_CONTRIB_CSINN_HTH1520_H_
#define TVM_RELAY_BACKEND_CONTRIB_CSINN_HTH1520_H_

#include <string>
#include <vector>

#include "gref.h"
#include "th1520.h"

namespace tvm {
namespace relay {
namespace contrib {
namespace csinn {

class CodegenHTH1520 : public CodegenHGref {
 public:
  CodegenHTH1520() : CodegenHGref() {
    target_op_list = {"qnn.csi.conv2d",
                      "qnn.csi.concatenate",
                      "qnn.csi.relu",
                      "qnn.csi.dense",
                      "qnn.csi.avgpool2d",
                      "qnn.csi.maxpool2d",
                      "qnn.csi.add",
                      "qnn.csi.clip",
                      "qnn.csi.upsampling",
                      "qnn.csi.mean",
                      "qnn.csi.mul",
                      "qnn.csi.bias_add",
                      "qnn.csi.deconv2d",
                      "qnn.csi.global_avgpool2d",
                      "qnn.csi.global_maxpool2d",
                      "qnn.csi.leaky_relu",
                      "qnn.csi.sigmoid",
                      "qnn.csi.split",
                      "qnn.csi.strided_slice",
                      "qnn.csi.transpose",
                      "qnn.csi.reshape"};

    if (cfg->quantization_scheme == "CSINN_QUANT_INT8_SYM" ||
        cfg->quantization_scheme == "CSINN_QUANT_INT16_SYM") {
      cfg->dtype_activation = "float";
      cfg->dtype_input = "float";
      cfg->dtype_weight = "float";
      cfg->weight_quantized_type = "sym";
      cfg->activate_quantized_type = "sym";
      base_dtype_ = "CSINN_DTYPE_FLOAT32";
    }
  }
  virtual ~CodegenHTH1520() {}
  virtual void params_common_setup(std::ostringstream& decl, const CallNode* call, string op_name,
                                   string params_name, string layer_name, string layout);

  void EmitSessionSetup();
  void ModelBinarySave();
  string get_ccode();
  void phase1() final;
  void SessionRunMode() { func_def_.OneLine("sess->base_run_mode = CSINN_RM_CPU_BASE_HYBRID;"); }
};

}  // namespace csinn
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_BACKEND_CONTRIB_CSINN_HTH1520_H_
