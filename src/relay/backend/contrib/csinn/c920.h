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
 * \file src/relay/backend/contrib/csinn/c920.h
 * \brief The base class for c920.
 */
#ifndef TVM_RELAY_BACKEND_CONTRIB_CSINN_C920_H_
#define TVM_RELAY_BACKEND_CONTRIB_CSINN_C920_H_

#include <string>
#include <vector>

#include "gref.h"

namespace tvm {
namespace relay {
namespace contrib {
namespace csinn {
class CodegenC920 : public CodegenGref {
 public:
  void EmitNBGSetup(void) {}
  void ModelBinarySave() {
    std::ostringstream t0;

    t0 << "sess->base_quant_type = " << cfg->quantization_scheme << ";";
    func_def_.OneLine(t0);

    if (model_save == "run_only") {
      t0 << "sess->model.save_mode = CSINN_RUN_ONLY;";
    } else if (model_save == "save_and_run") {
      t0 << "sess->model.save_mode = CSINN_SAVE_AND_RUN;";
    } else {
      std::cerr << "Unsupport for model save_mode type: " << model_save << "\n";
      exit(-1);
    }
    func_def_.OneLine(t0);
  }
};

}  // namespace csinn
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_BACKEND_CONTRIB_CSINN_C920_H_
