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
 * \file src/relay/backend/contrib/csinn/gref.h
 * \brief The base class for gref.
 */
#ifndef TVM_RELAY_BACKEND_CONTRIB_CSINN_GREF_H_
#define TVM_RELAY_BACKEND_CONTRIB_CSINN_GREF_H_

#include <string>
#include <vector>

#include "csinn.h"

namespace tvm {
namespace relay {
namespace contrib {
namespace csinn {

class CodegenGref : public CodegenCSINN {
 public:
  virtual string get_ccode(void);
  virtual void EmitNBGSetup(void);

  void SessionRunMode() {
    if (auto_hybrid_quantization) {
      func_def_.OneLine("sess->base_run_mode = CSINN_RM_CPU_BASE_HYBRID;");
    } else {
      func_def_.OneLine("sess->base_run_mode = CSINN_RM_CPU_GRAPH;");
    }
  }
};

class CodegenHGref : public CodegenGref {};

}  // namespace csinn
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_BACKEND_CONTRIB_CSINN_GREF_H_
