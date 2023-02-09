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
 * \file src/relay/backend/contrib/csinn/anole.h
 * \brief The base class for anole.
 */
#ifndef TVM_RELAY_BACKEND_CONTRIB_CSINN_ANOLE_H_
#define TVM_RELAY_BACKEND_CONTRIB_CSINN_ANOLE_H_

#include <string>
#include <vector>

#include "gref.h"

namespace tvm {
namespace relay {
namespace contrib {
namespace csinn {
class CodegenAnole : public CodegenGref {
 public:
  virtual void visit_expr(const CallNode* call);
  virtual void EmitHeader(void);
  virtual void EmitSessionSetup(void);
  virtual void EmitNBGSetup(void);
  void ModelBinarySave();

  virtual void DisoOp(const CallNode* call, string op_name);
  virtual void Flatten(const CallNode* call);
  virtual void Squeeze(const CallNode* call);
  virtual void Reshape(const CallNode* call);
  void SessionRunMode() { func_def_.OneLine("sess->base_run_mode = CSINN_RM_NPU_GRAPH;"); }
};

}  // namespace csinn
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_BACKEND_CONTRIB_CSINN_ANOLE_H_
