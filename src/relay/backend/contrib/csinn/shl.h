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
 * \file src/relay/backend/contrib/csinn/pnna.h
 * \brief The base class for pnna.
 */
#ifndef TVM_RELAY_BACKEND_CONTRIB_CSINN_SHL_H_
#define TVM_RELAY_BACKEND_CONTRIB_CSINN_SHL_H_

#include <algorithm>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "backend.h"
#include "c906.h"
#include "c908.h"
#include "c920.h"
#include "c920v2.h"
#include "csinn.h"
#include "e907.h"
#include "gref.h"
#include "hth1520.h"
#include "ref.h"
#include "rvm.h"
#include "th1520.h"
#include "xuantie_aot.h"

namespace tvm {
namespace relay {
namespace contrib {
namespace csinn {

class SHL : public Pass {
 public:
  virtual string get_ccode(void) { return code_stream_; }

  void compiler(void);

  template <typename T>
  void target_build(T* builder);

  void SetExtFuncId(string func_id) { this->ext_func_id_ = func_id; }

  Map<String, Array<Array<Array<IndexExpr>>>> ret_quant_info(void) { return quant_info; }

  Map<String, Array<Array<Array<IndexExpr>>>> quant_info;
  string ext_func_id_{""};
  string code_stream_;
};

}  // namespace csinn
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_BACKEND_CONTRIB_CSINN_SHL_H_
