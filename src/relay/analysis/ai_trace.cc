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
 *
 * \file ai_trace.cc
 * \brief Pass to profile the model, and get some ai trace information.
 */

#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>

#include <iostream>
#include <string>

#include "../transforms/pattern_utils.h"
#include "get_aitrace_data.h"
#include "profiler_parser.h"
#include "qnn_aitrace_data.h"
namespace tvm {
namespace relay {

namespace aitrace {

static int cnt = 0;

inline bool FindType(Array<String> type, String key) {
  bool res = false;
  for (auto t : type) {
    if (t == key) {
      res = true;
      break;
    }
  }
  return res;
}

class AiTraceCounter : private ExprVisitor {
 public:
  AiTraceCounter() {}
  explicit AiTraceCounter(Array<String> type) : type_(type) {}

  Array<AiTraceDataFrame> GetAiTraceData(const Expr& expr) {
    (*this)(expr);
    return aitrace_data_;
  }

 private:
  void VisitExpr_(const CallNode* call_node) final {
    static const auto& fcal_map = Op::GetAttrMap<FCalAmount>("FCalAmount");
    static const auto& fmemory_map = Op::GetAttrMap<FMemory>("FMemory");
    static const auto& fopname_map = Op::GetAttrMap<FOpName>("FOpName");
    ExprVisitor::VisitExpr_(call_node);
    auto fcal = fcal_map.get(call_node->op, nullptr);
    auto fmemory = fmemory_map.get(call_node->op, nullptr);
    auto fopname = fopname_map.get(call_node->op, nullptr);
    Array<AiTraceDataFrame> all_trace_data;

    /* Add op info */
    std::string op_type = "unknown";
    if (fopname != nullptr) {
      op_type = fopname();
    }
    std::string op_name = op_type + "_" + std::to_string(cnt++);
    ReplaceInvalidSymbol(&op_name, "_");

    Map<String, ObjectRef> op_info;
    op_info.Set("type", String(op_type));
    op_info.Set("name", String(op_name));

    AiTraceDataFrame op_info_trace_data;
    op_info_trace_data.Set("op", op_info);

    all_trace_data.push_back(op_info_trace_data);

    /* Add other info */
    AiTraceDataFrame tmp_data;
    // calculation amount
    if (FindType(type_, "cal") || FindType(type_, "all")) {
      if (fcal != nullptr) {
        tmp_data = fcal(GetRef<Call>(call_node));
      } else {
        CalculationAmontIndicator cai;
        tmp_data = cai.GetIndicatorMap();
      }
      all_trace_data.push_back(tmp_data);
    }

    // memory
    if (FindType(type_, "mem") || FindType(type_, "all")) {
      if (fmemory != nullptr) {
        tmp_data = fmemory(GetRef<Call>(call_node));
      } else {
        MemoryIndicator mi;
        tmp_data = mi.GetIndicatorMap();
      }
      all_trace_data.push_back(tmp_data);
    }

    /* combine all trace data. */
    AiTraceDataFrame combine_data;
    for (auto atd : all_trace_data) {
      for (auto& td : atd) {
        combine_data.Set(td.first, td.second);
      }
    }

    aitrace_data_.push_back(combine_data);
  }
  Array<AiTraceDataFrame> aitrace_data_;
  Array<String> type_;
};

Array<AiTraceDataFrame> GetAiTraceData(const Expr& expr, const tvm::Target& target) {
  CHECK(target.defined()) << "target is empty, please set tvm::Target.";
  Array<String> type = target->GetAttr<Array<String>>("type", Array<String>({""})).value();
  String path = target->GetAttr<String>("path", String("")).value();

  AiTraceCounter atc = AiTraceCounter(type);
  Array<AiTraceDataFrame> result = atc.GetAiTraceData(expr);

  if (path != "") {
    AiTraceData atdata = Convert2ATData(result);
    atdata.ToFile(path);
  }

  return result;
}

Array<AiTraceDataFrame> QnnAiTraceData(const Expr& expr, const tvm::Target& target) {
  CHECK(target.defined()) << "target is empty, please set tvm::Target.";
  Array<String> type = target->GetAttr<Array<String>>("type", Array<String>({""})).value();
  String path = target->GetAttr<String>("path", String("")).value();

  AiTraceCounter atc = AiTraceCounter(type);
  Array<AiTraceDataFrame> result = atc.GetAiTraceData(expr);

  if (path != "") {
    AiTraceData atdata = QnnConvert2ATData(result);
    atdata.ToFile(path);
  }

  return result;
}

TVM_REGISTER_GLOBAL("relay.analysis.GetAiTraceData").set_body_typed(GetAiTraceData);
TVM_REGISTER_GLOBAL("relay.analysis.QnnAiTraceData").set_body_typed(QnnAiTraceData);
}  // namespace aitrace
}  // namespace relay
}  // namespace tvm
