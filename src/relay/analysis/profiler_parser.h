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
 * \file profiler_parser.h
 * \brief parse profiler data.
 */
#ifndef TVM_RELAY_ANALYSIS_PROFILER_PARSER_H_
#define TVM_RELAY_ANALYSIS_PROFILER_PARSER_H_

#include <iostream>
#include <string>
#include <vector>

/* ai trace type */
/* common type */
#define AI_TRACE_NULL 0x0
#define AI_TRACE_VERSION 0x1
#define AI_TRACE_END 0x2
#define AI_TRACE_INSN_TYPE 0x3
#define AI_TRACE_INSN_NAME 0x4
/* calculation amount */
#define AI_TRACE_FUSED_MUL_ADD_NUM 0x21
#define AI_TRACE_MUL_NUM 0x22
#define AI_TRACE_DIV_NUM 0x23
#define AI_TRACE_ADD_NUM 0x24
#define AI_TRACE_EXP_NUM 0x25
#define AI_TRACE_COMP_NUM 0x26
#define AI_TRACE_SUB_NUM 0x27
/* memory */
#define AI_TRACE_PARAMS_NUM 0x30
#define AI_TRACE_OUTPUT_NUM 0x31

/* some length(byte) define */
#define VERSION_BYTE_NUM 3
#define INSN_TYPE_BYTE_NUM 2
#define INSN_NAME_BYTE_NUM 36
#define CAL_AMOUNT_IND_BYTE_NUM 8
#define MEMORY_IND_BYTE_NUM 8

/* relay ops */
enum {
  NOOP = 0,
  /* relay ops */
  RELAY_OP_ADD,
  RELAY_OP_AVGPOOL_2D,
  RELAY_OP_BATCH_FLATTEN,
  RELAY_OP_BATCH_NORM,
  RELAY_OP_BIAS_ADD,
  RELAY_OP_CONCATENATE,
  RELAY_OP_CONV2D,
  RELAY_OP_CONV2D_TRANSPOSE,
  RELAY_OP_DENSE,
  RELAY_OP_DROPOUT,
  RELAY_OP_GLOBAL_AVGPOOL_2D,
  RELAY_OP_GLOBAL_MAXPOOL_2D,
  RELAY_OP_LRN,
  RELAY_OP_L2NORMALIZE,
  RELAY_OP_MAXIMUM,
  RELAY_OP_MAXPOOL_2D,
  RELAY_OP_MAXPOOL_2D_LOCATION,
  RELAY_OP_MAXPOOL_2D_WITH_ARGMAX,
  RELAY_OP_MULTIPLY,
  RELAY_OP_PRELU,
  RELAY_OP_PROPOSAL,
  RELAY_OP_PSROIPOOLING,
  RELAY_OP_RELU,
  RELAY_OP_RESHAPE,
  RELAY_OP_ROIPOOL,
  RELAY_OP_SIGMOID,
  RELAY_OP_SOFTMAX,
  RELAY_OP_SPLIT,
  RELAY_OP_STRIDED_SLICE,
  RELAY_OP_TANH,
  RELAY_OP_TRANSPOSE,
  RELAY_OP_UNPOOLING,
  RELAY_OP_UNPSAMPLING,
  RELAY_OP_SIZE,
};

/* qnn ops */
enum {
  QNN_NOOP = 0,
  /* qnn ops */
  QNN_OP_ADD,
  QNN_OP_AVGPOOL_2D,
  QNN_OP_BATCH_NORM,
  QNN_OP_BIAS_ADD,
  QNN_OP_CONCATENATE,
  QNN_OP_CONV2D,
  QNN_OP_CONV2D_TRANSPOSE,
  QNN_OP_DENSE,
  QNN_OP_GLOBAL_AVGPOOL_2D,
  QNN_OP_GLOBAL_MAXPOOL_2D,
  QNN_OP_LRN,
  QNN_OP_MAXIMUM,
  QNN_OP_MAXPOOL_2D,
  QNN_OP_MAXPOOL_2D_LOCATION,
  QNN_OP_MAXPOOL_2D_WITH_ARGMAX,
  QNN_OP_MULTIPLY,
  QNN_OP_PRELU,
  QNN_OP_PROPOSAL,
  QNN_OP_PSROIPOOLING,
  QNN_OP_RELU,
  QNN_OP_RESHAPE,
  QNN_OP_ROIPOOL,
  QNN_OP_SIGMOID,
  QNN_OP_SOFTMAX,
  QNN_OP_SPLIT,
  QNN_OP_STRIDED_SLICE,
  QNN_OP_TANH,
  QNN_OP_TRANSPOSE,
  QNN_OP_UNPOOLING,
  QNN_OP_UNPSAMPLING,
  QNN_OP_SIZE,
};

class AiTraceVersion {
 public:
  AiTraceVersion() : version_str_("0.0.0"), major_(0), minor_(0), patch_(0) {}
  ~AiTraceVersion() {}

  void Parse(const std::vector<char> data);
  std::vector<char> Dump();

 public:
  std::string version_str_;

  uint8_t major_;
  uint8_t minor_;
  uint8_t patch_;
};

class AiTraceCalAmountData {
 public:
  AiTraceCalAmountData()
      : fused_mul_add_(0),
        mul_(0),
        div_(0),
        add_(0),
        sub_(0),
        exp_(0),
        comp_(0),
        have_cal_data_(false) {}
  ~AiTraceCalAmountData() {}

  void Parse(const std::vector<char>& data);
  std::vector<char> Dump(uint8_t type);

 public:
  uint64_t fused_mul_add_;
  uint64_t mul_;
  uint64_t div_;
  uint64_t add_;
  uint64_t sub_;
  uint64_t exp_;
  uint64_t comp_;

  bool have_cal_data_;
};

class AiTraceMemory {
 public:
  AiTraceMemory() : params_(0), output_(0), have_mem_data_(false) {}
  ~AiTraceMemory() {}

  void Parse(const std::vector<char>& data);
  std::vector<char> Dump(uint8_t type);

 public:
  uint64_t params_;
  uint64_t output_;

  bool have_mem_data_;
};

class AiTraceBlock {
 public:
  AiTraceBlock() : insn_type_(0), insn_name_("") {}
  ~AiTraceBlock() {}

  void Parse(const std::vector<char>& data);
  std::vector<char> Dump();

 public:
  uint16_t insn_type_;
  std::string insn_name_;

  AiTraceCalAmountData at_cal_data_;
  AiTraceMemory at_mem_data_;
};

class AiTraceData {
 public:
  AiTraceData() {}
  ~AiTraceData() {}

  void Parse(const std::vector<char>& data);
  std::vector<char> Dump();

  void ToFile(std::string path);

 public:
  AiTraceVersion at_version_;
  std::vector<AiTraceBlock> at_block_;
};

#endif  // TVM_RELAY_ANALYSIS_PROFILER_PARSER_H_
