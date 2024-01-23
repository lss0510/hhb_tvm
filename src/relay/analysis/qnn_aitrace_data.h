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
 * \file qnn_aitrace_data.h
 * \brief qnn aitrace data for profiler.
 */
#ifndef TVM_RELAY_ANALYSIS_QNN_AITRACE_DATA_H_
#define TVM_RELAY_ANALYSIS_QNN_AITRACE_DATA_H_

#include <tvm/ir/error.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/attrs/vision.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/qnn/attrs.h>
#include <tvm/relay/type.h>

#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../backend/utils.h"
#include "get_aitrace_data.h"
#include "profiler_parser.h"

namespace tvm {
namespace relay {
namespace aitrace {

template <typename T>
AiTraceDataFrame GetPoolCalAmountCommon(const Call& call_node, T attrs,
                                        Array<IndexExpr> kernel_shape, const std::string& op_type,
                                        const bool& is_global) {
  AiTraceDataFrame res;

  if (!call_node->checked_type_.defined()) {
    LOG(WARNING) << "The infer type pass should be called before the aitrace pass";
    return res;
  }
  Array<Expr> args = call_node->args;
  CHECK_EQ(args.size(), 1) << "The number of input arguments of the node should be 1.";
  std::string data_layout = attrs->layout;
  Array<IndexExpr> in_shape = args[0]->checked_type().as<TensorTypeNode>()->shape;
  Array<IndexExpr> out_shape = call_node->checked_type().as<TensorTypeNode>()->shape;

  std::unordered_map<std::string, int64_t> cal_amount = PoolCalAmountCommon(
      tvm::relay::backend::GetIntShape(in_shape), tvm::relay::backend::GetIntShape(out_shape),
      tvm::relay::backend::GetIntShape(kernel_shape), data_layout, op_type, is_global);

  CalculationAmontIndicator cai(cal_amount);
  res = cai.GetIndicatorMap();
  return res;
}

AiTraceData QnnConvert2ATData(Array<AiTraceDataFrame> origin_data);

const std::unordered_map<std::string, uint16_t> qnn_map = {
    {"qnn.csi.add", QNN_OP_ADD},
    {"qnn.csi.avgpool2d", QNN_OP_AVGPOOL_2D},
    {"qnn.csi.bn", QNN_OP_BATCH_NORM},
    {"qnn.csi.bias_add", QNN_OP_BIAS_ADD},
    {"qnn.csi.concatenate", QNN_OP_CONCATENATE},
    {"qnn.csi.conv2d", QNN_OP_CONV2D},
    {"qnn.csi.deconv2d", QNN_OP_CONV2D_TRANSPOSE},
    {"qnn.csi.dense", QNN_OP_DENSE},
    {"qnn.csi.global_avgpool2d", QNN_OP_GLOBAL_AVGPOOL_2D},
    {"qnn.csi.global_maxpool2d", QNN_OP_GLOBAL_MAXPOOL_2D},
    {"qnn.csi.lrn", QNN_OP_LRN},
    {"qnn.csi.maximum", QNN_OP_MAXIMUM},
    {"qnn.csi.maxpool2d", QNN_OP_MAXPOOL_2D},
    {"qnn.csi.maxpool2d_locat", QNN_OP_MAXPOOL_2D_LOCATION},
    {"qnn.csi.maxpool2d_with_argmax", QNN_OP_MAXPOOL_2D_WITH_ARGMAX},
    {"qnn.csi.mul", QNN_OP_MULTIPLY},
    {"qnn.csi.prelu", QNN_OP_PRELU},
    {"qnn.csi.proposal", QNN_OP_PROPOSAL},
    {"qnn.csi.psroipooling", QNN_OP_PSROIPOOLING},
    {"qnn.csi.relu", QNN_OP_RELU},
    {"qnn.csi.reshape", QNN_OP_RESHAPE},
    {"qnn.csi.roipooling", QNN_OP_ROIPOOL},
    {"qnn.csi.sigmoid", QNN_OP_SIGMOID},
    {"qnn.csi.softmax", QNN_OP_SOFTMAX},
    {"qnn.csi.split", QNN_OP_SPLIT},
    {"qnn.csi.strided_slice", QNN_OP_STRIDED_SLICE},
    {"qnn.csi.tanh", QNN_OP_TANH},
    {"qnn.csi.transpose", QNN_OP_TRANSPOSE},
    {"qnn.csi.unpooling", QNN_OP_UNPOOLING},
    {"qnn.csi.upsampling", QNN_OP_UNPSAMPLING}};

class QnnAddProfiler : public BaseProfiler {
 public:
  static AiTraceDataFrame GetCalculationAmount(const Call& node);
  static AiTraceDataFrame GetMemory(const Call& node);
};

class QnnAvgPool2dProfiler : public BaseProfiler {
 public:
  static AiTraceDataFrame GetCalculationAmount(const Call& node);
  static AiTraceDataFrame GetMemory(const Call& node);
};

class QnnBatchNormProfiler : public BaseProfiler {
 public:
  static AiTraceDataFrame GetCalculationAmount(const Call& node);
  static AiTraceDataFrame GetMemory(const Call& node);
};

class QnnBiasAddProfiler : public BaseProfiler {
 public:
  static AiTraceDataFrame GetCalculationAmount(const Call& node);
  static AiTraceDataFrame GetMemory(const Call& node);
};

class QnnConcatenateProfiler : public BaseProfiler {
 public:
  static AiTraceDataFrame GetCalculationAmount(const Call& node);
  static AiTraceDataFrame GetMemory(const Call& node);
};

class QnnConv2dProfiler : public BaseProfiler {
 public:
  static AiTraceDataFrame GetCalculationAmount(const Call& node);
  static AiTraceDataFrame GetMemory(const Call& node);
};

class QnnConv2dTranposeProfiler : public BaseProfiler {
 public:
  static AiTraceDataFrame GetCalculationAmount(const Call& node);
  static AiTraceDataFrame GetMemory(const Call& node);
};

class QnnDenseProfiler : public BaseProfiler {
 public:
  static AiTraceDataFrame GetCalculationAmount(const Call& node);
  static AiTraceDataFrame GetMemory(const Call& node);
};

class QnnGlobalAvgPool2dProfiler : public BaseProfiler {
 public:
  static AiTraceDataFrame GetCalculationAmount(const Call& node);
  static AiTraceDataFrame GetMemory(const Call& node);
};

class QnnGlobalMaxPool2dProfiler : public BaseProfiler {
 public:
  static AiTraceDataFrame GetCalculationAmount(const Call& node);
  static AiTraceDataFrame GetMemory(const Call& node);
};

class QnnLRNProfiler : public BaseProfiler {
 public:
  static AiTraceDataFrame GetCalculationAmount(const Call& node);
  static AiTraceDataFrame GetMemory(const Call& node);
};

class QnnMaximumProfiler : public BaseProfiler {
 public:
  static AiTraceDataFrame GetCalculationAmount(const Call& node);
  static AiTraceDataFrame GetMemory(const Call& node);
};

class QnnMaxPool2dProfiler : public BaseProfiler {
 public:
  static AiTraceDataFrame GetCalculationAmount(const Call& node);
  static AiTraceDataFrame GetMemory(const Call& node);
};

class QnnMaxPool2dLocationProfiler : public BaseProfiler {
 public:
  static AiTraceDataFrame GetCalculationAmount(const Call& node);
  static AiTraceDataFrame GetMemory(const Call& node);
};

class QnnMaxPool2dWithArgmaxProfiler : public BaseProfiler {
 public:
  static AiTraceDataFrame GetCalculationAmount(const Call& node);
  static AiTraceDataFrame GetMemory(const Call& node);
};

class QnnMultiplyProfiler : public BaseProfiler {
 public:
  static AiTraceDataFrame GetCalculationAmount(const Call& node);
  static AiTraceDataFrame GetMemory(const Call& node);
};

class QnnPreluProfiler : public BaseProfiler {
 public:
  static AiTraceDataFrame GetCalculationAmount(const Call& node);
  static AiTraceDataFrame GetMemory(const Call& node);
};

class QnnProposalProfiler : public BaseProfiler {
 public:
  static AiTraceDataFrame GetCalculationAmount(const Call& node);
  static AiTraceDataFrame GetMemory(const Call& node);
};

class QnnPsroipoolingProfiler : public BaseProfiler {
 public:
  static AiTraceDataFrame GetCalculationAmount(const Call& node);
  static AiTraceDataFrame GetMemory(const Call& node);
};

class QnnReluProfiler : public BaseProfiler {
 public:
  static AiTraceDataFrame GetCalculationAmount(const Call& node);
  static AiTraceDataFrame GetMemory(const Call& node);
};

class QnnReshapeProfiler : public BaseProfiler {
 public:
  static AiTraceDataFrame GetCalculationAmount(const Call& node);
  static AiTraceDataFrame GetMemory(const Call& node);
};

class QnnRoiPoolProfiler : public BaseProfiler {
 public:
  static AiTraceDataFrame GetCalculationAmount(const Call& node);
  static AiTraceDataFrame GetMemory(const Call& node);
};

class QnnSigmoidProfiler : public BaseProfiler {
 public:
  static AiTraceDataFrame GetCalculationAmount(const Call& node);
  static AiTraceDataFrame GetMemory(const Call& node);
};

class QnnSoftmaxProfiler : public BaseProfiler {
 public:
  static AiTraceDataFrame GetCalculationAmount(const Call& node);
  static AiTraceDataFrame GetMemory(const Call& node);
};

class QnnSplitProfiler : public BaseProfiler {
 public:
  static AiTraceDataFrame GetCalculationAmount(const Call& node);
  static AiTraceDataFrame GetMemory(const Call& node);
};

class QnnStridedSliceProfiler : public BaseProfiler {
 public:
  static AiTraceDataFrame GetCalculationAmount(const Call& node);
  static AiTraceDataFrame GetMemory(const Call& node);
};

class QnnTanhProfiler : public BaseProfiler {
 public:
  static AiTraceDataFrame GetCalculationAmount(const Call& node);
  static AiTraceDataFrame GetMemory(const Call& node);
};

class QnnTransposeProfiler : public BaseProfiler {
 public:
  static AiTraceDataFrame GetCalculationAmount(const Call& node);
  static AiTraceDataFrame GetMemory(const Call& node);
};

class QnnUnpoolingProfiler : public BaseProfiler {
 public:
  static AiTraceDataFrame GetCalculationAmount(const Call& node);
  static AiTraceDataFrame GetMemory(const Call& node);
};

class QnnUpsamplingProfiler : public BaseProfiler {
 public:
  static AiTraceDataFrame GetCalculationAmount(const Call& node);
  static AiTraceDataFrame GetMemory(const Call& node);
};

}  // namespace aitrace
}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_ANALYSIS_QNN_AITRACE_DATA_H_
