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
 * \file src/relay/backend/contrib/csinn/quant_cal.h
 * \brief The base class for pass.
 */
#ifndef TVM_RELAY_BACKEND_CONTRIB_CSINN_QUANT_CAL_H_
#define TVM_RELAY_BACKEND_CONTRIB_CSINN_QUANT_CAL_H_

#include <string>
#include <utility>
#include <vector>

#include "csinn.h"

namespace tvm {
namespace relay {
namespace contrib {
namespace csinn {
struct QConfig_ : public Object {
  string quantization_scheme;
  string dtype_input;
  string dtype_weight;
  string dtype_activation;
  int nbit_input;
  int nbit_weight;
  int nbit_activation;
  string activate_quantized_type;
  string weight_quantized_type;
  bool fuse_zp2bias;
  string calibrate_mode;
  double high_bound_scale;
  double low_bound_scale;

 public:
  void VisitAttrs(AttrVisitor* v) {
    v->Visit("quantization_scheme", &quantization_scheme);
    v->Visit("dtype_input", &dtype_input);
    v->Visit("dtype_weight", &dtype_weight);
    v->Visit("dtype_activation", &dtype_activation);
    v->Visit("nbit_input", &nbit_input);
    v->Visit("nbit_weight", &nbit_weight);
    v->Visit("nbit_activation", &nbit_activation);
    v->Visit("activate_quantized_type", &activate_quantized_type);
    v->Visit("weight_quantized_type", &weight_quantized_type);
    v->Visit("fuse_zp2bias", &fuse_zp2bias);
    v->Visit("calibrate_mode", &calibrate_mode);
    v->Visit("high_bound_scale", &high_bound_scale);
    v->Visit("low_bound_scale", &low_bound_scale);
  }
};

class QnnConfig : public ObjectRef {
 public:
  QnnConfig() {
    auto n = make_object<QConfig_>();
    data_ = std::move(n);
  }

  /*!
   * \brief Construct from an object pointer.
   * \param n The object pointer.
   */
  explicit QnnConfig(ObjectPtr<Object> n) : ObjectRef(n) {}

  /*! \return Mutable pointers to the node. */
  QConfig_* operator->() const {
    auto* ptr = get_mutable();
    ICHECK(ptr != nullptr);
    return static_cast<QConfig_*>(ptr);
  }
};

class QuantCalculator {
 public:
  bool is_depthwise(const std::vector<int>& ishape, const std::vector<int>& kshape, int group,
                    string target_layout);
  Array<Array<IndexExpr>> get_quant_params_expr(Array<Array<IndexExpr>> q_params, int index);
  std::vector<string> split_string(string str, string pattern);
  template <typename T>
  bool is_contain_item(std::vector<T> arr, T target_item);
  bool IsIntegralOrNot(string const_kind, string quantization_scheme);
  void GetMultiplierAndShift(double double_multiplier, int32_t* multiplier, int32_t* shift);
  virtual void GetAsymScale(float min_value, float max_value, int bits, Qinfo* qinfo, string dtype);
  virtual void GetSymScale(float min_value, float max_value, int bits, Qinfo* qinfo, QConfig_* cfg);
  QuantParams* GetQuantParamsBase(float scale, int32_t zp);
  QuantParams* GetQuantParamsBase(float min_value, float max_value, int32_t tensor_type,
                                  QConfig_* quantize_cfg);
  QuantParams* GetIntegralQuantParams(QuantParams* q_params, int32_t tensor_type, QConfig_* cfg);
  QuantParams* GetQuantParams(Array<Array<IndexExpr>> q_params, QConfig_* quantize_cfg,
                              string const_kind);
  QuantParams* RecalQuantParams(QuantParams* oquant, QConfig_* quantize_cfg);
};
}  // namespace csinn
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_BACKEND_CONTRIB_CSINN_QUANT_CAL_H_
