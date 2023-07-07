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
 * \file src/relay/backend/contrib/csinn/shl.cc
 * \brief Implementation of SHL backend.
 */

#include "shl.h"

namespace tvm {
namespace relay {
namespace contrib {
namespace csinn {

template <typename T>
void SHL::target_build(T* builder) {
  builder->compile(expr);
  code_stream_ = builder->get_ccode();
  quant_info = builder->ret_quant_info();
}

void SHL::compiler(void) {
  auto ctx = transform::PassContext::Current();
  auto cfg = ctx->GetConfig<CSINNConfig>("relay.ext.csinn.options");
  if (!cfg.defined()) {
    cfg = AttrsWithDefaultValues<CSINNConfig>();
  }
  String device = cfg.value()->target;
  bool auto_quant = cfg.value()->auto_hybrid_quantization;
  string ahead_of_time = cfg.value()->ahead_of_time;

  if (ahead_of_time != "unset") {
    /* Ahead of time compile */
    if (device == "c906" || device == "c920") {
      XuanTie_AOT* builder = new XuanTie_AOT(cfg.value()->quantization_scheme);
      builder->compile(expr);
      code_stream_ = builder->get_ccode();
    } else {
      LOG(FATAL) << "Unsupport AOT on " << device;
    }
  } else {
    /* codegen c API for JIT or interpreter */
    if (device == "th1520" && !auto_quant) {
      CodegenTH1520* builder = new CodegenTH1520;
      target_build<CodegenTH1520>(builder);
    } else if (device == "hth1520" || (device == "th1520" && auto_quant)) {
      CodegenHTH1520* builder = new CodegenHTH1520;
      target_build<CodegenHTH1520>(builder);
    } else if (device == "e907") {
      CodegenE907* builder = new CodegenE907;
      target_build<CodegenE907>(builder);
    } else if (device == "c906") {
      CodegenC906* builder = new CodegenC906;
      target_build<CodegenC906>(builder);
    } else if (device == "rvm") {
      CodegenRVM* builder = new CodegenRVM;
      target_build<CodegenRVM>(builder);
    } else if (device == "c908") {
      CodegenC908* builder = new CodegenC908;
      target_build<CodegenC908>(builder);
    } else if (device == "c920") {
      CodegenC920* builder = new CodegenC920;
      target_build<CodegenC920>(builder);
    } else if (device == "c920v2") {
      CodegenC920v2* builder = new CodegenC920v2;
      target_build<CodegenC920v2>(builder);
    } else {
      CodegenRef* builder = new CodegenRef;
      builder->SetExtFuncId(ext_func_id_);
      target_build<CodegenRef>(builder);
    }
  }
}

TVM_REGISTER_NODE_TYPE(QConfig_);
TVM_REGISTER_GLOBAL("relay.ext.csinn.QnnConfig").set_body_typed([]() {
  auto ctx = transform::PassContext::Current();
  auto cfg = ctx->GetConfig<CSINNConfig>("relay.ext.csinn.options");
  if (!cfg.defined()) {
    cfg = AttrsWithDefaultValues<CSINNConfig>();
  }
  String device = cfg.value()->target;

  CodegenCSINN* builder;
  QnnConfig qconfig;
  if (device == "th1520") {
    builder = new CodegenTH1520;
  } else if (device == "hth1520") {
    builder = new CodegenHTH1520;
  } else if (device == "e907") {
    builder = new CodegenE907;
  } else if (device == "c906") {
    builder = new CodegenC906;
  } else if (device == "rvm") {
    builder = new CodegenRVM;
  } else if (device == "c908") {
    builder = new CodegenC908;
  } else if (device == "c920") {
    builder = new CodegenC920;
  } else {
    builder = new CodegenRef;
  }

  auto n = make_object<QConfig_>(*builder->GetQuantConfig());
  qconfig = QnnConfig(n);

  return qconfig;
});

TVM_REGISTER_NODE_TYPE(QinfoNode);
TVM_REGISTER_NODE_TYPE(QuantParamsNode);

TVM_REGISTER_GLOBAL("relay.ext.csinn.GetQuantParams")
    .set_body_typed([](Array<Array<IndexExpr>> q_params, QnnConfig qnnconfig, String const_kind) {
      auto qc = new QuantCalculator;
      QuantParams* quant_params = qc->GetQuantParams(q_params, qnnconfig.operator->(), const_kind);

      QuantParamsNode qpn(quant_params);
      auto n = make_object<QuantParamsNode>(qpn);
      QuantParamsRef qr(n);
      return qr;
    });

TVM_REGISTER_GLOBAL("relay.ext.csinn.QuantizeWeight")
    .set_body_typed([](runtime::NDArray data, String target_dtype, QuantParamsRef quant_params,
                       Bool depthwise_kernel) {
      CodegenCSINN codegen;
      CSIConstant* constant =
          new CSIConstant(codegen.GetDtypeString(data.DataType()),
                          std::vector<int>(data.Shape().begin(), data.Shape().end()));
      data.CopyToBytes(constant->get_data_buf(), constant->byte_size());

      CSIConstant* quantized = codegen.CastParams(constant, target_dtype,
                                                  quant_params->GetQuantParams(), depthwise_kernel);

      Device device = {DLDeviceType::kDLCPU, 0};
      auto quantized_data =
          runtime::NDArray::Empty(data.Shape(), codegen.GetStringDtype(target_dtype), device);
      quantized_data.CopyFromBytes(quantized->get_data_buf(), quantized->byte_size());

      return quantized_data;
    });

TVM_REGISTER_GLOBAL("relay.ext.csinn.QuantizeBias")
    .set_body_typed([](runtime::NDArray data, String target_dtype,
                       QuantParamsRef input_quant_params, QuantParamsRef weight_quant_params) {
      CodegenCSINN codegen;
      CSIConstant* constant =
          new CSIConstant(codegen.GetDtypeString(data.DataType()),
                          std::vector<int>(data.Shape().begin(), data.Shape().end()));
      data.CopyToBytes(constant->get_data_buf(), constant->byte_size());

      CSIConstant* quantized =
          codegen.CastParams(constant, target_dtype, *input_quant_params->GetQuantParams(),
                             *weight_quant_params->GetQuantParams());

      Device device = {DLDeviceType::kDLCPU, 0};
      auto quantized_data =
          runtime::NDArray::Empty(data.Shape(), codegen.GetStringDtype(target_dtype), device);
      quantized_data.CopyFromBytes(quantized->get_data_buf(), quantized->byte_size());

      return quantized_data;
    });

}  // namespace csinn
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
