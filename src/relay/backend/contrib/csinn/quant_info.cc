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
 * \file src/relay/backend/contrib/csinn/quant_info.cc
 * \brief Implementation of quant_info Pass.
 */

#include <math.h>

#include "data_rearrange.h"
#include "pass.h"

namespace tvm {
namespace relay {
namespace contrib {
namespace csinn {
using namespace tvm::relay::qnn;

bool QuantCalculator::is_depthwise(const std::vector<int>& ishape, const std::vector<int>& kshape,
                                   int group, string target_layout) {
  bool is_d = false;
  if (target_layout == "NCHW" && kshape[1] == 1 && group == ishape[1]) {
    is_d = true;
  } else if (target_layout == "NHWC" && kshape[0] == 1 && group == ishape[3]) {
    is_d = true;
  }
  return is_d;
}

Array<Array<IndexExpr>> QuantCalculator::get_quant_params_expr(Array<Array<IndexExpr>> q_params,
                                                               int index) {
  Array<Array<IndexExpr>> new_params;
  new_params.push_back(q_params[index]);

  return new_params;
}

std::vector<string> QuantCalculator::split_string(string str, string pattern) {
  std::string::size_type pos;
  std::vector<std::string> result;
  // extend str for process
  str += pattern;
  size_t size = str.size();
  for (size_t i = 0; i < size; i++) {
    pos = str.find(pattern, i);
    if (pos < size) {
      std::string s = str.substr(i, pos - i);
      result.push_back(s);
      i = pos + pattern.size() - 1;
    }
  }
  return result;
}

template <typename T>
bool QuantCalculator::is_contain_item(std::vector<T> arr, T target_item) {
  for (auto item : arr) {
    if (item == target_item) {
      return true;
    }
  }
  return false;
}

bool QuantCalculator::IsIntegralOrNot(string const_kind, string quantization_scheme) {
  std::vector<string> per_channel = {"conv_kernel", "depthwise_kernel", "conv_bias",
                                     "depthwise_bias"};
  if (!is_contain_item<string>(per_channel, const_kind)) {
    return true;
  }
  return false;
}

void QuantCalculator::GetMultiplierAndShift(double double_multiplier, int32_t* multiplier,
                                            int32_t* shift) {
  int32_t significand, exponent;
  if (double_multiplier == 0) {
    *multiplier = 0;
    *shift = 0;
    return;
  }

  // Get the significand and exponent.
  double significand_d = frexp(double_multiplier, &exponent);

  // Convert the double significand to int significand, i.e., convert into a
  // integer where the decimal point is between bit 31 and 30. This is done by
  // multiplying the double value with 2^31 and then casting to int.
  significand_d = std::round(significand_d * (1ll << 31));
  int64_t significand_int64 = significand_d;
  if (significand_int64 == (1ll << 31)) {
    significand_int64 /= 2;
    ++exponent;
  }
  significand = significand_int64;
  *multiplier = significand;
  *shift = exponent;
}

void QuantCalculator::GetAsymScale(float min_value, float max_value, int bits, Qinfo* qinfo,
                                   string dtype) {
  int valid_range = std::pow(2, bits) - 1;

  if (std::isinf(min_value) || std::isinf(max_value)) {
    qinfo->scale = 1.0;
    qinfo->zero_point = 0;
    return;
  }
  max_value = std::max(max_value, 0.0f);
  min_value = std::min(min_value, 0.0f);
  if (dtype == "uint8_t") {
    qinfo->scale = (max_value - min_value) / valid_range;
    if (qinfo->scale == 0) {
      qinfo->scale = std::abs(max_value);
    }
    qinfo->zero_point = std::min(
        valid_range, static_cast<int>(std::max(0.0f, std::round(0 - min_value / qinfo->scale))));
  } else if (dtype == "int8_t") {
    qinfo->scale = (max_value - min_value) / valid_range;
    if (qinfo->scale == 0) {
      qinfo->scale = 1;
    }
    float low_bound = -std::pow(2, bits - 1);
    int high_bound = std::pow(2, bits - 1) - 1;
    qinfo->zero_point = std::min(
        high_bound,
        static_cast<int>(std::max(low_bound, std::round(-128 - min_value / qinfo->scale))));
  } else if (dtype == "int4_t") {
    qinfo->scale = (max_value - min_value) / valid_range;
    if (qinfo->scale == 0) {
      qinfo->scale = 1;
    }
    float low_bound = -127;
    int high_bound = 127;
    qinfo->zero_point =
        std::min(high_bound,
                 static_cast<int>(std::max(low_bound, std::round(-8 - min_value / qinfo->scale))));
  } else if (is_one_of<string>(dtype, {"float", "int64_t", "int32_t", "bool"})) {
    qinfo->scale = 1.0;
    qinfo->zero_point = 0;
  } else {
    LOG(ERROR) << "get error dtype:" << dtype;
  }
}

void QuantCalculator::GetSymScale(float min_value, float max_value, int bits, Qinfo* qinfo,
                                  QConfig_* cfg) {
  if (bits == 16 && is_one_of<string>(cfg->quantization_scheme,
                                      {"CSINN_QUANT_FLOAT16", "CSINN_QUANT_FLOAT16_W_INT8"})) {
    if (cfg->calibrate_mode != "scale") {
      qinfo->scale = 1.0;
      qinfo->zero_point = 0;
    } else {
      // do something
      float high_bound = 65504 * cfg->high_bound_scale;
      float low_bound = 5.96e-8 * cfg->low_bound_scale;
      CHECK(high_bound > low_bound) << "Scaled low_bound should be less than scaled hight_bound.";
      qinfo->zero_point = 0;

      /** There may be 6 situations as follows:
       *  (Min, Max) represents the range of fp16
       *  (o, x) represents the min/max values
       *
       *           Min                     Max
       *            +-----------------------+
       *    o    x  |                       |
       *    o       |            x          |
       *    o       |                       |   x
       *            |    o       x          |
       *            |    o                  |   x
       *            |                       |   o     x
       */
      if (min_value < low_bound && max_value < low_bound) {
        qinfo->scale = min_value / low_bound;
      } else if (min_value < low_bound && (max_value >= low_bound && max_value <= high_bound)) {
        qinfo->scale = 1.0;
      } else if (min_value < low_bound && max_value > high_bound) {
        qinfo->scale = max_value / high_bound;
      } else if ((min_value >= low_bound && min_value <= high_bound) &&
                 (max_value >= low_bound && max_value <= high_bound)) {
        qinfo->scale = 1.0;
      } else if ((min_value >= low_bound && min_value <= high_bound) && max_value > high_bound) {
        qinfo->scale = max_value / high_bound;
      } else if (min_value > high_bound && max_value > high_bound) {
        qinfo->scale = max_value / high_bound;
      } else {
        qinfo->scale = 1.0;
      }
    }
  } else {
    int valid_range = std::pow(2, bits - 1) - 1;
    float abs_max = std::max(std::abs(min_value), std::abs(max_value));
    qinfo->scale = abs_max / valid_range;
    qinfo->zero_point = 0;
  }
}

QuantParams* QuantCalculator::GetQuantParamsBase(float scale, int32_t zp) {
  QuantParams* q_params = new QuantParams();
  Qinfo* qinfo = new Qinfo();
  qinfo->scale = scale;
  qinfo->zero_point = zp;
  qinfo->min = 0;
  qinfo->max = 0;
  GetMultiplierAndShift(scale, &qinfo->multiplier, &qinfo->shift);
  q_params->qinfo = qinfo;
  return q_params;
}

QuantParams* QuantCalculator::GetQuantParamsBase(float min_value, float max_value,
                                                 int32_t tensor_type, QConfig_* quantize_cfg) {
  string quant_type;
  if (tensor_type == ACTIVATE) {
    quant_type = quantize_cfg->activate_quantized_type;
  } else if (tensor_type == WEIGHT) {
    quant_type = quantize_cfg->weight_quantized_type;
  }
  int bits = quantize_cfg->nbit_input;
  QuantParams* params = new QuantParams();
  Qinfo* qinfo = new Qinfo();
  qinfo->min = min_value;
  qinfo->max = max_value;
  if (quant_type == "asym") {
    GetAsymScale(min_value, max_value, bits, qinfo, quantize_cfg->dtype_input);
  } else if (quant_type == "sym") {
    GetSymScale(min_value, max_value, bits, qinfo, quantize_cfg);
  }

  if (qinfo->scale == 0) {
    qinfo->scale = 1.0;
  }

  GetMultiplierAndShift(qinfo->scale, &qinfo->multiplier, &qinfo->shift);
  params->qinfo = qinfo;

  return params;
}

QuantParams* QuantCalculator::GetIntegralQuantParams(QuantParams* q_params, int32_t tensor_type,
                                                     QConfig_* cfg) {
  if (q_params->value_type == USE_SCALE) {
    return q_params;
  }

  Qinfo* qinfo = q_params->qinfo;
  float min_value = qinfo[0].min;
  float max_value = qinfo[0].max;
  for (int i = 1; i < q_params->q_size; i++) {
    min_value = std::min(qinfo[i].min, min_value);
    max_value = std::max(qinfo[i].max, max_value);
  }
  QuantParams* ret = GetQuantParamsBase(min_value, max_value, tensor_type, cfg);
  ret->q_size = 1;
  return ret;
}

QuantParams* QuantCalculator::GetQuantParams(Array<Array<IndexExpr>> q_params,
                                             QConfig_* quantize_cfg, string const_kind) {
  int size = q_params.size();
  QuantParams* out_q_params = new QuantParams[size];
  for (int i = 0; i < size; i++) {
    auto q_param = q_params[i];
    int32_t tensor_type = q_param[0].as<IntImmNode>()->value;
    int32_t value_type = q_param[1].as<IntImmNode>()->value;
    int32_t q_type = q_param[2].as<IntImmNode>()->value;
    uint start_idx = 3;
    if (q_type == PER_TENSOR) {
      if (value_type == USE_MINMAX) {
        float min_value = q_param[start_idx].as<FloatImmNode>()->value;
        float max_value = q_param[start_idx + 1].as<FloatImmNode>()->value;
        out_q_params[i] = *GetQuantParamsBase(min_value, max_value, tensor_type, quantize_cfg);
        out_q_params[i].q_size = 1;
        out_q_params[i].value_type = value_type;
      } else if (value_type == USE_SCALE) {
        float scale = q_param[start_idx].as<FloatImmNode>()->value;
        float zp = q_param[start_idx + 1].as<IntImmNode>()->value;
        out_q_params[i] = *GetQuantParamsBase(scale, zp);
        out_q_params[i].q_size = 1;
        out_q_params[i].value_type = value_type;
      }
    } else if (q_type == PER_CHANNEL) {
      string target_kind;
      if (const_kind.find(";") != string::npos) {
        target_kind = split_string(const_kind, ";")[i];
      } else {
        target_kind = const_kind;
      }
      bool is_integral = IsIntegralOrNot(target_kind, quantize_cfg->quantization_scheme);
      // flag + single channel == 3
      uint length = (q_param.size() - 3) / 2;
      out_q_params[i] = *new QuantParams();
      Qinfo* q_infos = new Qinfo[length];
      for (uint j = start_idx; j < q_param.size(); j = j + 2) {
        int index = (j - start_idx) / 2;
        if (value_type == USE_MINMAX) {
          float min_value = q_param[j].as<FloatImmNode>()->value;
          float max_value = q_param[j + 1].as<FloatImmNode>()->value;
          QuantParams* tmp = GetQuantParamsBase(min_value, max_value, tensor_type, quantize_cfg);
          q_infos[index] = *tmp->qinfo;
        } else if (value_type == USE_SCALE) {
          q_infos[index].scale = q_param[j].as<FloatImmNode>()->value;
          q_infos[index].zero_point = q_param[j + 1].as<IntImmNode>()->value;
          q_infos[index].min = 0;
          q_infos[index].max = 0;
          int multiplier, shift;
          GetMultiplierAndShift(q_infos[index].scale, &multiplier, &shift);
        }
      }
      out_q_params[i].qinfo = q_infos;
      out_q_params[i].q_size = length;
      out_q_params[i].value_type = value_type;
      if (is_integral) {
        out_q_params[i] = *GetIntegralQuantParams(&out_q_params[i], tensor_type, quantize_cfg);
      }
    }
    out_q_params[i].dtype = quantize_cfg->dtype_weight;
  }
  return out_q_params;
}

/* TODO: only support ACTIVATE and pertensor  */
QuantParams* QuantCalculator::RecalQuantParams(QuantParams* oquant, QConfig_* quantize_cfg) {
  QuantParams* ret;
  ret = GetQuantParamsBase(oquant->qinfo->min, oquant->qinfo->max, ACTIVATE, quantize_cfg);
  ret->q_size = oquant->q_size;
  ret->shape = oquant->shape;
  ret->name = oquant->name;
  ret->dtype = quantize_cfg->dtype_weight;
  return ret;
}

class QuantInfoMutator : public HHBExprMutator {
 public:
  QuantInfoMutator() {
    auto ctx = transform::PassContext::Current();
    auto opt = ctx->GetConfig<CSINNConfig>("relay.ext.csinn.options");
    if (!opt.defined()) {
      opt = AttrsWithDefaultValues<CSINNConfig>();
    }
    auto opt_cfg = opt.value();
    layout_ = opt_cfg->layout;
    quant_calculator_ = new QuantCalculator;
  }

  explicit QuantInfoMutator(QuantCalculator* quant_calculator) {
    auto ctx = transform::PassContext::Current();
    auto opt = ctx->GetConfig<CSINNConfig>("relay.ext.csinn.options");
    if (!opt.defined()) {
      opt = AttrsWithDefaultValues<CSINNConfig>();
    }
    auto opt_cfg = opt.value();
    layout_ = opt_cfg->layout;
    quant_calculator_ = quant_calculator;
  }

  Array<Array<IndexExpr>> get_quant_params_expr(Array<Array<IndexExpr>> params, int index) {
    return quant_calculator_->get_quant_params_expr(params, index);
  }

  QuantParams* GetQuantParams(Array<Array<IndexExpr>> params, QConfig_* cfg, string const_kind) {
    return quant_calculator_->GetQuantParams(params, cfg, const_kind);
  }

  bool is_depthwise(const std::vector<int>& ishape, const std::vector<int>& kshape, int group,
                    string target_layout) {
    return quant_calculator_->is_depthwise(ishape, kshape, group, target_layout);
  }

  void change_dtype(Expr call, struct QuantParams* quant_params) {
    auto str_dtype = GetDtypeString(call->hhb_expr_extend_->dtype);
    if (is_one_of<string>(str_dtype, {"bool", "int32_t", "int64_t"})) {
      quant_params->dtype = str_dtype;
    }
  }

  void change_dtype(const CallNode* call, struct QuantParams* quant_params) {
    auto str_dtype = GetDtypeString(call->hhb_expr_extend_->dtype);
    if (is_one_of<string>(str_dtype, {"bool", "int32_t", "int64_t"})) {
      quant_params->dtype = str_dtype;
    }
  }

  /* single input single output */
  template <typename T>
  Expr SisoOp(const CallNode* call) {
    const auto* attr = call->attrs.as<T>();
    tvm::Array<Expr> call_args;
    call_args.reserve(call->args.size());
    for (auto arg : call->args) {
      auto new_arg = visit(arg);
      call_args.push_back(new_arg);
    }
    auto ret =
        Call(call->op, call_args, call->hhb_expr_extend_, call->attrs, call->type_args, call->span);

    auto input_quant =
        GetQuantParams(get_quant_params_expr(attr->q_params, 0), call->get_quant_config(), "input");
    change_dtype(call->args[0], input_quant);
    ret.push_op_quant(input_quant);

    auto output_quant = GetQuantParams(get_quant_params_expr(attr->q_params, 1),
                                       call->get_quant_config(), "output");
    change_dtype(call, output_quant);
    ret.push_op_quant(output_quant);
    ret.set_quant_config(call->get_quant_config());

    return ret;
  }

  /* double input single output */
  template <typename T>
  Expr DisoOp(const CallNode* call) {
    const auto* attr = call->attrs.as<T>();

    tvm::Array<Expr> call_args;
    call_args.reserve(call->args.size());
    for (auto arg : call->args) {
      auto new_arg = visit(arg);
      call_args.push_back(new_arg);
    }
    auto ret =
        Call(call->op, call_args, call->hhb_expr_extend_, call->attrs, call->type_args, call->span);

    auto input0_quant = GetQuantParams(get_quant_params_expr(attr->q_params, 0),
                                       call->get_quant_config(), "input0");
    change_dtype(call->args[0], input0_quant);
    ret.push_op_quant(input0_quant);
    auto input1_quant = GetQuantParams(get_quant_params_expr(attr->q_params, 1),
                                       call->get_quant_config(), "input1");

    change_dtype(call->args[1], input1_quant);
    ret.push_op_quant(input1_quant);
    auto output_quant = GetQuantParams(get_quant_params_expr(attr->q_params, 2),
                                       call->get_quant_config(), "output");
    change_dtype(call, output_quant);
    ret.push_op_quant(output_quant);
    ret.set_quant_config(call->get_quant_config());

    return ret;
  }

  /* triple input single output */
  template <typename T>
  Expr TisoOp(const CallNode* call) {
    const auto* attr = call->attrs.as<T>();

    tvm::Array<Expr> call_args;
    call_args.reserve(call->args.size());
    for (auto arg : call->args) {
      auto new_arg = visit(arg);
      call_args.push_back(new_arg);
    }
    auto ret =
        Call(call->op, call_args, call->hhb_expr_extend_, call->attrs, call->type_args, call->span);

    auto input0_quant = GetQuantParams(get_quant_params_expr(attr->q_params, 0),
                                       call->get_quant_config(), "input0");
    change_dtype(call->args[0], input0_quant);
    ret.push_op_quant(input0_quant);
    auto input1_quant = GetQuantParams(get_quant_params_expr(attr->q_params, 1),
                                       call->get_quant_config(), "input1");
    change_dtype(call->args[1], input1_quant);

    if (auto const_node = call->args[1].as<ConstantNode>()) {
      // for where softmax
      bool flag = const_node->data.DataType() == DataType::Float(32);
      flag &= const_node->get_shape().size() == 0;
      if (flag) {
        void* data_buf_ = malloc(sizeof(float));
        const_node->data.CopyToBytes(data_buf_, sizeof(float));
        flag &= std::isinf(*reinterpret_cast<float*>(data_buf_));
        input1_quant->dtype = flag ? "float" : input1_quant->dtype;
      }
    }

    ret.push_op_quant(input1_quant);
    auto input2_quant = GetQuantParams(get_quant_params_expr(attr->q_params, 2),
                                       call->get_quant_config(), "input2");
    change_dtype(call->args[2], input2_quant);

    ret.push_op_quant(input2_quant);
    auto output_quant = GetQuantParams(get_quant_params_expr(attr->q_params, 3),
                                       call->get_quant_config(), "output");
    change_dtype(call, output_quant);
    ret.push_op_quant(output_quant);
    ret.set_quant_config(call->get_quant_config());

    return ret;
  }

  template <typename T>
  Expr Conv(const CallNode* call) {
    const auto* attr = call->attrs.as<T>();

    auto ishape = call->args[0]->get_shape();
    auto wshape = call->args[1]->get_shape();
    bool depthwise = is_depthwise(ishape, wshape, attr->groups, layout_);

    tvm::Array<Expr> call_args;
    call_args.reserve(call->args.size());
    for (auto arg : call->args) {
      auto new_arg = visit(arg);
      call_args.push_back(new_arg);
    }
    auto ret =
        Call(call->op, call_args, call->hhb_expr_extend_, call->attrs, call->type_args, call->span);

    auto input0_quant = GetQuantParams(get_quant_params_expr(attr->q_params, 0),
                                       call->get_quant_config(), "input0");
    ret.push_op_quant(input0_quant);
    auto kernel_quant =
        GetQuantParams(get_quant_params_expr(attr->q_params, 1), call->get_quant_config(),
                       depthwise ? "depthwise_kernel" : "conv_kernel");
    ret.push_op_quant(kernel_quant);
    /* FIXME: fix bias */
    auto bias_quant =
        GetQuantParams(get_quant_params_expr(attr->q_params, 2), call->get_quant_config(),
                       depthwise ? "depthwise_bias" : "conv_bias");
    ret.push_op_quant(bias_quant);
    auto output_quant = GetQuantParams(get_quant_params_expr(attr->q_params, 3),
                                       call->get_quant_config(), "output");
    ret.push_op_quant(output_quant);
    ret.set_quant_config(call->get_quant_config());

    return ret;
  }

  Expr Split(const CallNode* call) {
    const auto* attr = call->attrs.as<QnnCSISplitAttrs>();
    tvm::Array<Expr> call_args;
    call_args.reserve(call->args.size());
    for (auto arg : call->args) {
      auto new_arg = visit(arg);
      call_args.push_back(new_arg);
    }
    auto ret =
        Call(call->op, call_args, call->hhb_expr_extend_, call->attrs, call->type_args, call->span);

    for (uint i = 0; i < attr->q_params.size(); i++) {
      auto quant = GetQuantParams(get_quant_params_expr(attr->q_params, i),
                                  call->get_quant_config(), "split");
      ret.push_op_quant(quant);
    }

    ret.set_quant_config(call->get_quant_config());

    return ret;
  }

  template <typename T>
  Expr Miso(const CallNode* call) {
    const auto* attr = call->attrs.as<T>();
    tvm::Array<Expr> call_args;
    call_args.reserve(call->args.size());
    for (auto arg : call->args) {
      auto new_arg = visit(arg);
      call_args.push_back(new_arg);
    }
    auto ret =
        Call(call->op, call_args, call->hhb_expr_extend_, call->attrs, call->type_args, call->span);

    for (uint i = 0; i < attr->q_params.size(); i++) {
      auto quant = GetQuantParams(get_quant_params_expr(attr->q_params, i),
                                  call->get_quant_config(), "in_out");
      ret.push_op_quant(quant);
    }

    ret.set_quant_config(call->get_quant_config());

    return ret;
  }

  template <typename T>
  Expr SiInt32soPool(const CallNode* call) {
    const auto* attr = call->attrs.as<T>();

    tvm::Array<Expr> call_args;
    call_args.reserve(call->args.size());
    for (auto arg : call->args) {
      auto new_arg = visit(arg);
      call_args.push_back(new_arg);
    }
    auto ret =
        Call(call->op, call_args, call->hhb_expr_extend_, call->attrs, call->type_args, call->span);

    auto input0_quant = GetQuantParams(get_quant_params_expr(attr->q_params, 0),
                                       call->get_quant_config(), "input0");
    ret.push_op_quant(input0_quant);
    auto input1_quant = GetQuantParams(get_quant_params_expr(attr->q_params, 1),
                                       call->get_quant_config(), "input1");
    change_dtype(call->args[1], input1_quant);
    ret.push_op_quant(input1_quant);
    auto output_quant = GetQuantParams(get_quant_params_expr(attr->q_params, 2),
                                       call->get_quant_config(), "output");
    ret.push_op_quant(output_quant);
    ret.set_quant_config(call->get_quant_config());

    return ret;
  }

  Expr Fsmn(const CallNode* call) {
    const auto* attr = call->attrs.as<QnnCSIFsmnAttrs>();

    tvm::Array<Expr> call_args;
    call_args.reserve(call->args.size());
    for (auto arg : call->args) {
      auto new_arg = visit(arg);
      call_args.push_back(new_arg);
    }
    auto ret =
        Call(call->op, call_args, call->hhb_expr_extend_, call->attrs, call->type_args, call->span);

    auto input0_quant = GetQuantParams(get_quant_params_expr(attr->q_params, 0),
                                       call->get_quant_config(), "input0");
    ret.push_op_quant(input0_quant);
    auto input1_quant = GetQuantParams(get_quant_params_expr(attr->q_params, 1),
                                       call->get_quant_config(), "input1");
    ret.push_op_quant(input1_quant);
    auto input2_quant = GetQuantParams(get_quant_params_expr(attr->q_params, 2),
                                       call->get_quant_config(), "input2");
    ret.push_op_quant(input2_quant);
    auto input3_quant = GetQuantParams(get_quant_params_expr(attr->q_params, 3),
                                       call->get_quant_config(), "input3");
    ret.push_op_quant(input3_quant);
    auto input4_quant = GetQuantParams(get_quant_params_expr(attr->q_params, 4),
                                       call->get_quant_config(), "input4");
    input4_quant->dtype = "int32_t";
    ret.push_op_quant(input4_quant);
    auto output_quant = GetQuantParams(get_quant_params_expr(attr->q_params, 3),
                                       call->get_quant_config(), "output");
    ret.push_op_quant(output_quant);
    ret.set_quant_config(call->get_quant_config());

    return ret;
  }

  Expr ScatterND(const CallNode* call) {
    const auto* attr = call->attrs.as<QnnCSIUnaryAttrs>();

    tvm::Array<Expr> call_args;
    call_args.reserve(call->args.size());
    for (auto arg : call->args) {
      auto new_arg = visit(arg);
      call_args.push_back(new_arg);
    }
    auto ret =
        Call(call->op, call_args, call->hhb_expr_extend_, call->attrs, call->type_args, call->span);

    auto input0_quant = GetQuantParams(get_quant_params_expr(attr->q_params, 0),
                                       call->get_quant_config(), "input0");
    ret.push_op_quant(input0_quant);
    auto input1_quant = GetQuantParams(get_quant_params_expr(attr->q_params, 1),
                                       call->get_quant_config(), "input1");
    input1_quant->dtype = "int32_t";
    ret.push_op_quant(input1_quant);
    auto input2_quant = GetQuantParams(get_quant_params_expr(attr->q_params, 2),
                                       call->get_quant_config(), "input2");
    ret.push_op_quant(input2_quant);

    auto output_quant = GetQuantParams(get_quant_params_expr(attr->q_params, 3),
                                       call->get_quant_config(), "output");
    ret.push_op_quant(output_quant);
    ret.set_quant_config(call->get_quant_config());

    return ret;
  }

  Expr CropResize(const CallNode* call) {
    const auto* attr = call->attrs.as<QnnCSICropResizeAttrs>();

    tvm::Array<Expr> call_args;
    call_args.reserve(call->args.size());
    for (auto arg : call->args) {
      auto new_arg = visit(arg);
      call_args.push_back(new_arg);
    }
    auto ret =
        Call(call->op, call_args, call->hhb_expr_extend_, call->attrs, call->type_args, call->span);

    auto input0_quant = GetQuantParams(get_quant_params_expr(attr->q_params, 0),
                                       call->get_quant_config(), "input0");
    ret.push_op_quant(input0_quant);
    auto input1_quant = GetQuantParams(get_quant_params_expr(attr->q_params, 1),
                                       call->get_quant_config(), "input1");
    input1_quant->dtype = "int32_t";
    ret.push_op_quant(input1_quant);
    auto input2_quant = GetQuantParams(get_quant_params_expr(attr->q_params, 2),
                                       call->get_quant_config(), "input2");
    input2_quant->dtype = "int32_t";
    ret.push_op_quant(input2_quant);

    auto output_quant = GetQuantParams(get_quant_params_expr(attr->q_params, 3),
                                       call->get_quant_config(), "output");
    ret.push_op_quant(output_quant);
    ret.set_quant_config(call->get_quant_config());

    return ret;
  }

  Expr visit_expr(const CallNode* call) {
    Expr ret;
    /* TODO: reorg attr to opt below code */
    if (IsOp(call, "qnn.csi.abs") || IsOp(call, "qnn.csi.acos") || IsOp(call, "qnn.csi.acosh") ||
        IsOp(call, "qnn.csi.asin") || IsOp(call, "qnn.csi.asinh") || IsOp(call, "qnn.csi.atan") ||
        IsOp(call, "qnn.csi.atanh") || IsOp(call, "qnn.csi.cast") || IsOp(call, "qnn.csi.ceil") ||
        IsOp(call, "qnn.csi.cos") || IsOp(call, "qnn.csi.cosh") || IsOp(call, "qnn.csi.erf") ||
        IsOp(call, "qnn.csi.exp") || IsOp(call, "qnn.csi.floor") || IsOp(call, "qnn.csi.log") ||
        IsOp(call, "qnn.csi.negative") || IsOp(call, "qnn.csi.round") ||
        IsOp(call, "qnn.csi.sign") || IsOp(call, "qnn.csi.sin") || IsOp(call, "qnn.csi.sinh") ||
        IsOp(call, "qnn.csi.sqrt") || IsOp(call, "qnn.csi.rsqrt") || IsOp(call, "qnn.csi.tan") ||
        IsOp(call, "qnn.csi.tanh") || IsOp(call, "qnn.csi.relu") || IsOp(call, "qnn.csi.relu6") ||
        IsOp(call, "qnn.csi.flatten") || IsOp(call, "qnn.csi.sigmoid")) {
      ret = SisoOp<QnnCSIUnaryAttrs>(call);
    } else if (IsOp(call, "qnn.csi.add") || IsOp(call, "qnn.csi.bias_add") ||
               IsOp(call, "qnn.csi.div") || IsOp(call, "qnn.csi.equal") ||
               IsOp(call, "qnn.csi.floor_div") || IsOp(call, "qnn.csi.floor_mod") ||
               IsOp(call, "qnn.csi.left_shift") || IsOp(call, "qnn.csi.maximum") ||
               IsOp(call, "qnn.csi.minimum") || IsOp(call, "qnn.csi.mod") ||
               IsOp(call, "qnn.csi.mul") || IsOp(call, "qnn.csi.power") ||
               IsOp(call, "qnn.csi.right_shift") || IsOp(call, "qnn.csi.subtract") ||
               IsOp(call, "qnn.csi.less")) {
      ret = DisoOp<QnnBinaryOpAttrs>(call);
    } else if (IsOp(call, "qnn.csi.argmax") || IsOp(call, "qnn.csi.argmin") ||
               IsOp(call, "qnn.csi.max") || IsOp(call, "qnn.csi.mean") ||
               IsOp(call, "qnn.csi.min") || IsOp(call, "qnn.csi.prod") ||
               IsOp(call, "qnn.csi.sum")) {
      ret = SisoOp<QnnCSIReduceAttrs>(call);
    } else if (IsOp(call, "qnn.csi.avgpool2d")) {
      ret = SisoOp<QnnCSIAvgPool2DAttrs>(call);
    } else if (IsOp(call, "qnn.csi.avgpool3d")) {
      ret = SisoOp<QnnCSIAvgPool3DAttrs>(call);
    } else if (IsOp(call, "qnn.csi.batch_to_space_nd")) {
      ret = SisoOp<QnnCSIBatchToSpaceNDAttrs>(call);
    } else if (IsOp(call, "qnn.csi.broadcast_to")) {
      ret = SisoOp<QnnCSIBroadCastToAttrs>(call);
    } else if (IsOp(call, "qnn.csi.cache_matmul")) {
      ret = TisoOp<QnnCSICacheMatMulAttrs>(call);
    } else if (IsOp(call, "qnn.csi.cache_conv1d")) {
      ret = Conv<QnnCSICacheConv1DAttrs>(call);
    } else if (IsOp(call, "qnn.csi.clip")) {
      ret = SisoOp<QnnCSIClipAttrs>(call);
    } else if (IsOp(call, "qnn.csi.concatenate")) {
      ret = Miso<QnnConcatenateAttrs>(call);
    } else if (IsOp(call, "qnn.csi.conv1d")) {
      ret = Conv<QnnCSIConv1DAttrs>(call);
    } else if (IsOp(call, "qnn.csi.conv2d") || IsOp(call, "qnn.csi.conv2d_relu") ||
               IsOp(call, "qnn.csi.conv2d_relu6")) {
      ret = Conv<QnnCSIConv2DAttrs>(call);
    } else if (IsOp(call, "qnn.csi.conv3d")) {
      ret = Conv<QnnCSIConv3DAttrs>(call);
    } else if (IsOp(call, "qnn.csi.crop_resize")) {
      ret = TisoOp<QnnCSICropResizeAttrs>(call);
    } else if (IsOp(call, "qnn.csi.deconv2d")) {
      ret = Conv<QnnCSIDeConv2DAttrs>(call);
    } else if (IsOp(call, "qnn.csi.deconv3d")) {
      ret = Conv<QnnCSIDeConv3DAttrs>(call);
    } else if (IsOp(call, "qnn.csi.dense")) {
      ret = TisoOp<QnnCSIDenseAttrs>(call);
    } else if (IsOp(call, "qnn.csi.depth_to_space")) {
      ret = SisoOp<QnnCSISubPixelAttrs>(call);
    } else if (IsOp(call, "qnn.csi.dilation2d")) {
      ret = DisoOp<QnnCSIDilation2DAttrs>(call);
    } else if (IsOp(call, "qnn.csi.expand_dims")) {
      ret = SisoOp<QnnCSIExpandDimsAttrs>(call);
    } else if (IsOp(call, "qnn.csi.fsmn")) {
      ret = Fsmn(call);
    } else if (IsOp(call, "qnn.csi.full")) {
      ret = SisoOp<QnnCSIFullAttrs>(call);
    } else if (IsOp(call, "qnn.csi.global_avgpool2d")) {
      ret = SisoOp<QnnCSIGlobalAvgPoolAttrs>(call);
    } else if (IsOp(call, "qnn.csi.global_maxpool2d")) {
      ret = SisoOp<QnnCSIGlobalMaxPoolAttrs>(call);
    } else if (IsOp(call, "qnn.csi.leaky_relu")) {
      ret = SisoOp<QnnCSILeakyReluAttrs>(call);
    } else if (IsOp(call, "qnn.csi.layer_norm")) {
      ret = TisoOp<QnnCSILayerNormAttrs>(call);
    } else if (IsOp(call, "qnn.csi.log_softmax")) {
      ret = SisoOp<QnnCSIAxisAttrs>(call);
    } else if (IsOp(call, "qnn.csi.lrn")) {
      ret = SisoOp<QnnCSILRNAttrs>(call);
    } else if (IsOp(call, "qnn.csi.maxpool3d")) {
      ret = SisoOp<QnnCSIMaxPool3DAttrs>(call);
    } else if (IsOp(call, "qnn.csi.matmul")) {
      ret = TisoOp<QnnCSIMatMulAttrs>(call);
    } else if (IsOp(call, "qnn.csi.maxpool2d")) {
      ret = SisoOp<QnnCSIMaxPool2DAttrs>(call);
    } else if (IsOp(call, "qnn.csi.maxpool2d_locat")) {
      ret = SisoOp<QnnCSIMaxPool2DLocatAttrs>(call);
    } else if (IsOp(call, "qnn.csi.maxpool2d_with_argmax")) {
      ret = SisoOp<QnnCSIMaxPool2DAttrs>(call);
    } else if (IsOp(call, "qnn.csi.pad")) {
      ret = DisoOp<QnnCSIPadAttrs>(call);
    } else if (IsOp(call, "qnn.csi.prelu")) {
      ret = DisoOp<QnnCSIPReluAttrs>(call);
    } else if (IsOp(call, "qnn.csi.proposal")) {
      ret = TisoOp<QnnCSIProposalAttrs>(call);
    } else if (IsOp(call, "qnn.csi.psroipooling")) {
      ret = SiInt32soPool<QnnCSIPSROIPoolingAttrs>(call);
    } else if (IsOp(call, "qnn.csi.reshape")) {
      ret = SisoOp<QnnCSIReshapeAttrs>(call);
    } else if (IsOp(call, "qnn.csi.reverse")) {
      ret = SisoOp<QnnCSIAxisAttrs>(call);
    } else if (IsOp(call, "qnn.csi.roipooling")) {
      ret = SiInt32soPool<QnnCSIROIPoolingAttrs>(call);
    } else if (IsOp(call, "qnn.csi.segment_max") || IsOp(call, "qnn.csi.segment_mean") ||
               IsOp(call, "qnn.csi.segment_min") || IsOp(call, "qnn.csi.segment_prod") ||
               IsOp(call, "qnn.csi.segment_sum")) {
      ret = SiInt32soPool<QnnCSISegmentAttrs>(call);
    } else if (IsOp(call, "qnn.csi.softmax")) {
      ret = SisoOp<QnnCSIAxisAttrs>(call);
    } else if (IsOp(call, "qnn.csi.space_to_batch_nd")) {
      ret = SisoOp<QnnCSISpaceToBatchNDAttrs>(call);
    } else if (IsOp(call, "qnn.csi.space_to_depth")) {
      ret = SisoOp<QnnCSISubPixelAttrs>(call);
    } else if (IsOp(call, "qnn.csi.split")) {
      ret = Split(call);
    } else if (IsOp(call, "qnn.csi.squeeze")) {
      ret = SisoOp<QnnCSISqueezeAttrs>(call);
    } else if (IsOp(call, "qnn.csi.strided_slice")) {
      ret = SisoOp<QnnCSIStridedSliceAttrs>(call);
    } else if (IsOp(call, "qnn.csi.take")) {
      ret = SiInt32soPool<QnnCSITakeAttrs>(call);
    } else if (IsOp(call, "qnn.csi.tile")) {
      ret = SisoOp<QnnCSITileAttrs>(call);
    } else if (IsOp(call, "qnn.csi.transpose")) {
      ret = SisoOp<QnnCSITransposeAttrs>(call);
    } else if (IsOp(call, "qnn.csi.unpooling")) {
      ret = SiInt32soPool<QnnCSIUnPoolingAttrs>(call);
    } else if (IsOp(call, "qnn.csi.upsampling")) {
      ret = SisoOp<QnnCSIUpSamplingAttrs>(call);
    } else if (IsOp(call, "qnn.csi.scatter_nd")) {
      ret = ScatterND(call);
    } else if (IsOp(call, "qnn.csi.one_hot")) {
      ret = SisoOp<QnnCSIOneHotAttrs>(call);
    } else if (IsOp(call, "qnn.csi.where")) {
      ret = TisoOp<QnnCSIUnaryAttrs>(call);
    } else if (IsOp(call, "qnn.csi.where_softmax")) {
      ret = TisoOp<QnnCSIWhereSoftmaxAttrs>(call);
    } else if (IsOp(call, "qnn.csi.data_convert")) {
      ret = SisoOp<QnnCSIDataConvertAttrs>(call);
    } else {
      LOG(FATAL) << "Unsupported op: " << AsText(call->op, false);
    }
    return ret;
  }

 protected:
  string layout_{""};
  QuantCalculator* quant_calculator_;
};

void QuantInfo::calculate_quant_info() {
  if (quant_calulator_) {
    QuantInfoMutator qinfo_visitor(quant_calulator_);
    expr_ = qinfo_visitor.visit(expr_);
  } else {
    QuantInfoMutator qinfo_visitor;
    expr_ = qinfo_visitor.visit(expr_);
  }
}

void QuantInfo::set_quant_calulator(QuantCalculator* quant_calulator) {
  quant_calulator_ = quant_calulator;
}

class QuantConfigMutator : public HHBExprMutator {
 public:
  explicit QuantConfigMutator(struct QConfig_* cfg) { cfg_ = cfg; }
  Expr visit_expr(const CallNode* call) {
    tvm::Array<Expr> call_args;
    call_args.reserve(call->args.size());
    for (auto arg : call->args) {
      auto new_arg = visit(arg);
      call_args.push_back(new_arg);
    }

    auto ret =
        Call(call->op, call_args, call->hhb_expr_extend_, call->attrs, call->type_args, call->span);

    ret.set_quant_config(cfg_);
    return ret;
  }

 protected:
  struct QConfig_* cfg_;
};

void QuantConfig::set_quant_config(struct QConfig_* cfg) {
  base_cfg_ = cfg;
  QuantConfigMutator qconfig_mutator(cfg);
  expr_ = qconfig_mutator.visit(expr_);
}

string get_layer_name(const Attrs& attrs) {
  string ret;
  if (auto* attr = attrs.as<QnnCSIUnaryAttrs>()) {
    ret = attr->layer_name;
  } else if (auto* attr = attrs.as<QnnCSIDataConvertAttrs>()) {
    ret = attr->layer_name;
  } else if (auto* attr = attrs.as<QnnCSIConv2DAttrs>()) {
    ret = attr->layer_name;
  } else if (auto* attr = attrs.as<QnnCSILayerNormAttrs>()) {
    ret = attr->layer_name;
  } else if (auto* attr = attrs.as<QnnCSIMatMulAttrs>()) {
    ret = attr->layer_name;
  } else if (auto* attr = attrs.as<QnnConcatenateAttrs>()) {
    ret = attr->layer_name;
  } else if (auto* attr = attrs.as<QnnBinaryOpAttrs>()) {
    ret = attr->layer_name;
  } else if (auto* attr = attrs.as<QnnCSIDenseAttrs>()) {
    ret = attr->layer_name;
  } else if (auto* attr = attrs.as<QnnCSIWhereSoftmaxAttrs>()) {
    ret = attr->layer_name;
  } else if (auto* attr = attrs.as<QnnCSIReshapeAttrs>()) {
    ret = attr->layer_name;
  } else if (auto* attr = attrs.as<QnnCSITransposeAttrs>()) {
    ret = attr->layer_name;
  } else if (auto* attr = attrs.as<QnnCSITakeAttrs>()) {
    ret = attr->layer_name;
  } else if (auto* attr = attrs.as<QnnCSIAxisAttrs>()) {
    ret = attr->layer_name;
  }

  return ret;
}

class HybridQuantConfigMutator : public HHBExprMutator {
 public:
  explicit HybridQuantConfigMutator(struct QConfig_* cfg,
                                    const std::vector<string>& hybrid_layer_name) {
    cfg_ = cfg;
    hybrid_layer_name_ = hybrid_layer_name;
  }

  template <typename T>
  bool is_contain_item(std::vector<T> arr, T target_item) {
    for (auto item : arr) {
      if (item == target_item) {
        return true;
      }
    }
    return false;
  }

  Expr visit_expr(const CallNode* call) {
    tvm::Array<Expr> call_args;
    call_args.reserve(call->args.size());
    for (auto arg : call->args) {
      auto new_arg = visit(arg);
      call_args.push_back(new_arg);
    }

    auto ret =
        Call(call->op, call_args, call->hhb_expr_extend_, call->attrs, call->type_args, call->span);

    string complete_name = get_layer_name(call->attrs);
    bool is_layer_hybrid = is_contain_item<string>(hybrid_layer_name_, complete_name);

    if (is_layer_hybrid) {
      ret.set_quant_config(cfg_);
    } else {
      ret.set_quant_config(call->get_quant_config());
    }

    return ret;
  }

 protected:
  struct QConfig_* cfg_;
  std::vector<string> hybrid_layer_name_;
};

void QuantConfig::set_hybrid_quant_config(struct QConfig_* cfg,
                                          const std::vector<string>& hybrid_layer_name) {
  hybrid_cfg_ = cfg;
  hybrid_layer_name_ = hybrid_layer_name;
  HybridQuantConfigMutator qconfig_mutator(cfg, hybrid_layer_name);
  expr_ = qconfig_mutator.visit(expr_);
}

class ConvertInserterMutator : public HHBExprMutator {
 public:
  explicit ConvertInserterMutator(struct QConfig_* cfg) {
    base_cfg_ = cfg;
    quant_calulator_ = new QuantCalculator;
  }
  ConvertInserterMutator(struct QConfig_* cfg, QuantCalculator* quant_calulator) {
    base_cfg_ = cfg;
    quant_calulator_ = quant_calulator;
  }

  Expr insert_data_convert(Expr new_arg, const CallNode* call) {
    const Op& op = Op::Get("qnn.csi.data_convert");
    auto attrs = make_object<QnnCSIDataConvertAttrs>();
    attrs->layer_name = "data_convert_hybrid" + new_arg->hhb_expr_extend_->name;
    tvm::Array<Type> type_args;
    tvm::Array<Expr> call_args;
    call_args.push_back(new_arg);
    auto ret = Call(op, call_args, new_arg->hhb_expr_extend_, Attrs(attrs), type_args, call->span);
    if (auto pre_call = new_arg.as<CallNode>()) {
      if (call->get_quant_config()->quantization_scheme !=
          pre_call->get_quant_config()->quantization_scheme) {
        /* output is last one */
        ret.push_op_quant(pre_call->get_op_quant(pre_call->get_op_quant_size() - 1));
        ret.push_op_quant(call->get_op_quant(0));
      } else {
        /* do nothing if same with previous */
        return new_arg;
      }
    } else {
      /* pre_call is var */
      ret.push_op_quant(quant_calulator_->RecalQuantParams(call->get_op_quant(0), base_cfg_));
      ret.push_op_quant(call->get_op_quant(0));
    }
    ret.set_quant_config(call->get_quant_config());

    return ret;
  }

  Expr visit_expr(const CallNode* call) {
    tvm::Array<Expr> call_args;
    call_args.reserve(call->args.size());
    for (auto arg : call->args) {
      auto new_arg = visit(arg);
      if (call->get_quant_config()->quantization_scheme != base_cfg_->quantization_scheme) {
        /* this is bybrid node */
        if (new_arg.as<CallNode>()) {
          auto inserted_arg = insert_data_convert(new_arg, call);
          call_args.push_back(inserted_arg);
        } else {
          call_args.push_back(new_arg);
        }
      } else if (auto pre_call = new_arg.as<CallNode>()) {
        if (call->get_quant_config()->quantization_scheme !=
            pre_call->get_quant_config()->quantization_scheme) {
          /* previous is bybrid node */
          auto inserted_arg = insert_data_convert(new_arg, call);
          call_args.push_back(inserted_arg);
        } else {
          call_args.push_back(new_arg);
        }
      } else {
        call_args.push_back(new_arg);
      }
    }

    auto ret =
        Call(call->op, call_args, call->hhb_expr_extend_, call->attrs, call->type_args, call->span);
    ret.set_quant_config(call->get_quant_config());
    for (auto qt : call->hhb_call_extend_->op_quant) {
      ret.push_op_quant(qt);
    }
    return ret;
  }

 protected:
  struct QConfig_* base_cfg_;
  QuantCalculator* quant_calulator_;
};

void DataConvertInserter::Insert(struct QConfig_* cfg) {
  ConvertInserterMutator qconfig_mutator(cfg);
  expr_ = qconfig_mutator.visit(expr_);
}

void DataConvertInserter::set_quant_calulator(QuantCalculator* quant_calulator) {
  quant_calulator_ = quant_calulator;
}

}  // namespace csinn
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
