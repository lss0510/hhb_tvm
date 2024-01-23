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
 * \file src/relay/backend/contrib/csinn/rvm.h
 * \brief The base class for RISC-V matrix extension.
 */
#ifndef TVM_RELAY_BACKEND_CONTRIB_CSINN_RVM_H_
#define TVM_RELAY_BACKEND_CONTRIB_CSINN_RVM_H_

#include <string>
#include <vector>

#include "gref.h"

namespace tvm {
namespace relay {
namespace contrib {
namespace csinn {
class CodegenRVM final : public CodegenGref {
 public:
  CodegenRVM() : CodegenGref() {
    auto ctx = transform::PassContext::Current();
    auto opt = ctx->GetConfig<CSINNConfig>("relay.ext.csinn.options");
    if (!opt.defined()) {
      opt = AttrsWithDefaultValues<CSINNConfig>();
    }
    auto opt_cfg = opt.value();
    mlen_ = opt_cfg->matrix_extension_mlen;
    conv2d_algorithm_ = opt_cfg->conv2d_algorithm;
  }

  void Conv2d(const CallNode* call, string op_name) override;
  CSIConstant* CastParams(CSIConstant* data, string target_dtype, QuantParams* quant_params,
                          bool depthwise_kernel) override;

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

 private:
  int csrr_xmlenb() { return mlen_ / 8; }

  void reorder_kernel_int8(int8_t* src, int8_t* dst, int out_c, int in_c, int maxk) {
    const int col = csrr_xmlenb();
    const int row = col / 4;
    int oc = 0;
    for (; oc + 2 * row <= out_c; oc += 2 * row) {
      int8_t* src_m = src + oc * in_c * maxk;
      int k = 0;
      for (; k + col <= maxk * in_c; k += col) {
        int8_t* src_n = src_m + k;
        for (int i = 0; i < 2 * row; i++) {
          int8_t* src_i = src_n + i * in_c * maxk;
          memcpy(dst, src_i, col * sizeof(int8_t));
          dst += col;
        }
      }
      // k_tail
      if (k < maxk * in_c) {
        int8_t* src_n = src_m + k;
        for (int i = 0; i < 2 * row; i++) {
          int8_t* src_i = src_n + i * in_c * maxk;
          memcpy(dst, src_i, (maxk * in_c - k) * sizeof(int8_t));
          dst += col;
        }
      }
    }
    for (; oc + row <= out_c; oc += row) {
      int8_t* src_m = src + oc * in_c * maxk;
      int k = 0;
      for (; k + col <= maxk * in_c; k += col) {
        int8_t* src_n = src_m + k;
        for (int i = 0; i < row; i++) {
          int8_t* src_i = src_n + i * in_c * maxk;
          memcpy(dst, src_i, col * sizeof(int8_t));
          dst += col;
        }
      }
      if (k < maxk * in_c) {
        int8_t* src_n = src_m + k;
        for (int i = 0; i < row; i++) {
          int8_t* src_i = src_n + i * in_c * maxk;
          memcpy(dst, src_i, (maxk * in_c - k) * sizeof(int8_t));
          dst += col;
        }
      }
    }
    // oc_tail
    if (oc < out_c) {
      int8_t* src_m = src + oc * in_c * maxk;
      int k = 0;
      for (; k + col <= maxk * in_c; k += col) {
        int8_t* src_n = src_m + k;
        for (int i = 0; i < (out_c - oc); i++) {
          int8_t* src_i = src_n + i * in_c * maxk;
          memcpy(dst, src_i, col * sizeof(int8_t));
          dst += col;
        }
        dst += (oc + row - out_c) * col;  // padding
      }
      if (k < maxk * in_c) {
        int8_t* src_n = src_m + k;
        for (int i = 0; i < (out_c - oc); i++) {
          int8_t* src_i = src_n + i * in_c * maxk;
          memcpy(dst, src_i, (maxk * in_c - k) * sizeof(int8_t));
          dst += col;
        }
      }
    }
  }

  void reorder_kernel_fp16(int16_t* src, int16_t* dst, int out_c, int in_c, int maxk) {
    int m2rows = csrr_xmlenb() / 2;
    int cols = m2rows;
    int K = maxk * in_c;
    int oc = 0;
    for (; oc + m2rows <= out_c; oc += m2rows) {
      int16_t* src_m = src + oc * K;
      int j = 0;
      for (; j + cols - 1 < K; j += cols) {
        int16_t* src_n = src_m + j;
        for (int i = 0; i < m2rows; i++) {
          int16_t* src_i = src_n + i * K;
          memcpy(dst, src_i, cols * sizeof(int16_t));
          dst += cols;
        }
      }
      // k_tail
      if (j < K) {
        int16_t* src_n = src_m + j;
        for (int i = 0; i < m2rows; i++) {
          int16_t* src_i = src_n + i * K;
          memcpy(dst, src_i, (K - j) * sizeof(int16_t));
          dst += cols;
        }
      }
    }
    // oc_tail
    if (oc < out_c) {
      int16_t* src_m = src + oc * K;
      int j = 0;
      for (; j + cols - 1 < K; j += cols) {
        int16_t* src_n = src_m + j;
        for (int i = 0; i < (out_c - oc); i++) {
          int16_t* src_i = src_n + i * K;
          memcpy(dst, src_i, cols * sizeof(int16_t));
          dst += cols;
        }
        dst += (oc + m2rows - out_c) * cols;  // padding
      }
      // k_tail
      if (j < K) {
        int16_t* src_n = src_m + j;
        for (int i = 0; i < (out_c - oc); i++) {
          int16_t* src_i = src_n + i * K;
          memcpy(dst, src_i, (K - j) * sizeof(int16_t));
          dst += cols;
        }
      }
    }
  }

  void reorder_kernel(CSINNConstantTensor* ctensor, int group) {
    struct csinn_tensor* kernel = ctensor->tensor;

    for (int g = 0; g < group; g++) {
      if (ctensor->tensor->dtype == CSINN_DTYPE_FLOAT16) {
        int out_c = kernel->dim[0];
        int out_cp = out_c / group;  // per-group out channel
        int in_c = kernel->dim[3];
        int maxk = kernel->dim[1] * kernel->dim[2];
        int oc_per_group_align = ((out_cp - 1) & -(csrr_xmlenb() / 2)) + csrr_xmlenb() / 2;
        int k_align = ((in_c * maxk - 1) & -(csrr_xmlenb() / 2)) + csrr_xmlenb() / 2;
        int realloc_size = group * oc_per_group_align * k_align * sizeof(int16_t);
        int16_t* kernel_data = static_cast<int16_t*>(ctensor->const_data);
        int16_t* pa_reorder = static_cast<int16_t*>(calloc(1, realloc_size));
        /* align to im2col_gemm_reorder_kernel_nhwc_per_group_fp16 */
        int16_t* ker_ptr = kernel_data + g * out_cp * in_c * maxk;
        int16_t* ker_tm_ptr = pa_reorder + g * oc_per_group_align * k_align;
        reorder_kernel_fp16(ker_ptr, ker_tm_ptr, out_cp, in_c, maxk);

        ctensor->const_data = pa_reorder;
        ctensor->set_const_data_size(realloc_size);
        free(kernel_data);
      } else if (ctensor->tensor->dtype == CSINN_DTYPE_INT8) {
        int out_c = kernel->dim[0];
        int out_cp = out_c / group;  // per-group out channel
        int in_c = kernel->dim[3];
        int maxk = kernel->dim[1] * kernel->dim[2];
        int oc_per_group_align = ((out_cp - 1) & -(csrr_xmlenb() / 4)) + csrr_xmlenb() / 4;
        int k_align = ((in_c * maxk - 1) & -csrr_xmlenb()) + csrr_xmlenb();
        int realloc_size = group * oc_per_group_align * k_align * sizeof(int8_t);
        int8_t* kernel_data = static_cast<int8_t*>(ctensor->const_data);
        int8_t* pa_reorder = static_cast<int8_t*>(calloc(1, realloc_size));

        /* align to im2col_gemm_reorder_kernel_per_group_int8_matrix */
        int8_t* ker_ptr = kernel_data + g * out_cp * in_c * maxk;
        int8_t* ker_tm_ptr = pa_reorder + g * oc_per_group_align * k_align;
        reorder_kernel_int8(ker_ptr, ker_tm_ptr, out_cp, in_c, maxk);

        ctensor->const_data = pa_reorder;
        ctensor->set_const_data_size(realloc_size);
        free(kernel_data);
      } else {
        LOG(ERROR) << "Error dtype to reorder kernel";
      }
    }
  }

  bool is_gemm_kernel(bool depthwise, CSINNOP* op) {
    bool ret = false;
    if (conv2d_algorithm_ == "gemm") {
      if (!depthwise) {
        CSINNConstantTensor* ct = op->get_constant(0);
        if (ct->tensor->dtype == CSINN_DTYPE_FLOAT16 || ct->tensor->dtype == CSINN_DTYPE_INT8) {
          ret = true;
        }
      }
    }
    return ret;
  }

  int mlen_{0};
  string conv2d_algorithm_;
};

}  // namespace csinn
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_BACKEND_CONTRIB_CSINN_RVM_H_
