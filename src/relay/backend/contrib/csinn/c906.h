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
 * \file src/relay/backend/contrib/csinn/c906.h
 * \brief The base class for c906.
 */
#ifndef TVM_RELAY_BACKEND_CONTRIB_CSINN_C906_H_
#define TVM_RELAY_BACKEND_CONTRIB_CSINN_C906_H_

#include <string>
#include <vector>

#include "gref.h"

namespace tvm {
namespace relay {
namespace contrib {
namespace csinn {
class CodegenC906 : public CodegenGref {
 public:
  CodegenC906() : CodegenGref() {
    auto ctx = transform::PassContext::Current();
    auto opt = ctx->GetConfig<CSINNConfig>("relay.ext.csinn.options");
    if (!opt.defined()) {
      opt = AttrsWithDefaultValues<CSINNConfig>();
    }
    auto opt_cfg = opt.value();
    conv2d_algorithm_ = opt_cfg->conv2d_algorithm;
  }

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

  virtual CSIConstant* CastParams(CSIConstant* data, string target_dtype, QuantParams* quant_params,
                                  bool depthwise_kernel);
  virtual void Conv2d(const CallNode* call, string op_name);
  virtual void Dense(const CallNode* call);
  void EmitNBGSetup(void) {}

  bool is_gemm_kernel(bool depthwise, CSINNOP* op) {
    bool ret = false;
    if (conv2d_algorithm_ == "gemm") {
      if (!depthwise) {
        CSINNConstantTensor* ct = op->get_constant(0);
        if (ct->tensor->dtype == CSINN_DTYPE_FLOAT16) {
          ret = true;
        }
      }
    }
    return ret;
  }

  void reorder_kernel_fp16(int16_t* a, int16_t* sa, int m, int k, int ldx) {
    int i = 0;
    for (; i + 7 < m; i += 8) {
      int16_t* p0 = a;
      int16_t* p1 = a + ldx;
      int16_t* p2 = a + 2 * ldx;
      int16_t* p3 = a + 3 * ldx;
      int16_t* p4 = a + 4 * ldx;
      int16_t* p5 = a + 5 * ldx;
      int16_t* p6 = a + 6 * ldx;
      int16_t* p7 = a + 7 * ldx;
      int j = 0;
      for (; j + 7 < k; j += 8) {
        for (int ii = 0; ii < 8; ii++) {
          sa[0 + 8 * ii] = p0[ii];
          sa[1 + 8 * ii] = p1[ii];
          sa[2 + 8 * ii] = p2[ii];
          sa[3 + 8 * ii] = p3[ii];
          sa[4 + 8 * ii] = p4[ii];
          sa[5 + 8 * ii] = p5[ii];
          sa[6 + 8 * ii] = p6[ii];
          sa[7 + 8 * ii] = p7[ii];
        }

        sa += 64;
        p0 += 8;
        p1 += 8;
        p2 += 8;
        p3 += 8;
        p4 += 8;
        p5 += 8;
        p6 += 8;
        p7 += 8;
      }
      if (j + 3 < k) {
        j += 4;
        for (int ii = 0; ii < 4; ii++) {
          sa[0 + 8 * ii] = p0[ii];
          sa[1 + 8 * ii] = p1[ii];
          sa[2 + 8 * ii] = p2[ii];
          sa[3 + 8 * ii] = p3[ii];
          sa[4 + 8 * ii] = p4[ii];
          sa[5 + 8 * ii] = p5[ii];
          sa[6 + 8 * ii] = p6[ii];
          sa[7 + 8 * ii] = p7[ii];
        }

        sa += 32;
        p0 += 4;
        p1 += 4;
        p2 += 4;
        p3 += 4;
        p4 += 4;
        p5 += 4;
        p6 += 4;
        p7 += 4;
      }
      if (j + 1 < k) {
        j += 2;
        sa[0] = p0[0];
        sa[1] = p1[0];
        sa[2] = p2[0];
        sa[3] = p3[0];
        sa[4] = p4[0];
        sa[5] = p5[0];
        sa[6] = p6[0];
        sa[7] = p7[0];
        sa[8] = p0[1];
        sa[9] = p1[1];
        sa[10] = p2[1];
        sa[11] = p3[1];
        sa[12] = p4[1];
        sa[13] = p5[1];
        sa[14] = p6[1];
        sa[15] = p7[1];

        sa += 16;
        p0 += 2;
        p1 += 2;
        p2 += 2;
        p3 += 2;
        p4 += 2;
        p5 += 2;
        p6 += 2;
        p7 += 2;
      }
      if (j < k) {
        sa[0] = p0[0];
        sa[1] = p1[0];
        sa[2] = p2[0];
        sa[3] = p3[0];
        sa[4] = p4[0];
        sa[5] = p5[0];
        sa[6] = p6[0];
        sa[7] = p7[0];

        sa += 8;
      }
      a += 8 * ldx;
    }
    if (i + 3 < m) {
      i += 4;
      int16_t* p0 = a;
      int16_t* p1 = a + ldx;
      int16_t* p2 = a + 2 * ldx;
      int16_t* p3 = a + 3 * ldx;
      int j = 0;
      for (; j + 7 < k; j += 8) {
        sa[0] = p0[0];
        sa[16] = p0[4];
        sa[1] = p1[0];
        sa[17] = p1[4];
        sa[2] = p2[0];
        sa[18] = p2[4];
        sa[3] = p3[0];
        sa[19] = p3[4];

        sa[4] = p0[1];
        sa[20] = p0[5];
        sa[5] = p1[1];
        sa[21] = p1[5];
        sa[6] = p2[1];
        sa[22] = p2[5];
        sa[7] = p3[1];
        sa[23] = p3[5];

        sa[8] = p0[2];
        sa[24] = p0[6];
        sa[9] = p1[2];
        sa[25] = p1[6];
        sa[10] = p2[2];
        sa[26] = p2[6];
        sa[11] = p3[2];
        sa[27] = p3[6];

        sa[12] = p0[3];
        sa[28] = p0[7];
        sa[13] = p1[3];
        sa[29] = p1[7];
        sa[14] = p2[3];
        sa[30] = p2[7];
        sa[15] = p3[3];
        sa[31] = p3[7];

        sa += 32;
        p0 += 8;
        p1 += 8;
        p2 += 8;
        p3 += 8;
      }
      if (j + 3 < k) {
        j += 4;
        sa[0] = p0[0];
        sa[8] = p0[2];
        sa[1] = p1[0];
        sa[9] = p1[2];
        sa[2] = p2[0];
        sa[10] = p2[2];
        sa[3] = p3[0];
        sa[11] = p3[2];

        sa[4] = p0[1];
        sa[12] = p0[3];
        sa[5] = p1[1];
        sa[13] = p1[3];
        sa[6] = p2[1];
        sa[14] = p2[3];
        sa[7] = p3[1];
        sa[15] = p3[3];

        sa += 16;
        p0 += 4;
        p1 += 4;
        p2 += 4;
        p3 += 4;
      }
      if (j + 1 < k) {
        j += 2;
        sa[0] = p0[0];
        sa[1] = p1[0];
        sa[2] = p2[0];
        sa[3] = p3[0];

        sa[4] = p0[1];
        sa[5] = p1[1];
        sa[6] = p2[1];
        sa[7] = p3[1];

        sa += 8;
        p0 += 2;
        p1 += 2;
        p2 += 2;
        p3 += 2;
      }
      if (j < k) {
        sa[0] = p0[0];
        sa[1] = p1[0];
        sa[2] = p2[0];
        sa[3] = p3[0];

        sa += 4;
      }
      a += 4 * ldx;
    }
    if (i + 1 < m) {
      i += 2;
      int16_t* p0 = a;
      int16_t* p1 = a + ldx;

      int j = 0;
      for (; j + 7 < k; j += 8) {
        sa[0] = p0[0];
        sa[1] = p1[0];
        sa[2] = p0[1];
        sa[3] = p1[1];
        sa[4] = p0[2];
        sa[5] = p1[2];
        sa[6] = p0[3];
        sa[7] = p1[3];
        sa[8] = p0[4];
        sa[9] = p1[4];
        sa[10] = p0[5];
        sa[11] = p1[5];
        sa[12] = p0[6];
        sa[13] = p1[6];
        sa[14] = p0[7];
        sa[15] = p1[7];

        sa += 16;
        p0 += 8;
        p1 += 8;
      }
      if (j + 3 < k) {
        j += 4;
        sa[0] = p0[0];
        sa[1] = p1[0];
        sa[2] = p0[1];
        sa[3] = p1[1];
        sa[4] = p0[2];
        sa[5] = p1[2];
        sa[6] = p0[3];
        sa[7] = p1[3];

        sa += 8;
        p0 += 4;
        p1 += 4;
      }
      if (j + 1 < k) {
        j += 2;
        sa[0] = p0[0];
        sa[1] = p1[0];
        sa[2] = p0[1];
        sa[3] = p1[1];

        sa += 4;
        p0 += 2;
        p1 += 2;
      }
      if (j < k) {
        sa[0] = p0[0];
        sa[1] = p1[0];

        sa += 2;
      }
      a += 2 * ldx;
    }
    if (i < m) {
      memcpy(sa, a, sizeof(int16_t) * ldx);
    }
  }

  void reorder_kernel(CSINNConstantTensor* ctensor, int group) {
    int16_t* kernel_data = static_cast<int16_t*>(ctensor->const_data);
    struct csinn_tensor* kernel = ctensor->tensor;
    int m = kernel->dim[0] / group;  // m = out_ch / group
    int k = kernel->dim[1] * kernel->dim[2] * kernel->dim[3];

    int16_t* pa_reorder = static_cast<int16_t*>(calloc(1, ctensor->get_binary_model_const_size()));
    for (int g = 0; g < group; g++) {
      /* align to shl_c906_reorder_kernel_fp16 */
      reorder_kernel_fp16(kernel_data + g * m * k, pa_reorder + g * m * k, m, k, k);
    }

    memcpy(kernel_data, pa_reorder, group * m * k * sizeof(int16_t));
    free(pa_reorder);
  }

  void reorder_fcl_fp16(int16_t* src, int16_t* dst, int m, int k, int ldx) {
    int i = 0;
    for (; i + 15 < m; i += 16) {
      for (int j = 0; j < k; j++) {
        dst[i * k + 16 * j + 0] = src[(i + 0) * k + j];
        dst[i * k + 16 * j + 1] = src[(i + 1) * k + j];
        dst[i * k + 16 * j + 2] = src[(i + 2) * k + j];
        dst[i * k + 16 * j + 3] = src[(i + 3) * k + j];
        dst[i * k + 16 * j + 4] = src[(i + 4) * k + j];
        dst[i * k + 16 * j + 5] = src[(i + 5) * k + j];
        dst[i * k + 16 * j + 6] = src[(i + 6) * k + j];
        dst[i * k + 16 * j + 7] = src[(i + 7) * k + j];
        dst[i * k + 16 * j + 8] = src[(i + 8) * k + j];
        dst[i * k + 16 * j + 9] = src[(i + 9) * k + j];
        dst[i * k + 16 * j + 10] = src[(i + 10) * k + j];
        dst[i * k + 16 * j + 11] = src[(i + 11) * k + j];
        dst[i * k + 16 * j + 12] = src[(i + 12) * k + j];
        dst[i * k + 16 * j + 13] = src[(i + 13) * k + j];
        dst[i * k + 16 * j + 14] = src[(i + 14) * k + j];
        dst[i * k + 16 * j + 15] = src[(i + 15) * k + j];
      }
    }
    dst += i * k;
    src += i * k;
    for (; i < m; i++) {
      memcpy(dst, src, sizeof(int16_t) * ldx);
      dst += k;
      src += k;
    }
  }

  void reorder_fcl(CSINNConstantTensor* ctensor) {
    int16_t* weight_data = static_cast<int16_t*>(ctensor->const_data);
    struct csinn_tensor* weight = ctensor->tensor;

    int n = weight->dim[0];
    int k = weight->dim[1];

    int16_t* pa_reorder = static_cast<int16_t*>(calloc(1, ctensor->get_binary_model_const_size()));

    reorder_fcl_fp16(weight_data, pa_reorder, n, k, k);
    memcpy(weight_data, pa_reorder, n * k * sizeof(int16_t));
    free(pa_reorder);
  }

  string conv2d_algorithm_;
};

}  // namespace csinn
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_BACKEND_CONTRIB_CSINN_C906_H_
