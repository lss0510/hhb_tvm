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
 * \file src/relay/backend/contrib/csinn/llm_quantize.cc
 * \brief The base class for LLM quantize.
 */

#include "llm_quantize.h"

namespace tvm {
namespace relay {
namespace contrib {
namespace csinn {

TVMByteArray LLMQuantizeBlock32(TVMObjectHandle data, int32_t dim_count, TVMObjectHandle dim,
                                int32_t mtype) {
  int32_t* dim_i = reinterpret_cast<int32_t*>(dim);
  struct csinn_tensor* src = csinn_alloc_tensor(NULL);
  struct csinn_tensor* ret = csinn_alloc_tensor(NULL);
  src->mtype = (enum csinn_mem_type_enum)mtype;
  src->dim_count = dim_count;
  src->data = data;
  ret->mtype = (enum csinn_mem_type_enum)mtype;
  ret->dim_count = dim_count;
  for (int i = 0; i < dim_count; i++) {
    ret->dim[i] = dim_i[i];
    src->dim[i] = dim_i[i];
  }
  if (mtype == CSINN_MEM_TYPE_BLOCK_Q8_0) {
    ret->dtype = CSINN_DTYPE_INT8;
  } else if (mtype == CSINN_MEM_TYPE_BLOCK_Q4_0) {
    ret->dtype = CSINN_DTYPE_INT4;
  } else {
    shl_debug_error("Unsupport quantize type\n");
  }
  shl_block_quantize(src, ret);
  int64_t size = 0;
  if (ret->dtype == CSINN_DTYPE_FLOAT16) {
    size = csinn_tensor_byte_size(ret);
  } else if (ret->dtype == CSINN_DTYPE_FLOAT32) {
    size = csinn_tensor_byte_size(ret);
  } else if (ret->dtype == CSINN_DTYPE_INT8 && ret->mtype == CSINN_MEM_TYPE_BLOCK_Q8_0) {
    size = csinn_tensor_size(ret) + csinn_tensor_size(ret) / 32 * sizeof(int16_t);
  } else if (ret->dtype == CSINN_DTYPE_INT4 && ret->mtype == CSINN_MEM_TYPE_BLOCK_Q4_0) {
    size = csinn_tensor_size(ret) / 2 + csinn_tensor_size(ret) / 32 * sizeof(int16_t);
  } else {
    shl_debug_error("unsupport dump data type\n");
  }
  TVMByteArray arr;
  arr.size = size;
  arr.data = (const char*)ret->data;
  return arr;
}

TVM_REGISTER_GLOBAL("relay.ext.csinn.llm_quantize_block_32")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      TVMObjectHandle data = args[0];
      int32_t dim_count = args[1];
      TVMObjectHandle dim = args[2];
      int32_t mtype = args[3];
      *ret = LLMQuantizeBlock32(std::move(data), std::move(dim_count), std::move(dim),
                                std::move(mtype));
    });
}  // namespace csinn
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
