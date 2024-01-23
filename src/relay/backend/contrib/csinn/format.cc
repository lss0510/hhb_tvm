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
 * \file src/relay/backend/contrib/csinn/format.cc
 * \brief Implementation of CSINN codegen formater.
 */

#include "format.h"

namespace tvm {
namespace relay {
namespace contrib {
namespace csinn {

string CSINNTensor::to_layout(int32_t layout) {
  string ret;
  if (layout == CSINN_LAYOUT_N) {
    return "CSINN_LAYOUT_N";
  } else if (layout == CSINN_LAYOUT_NC) {
    return "CSINN_LAYOUT_NC";
  } else if (layout == CSINN_LAYOUT_NCW) {
    return "CSINN_LAYOUT_NCW";
  } else if (layout == CSINN_LAYOUT_NWC) {
    return "CSINN_LAYOUT_NWC";
  } else if (layout == CSINN_LAYOUT_NCHW) {
    return "CSINN_LAYOUT_NCHW";
  } else if (layout == CSINN_LAYOUT_NHWC) {
    return "CSINN_LAYOUT_NHWC";
  } else if (layout == CSINN_LAYOUT_NCDHW) {
    return "CSINN_LAYOUT_NCDHW";
  } else if (layout == CSINN_LAYOUT_NDHWC) {
    return "CSINN_LAYOUT_NDHWC";
  } else if (layout == CSINN_LAYOUT_NLCDHW) {
    return "CSINN_LAYOUT_NLCDHW";
  } else if (layout == CSINN_LAYOUT_O) {
    return "CSINN_LAYOUT_O";
  } else if (layout == CSINN_LAYOUT_OI) {
    return "CSINN_LAYOUT_OI";
  } else if (layout == CSINN_LAYOUT_OIW) {
    return "CSINN_LAYOUT_OIW";
  } else if (layout == CSINN_LAYOUT_OWI) {
    return "CSINN_LAYOUT_OWI";
  } else if (layout == CSINN_LAYOUT_OIHW) {
    return "CSINN_LAYOUT_OIHW";
  } else if (layout == CSINN_LAYOUT_OHWI) {
    return "CSINN_LAYOUT_OHWI";
  } else if (layout == CSINN_LAYOUT_OIDHW) {
    return "CSINN_LAYOUT_OIDHW";
  } else if (layout == CSINN_LAYOUT_ODHWI) {
    return "CSINN_LAYOUT_ODHWI";
  } else if (layout == CSINN_LAYOUT_1HWO) {
    return "CSINN_LAYOUT_1HWO";
  } else if (layout == CSINN_LAYOUT_O1HW) {
    return "CSINN_LAYOUT_O1HW";
  } else if (layout == CSINN_LAYOUT_O32I32) {
    return "CSINN_LAYOUT_O32I32";
  } else if (layout == CSINN_LAYOUT_O32HWI32) {
    return "CSINN_LAYOUT_O32HWI32";
  } else if (layout == CSINN_LAYOUT_IOHW) {
    return "CSINN_LAYOUT_IOHW";
  } else {
    return "CSINN_LAYOUT_NULL";
  }

  return ret;
}

string CSINNTensor::to_dtype(enum csinn_dtype_enum dtype) {
  string ret;
  if (dtype == CSINN_DTYPE_INT4) {
    return "CSINN_DTYPE_INT4";
  } else if (dtype == CSINN_DTYPE_UINT8) {
    return "CSINN_DTYPE_UINT8";
  } else if (dtype == CSINN_DTYPE_INT8) {
    return "CSINN_DTYPE_INT8";
  } else if (dtype == CSINN_DTYPE_FLOAT32) {
    return "CSINN_DTYPE_FLOAT32";
  } else if (dtype == CSINN_DTYPE_FLOAT16) {
    return "CSINN_DTYPE_FLOAT16";
  } else if (dtype == CSINN_DTYPE_BFLOAT16) {
    return "CSINN_DTYPE_BFLOAT16";
  } else if (dtype == CSINN_DTYPE_INT32) {
    return "CSINN_DTYPE_INT32";
  } else if (dtype == CSINN_DTYPE_BOOL) {
    return "CSINN_DTYPE_BOOL";
  } else if (dtype == CSINN_DTYPE_INT16) {
    return "CSINN_DTYPE_INT16";
  } else if (dtype == CSINN_DTYPE_INT64) {
    return "CSINN_DTYPE_INT64";
  }
  return ret;
}

string CSINNTensor::to_mtype(enum csinn_mem_type_enum mtype) {
  string ret = "";
  if (mtype == CSINN_MEM_TYPE_CPU_NOT_ALIGNED) {
    return "CSINN_MEM_TYPE_CPU_NOT_ALIGNED";
  } else if (mtype == CSINN_MEM_TYPE_CPU_ALIGNED) {
    return "CSINN_MEM_TYPE_CPU_ALIGNED";
  } else if (mtype == CSINN_MEM_TYPE_DMABUF) {
    return "CSINN_MEM_TYPE_DMABUF";
  } else if (mtype == CSINN_MEM_TYPE_ASP42) {
    return "CSINN_MEM_TYPE_ASP42";
  } else if (mtype == CSINN_MEM_TYPE_ASP41) {
    return "CSINN_MEM_TYPE_ASP41";
  } else {
    LOG(ERROR) << "Get error mtype convert:" << mtype;
  }
  return ret;
}

std::vector<string> CSINNTensor::serialize(size_t qoffset, size_t coffset) {
  std::ostringstream t0;
  t0 << "struct csinn_tensor *" << name << " = csinn_alloc_tensor(sess)";
  push_str(t0);
  string t_name = org_name == "" ? name : org_name;
  t0 << name << "->name = "
     << "\"" << t_name << "\"";
  push_str(t0);
  if (tensor->is_const) {
    t0 << name << "->data = params_base + " << std::to_string(coffset);
    push_str(t0);
    t0 << name << "->is_const = 1";
    push_str(t0);

    if (tensor->mtype != CSINN_MEM_TYPE_CPU_NOT_ALIGNED) {
      t0 << name << "->mtype = " << to_mtype(tensor->mtype);
      push_str(t0);
    }
  }
  t0 << name << "->dtype = " << to_dtype(tensor->dtype);
  push_str(t0);
  t0 << name << "->layout = " << to_layout(tensor->layout);
  push_str(t0);
  for (int i = 0; i < tensor->dim_count; i++) {
    t0 << name << "->dim[" << i << "] = " << tensor->dim[i];
    push_str(t0);
  }
  t0 << name << "->dim_count = " << tensor->dim_count;
  push_str(t0);

  if (tensor->quant_channel != 1) {
    t0 << "csinn_realloc_quant_info(" << name << ", " << std::to_string(tensor->quant_channel)
       << ")";
    push_str(t0);
  }

  t0 << "memcpy(" << name << "->qinfo"
     << ", params_base + " << std::to_string(qoffset) << ", sizeof(struct csinn_quant_info) * "
     << std::to_string(tensor->quant_channel) << ")";
  push_str(t0);

  for (string s : astr) {
    str.push_back(s);
  }
  return str;
}

void CSINNVarTensor::to_file(std::ofstream& file) {
  /* write quant data to file */
  char* quant_buf = static_cast<char*>(calloc(1, quant_mem_->get_space_size()));
  memcpy(quant_buf + quant_mem_->get_space_size() - quant_mem_->get_size(), quant_data,
         quant_mem_->get_size());
  file.write(quant_buf, quant_mem_->get_space_size());
  free(quant_buf);
}

void CSINNConstantTensor::to_file(std::ofstream& file) {
  /* write quant data to file */
  char* quant_buf = static_cast<char*>(calloc(1, quant_mem_->get_space_size()));
  memcpy(quant_buf + quant_mem_->get_space_size() - quant_mem_->get_size(), quant_data,
         quant_mem_->get_size());
  file.write(quant_buf, quant_mem_->get_space_size());
  free(quant_buf);

  /* write const data to file */
  char* const_buf = static_cast<char*>(calloc(1, const_mem_->get_space_size()));
  memcpy(const_buf + const_mem_->get_space_size() - const_mem_->get_size(), const_data,
         const_mem_->get_size());
  file.write(const_buf, const_mem_->get_space_size());
  free(const_buf);

  /* write sparse to file */
  if (tensor->mtype == CSINN_MEM_TYPE_ASP42) {
    file.write(sparse_index_data, binary_model_sparse_index_size);
  } else if (tensor->mtype == CSINN_MEM_TYPE_ASP41) {
    file.write(sparse_index_data, binary_model_sparse_index_size);
  }
}

void CSINNOP::push_input(CSINNVarTensor* in) { inputs.push_back(in); }

void CSINNOP::push_output(CSINNVarTensor* out) { outputs.push_back(out); }

void CSINNOP::push_constant(CSINNConstantTensor* cst) { consts.push_back(cst); }

CSINNTensor* CSINNOP::get_tensor(string name) {
  for (auto tensor : inputs) {
    if (tensor->name == name) {
      return tensor;
    }
  }
  for (auto tensor : outputs) {
    if (tensor->name == name) {
      return tensor;
    }
  }
  for (auto tensor : consts) {
    if (tensor->name == name) {
      return tensor;
    }
  }
  return NULL;
}

std::vector<string> CSINNOP::serialize() {
  size_t qoffset = op_binary_model_base;
  size_t coffset = op_binary_model_base;
  for (auto tensor : inputs) {
    if (!tensor->dumped) {
      std::vector<string> str = tensor->serialize(qoffset, coffset);
      tensor->set_binary_model_qinfo_start(qoffset);
      for (auto s : str) {
        strs.push_back(s);
      }
      op_size += tensor->bm_size();
      coffset = op_binary_model_base + op_size;
      qoffset = op_binary_model_base + op_size;
      tensor->dumped = true;
    }
  }
  for (auto tensor : outputs) {
    std::vector<string> str = tensor->serialize(qoffset, coffset);
    tensor->set_binary_model_qinfo_start(qoffset);
    for (auto s : str) {
      strs.push_back(s);
    }
    op_size += tensor->bm_size();
    coffset = op_binary_model_base + op_size;
    qoffset = op_binary_model_base + op_size;
  }
  for (auto tensor : consts) {
    coffset += tensor->get_binary_model_qinfo_space_size();
    /* set original start */
    tensor->set_binary_model_const_start(coffset);
    /* get real start */
    size_t real_coffset = tensor->get_binary_model_const_real_start();
    std::vector<string> str = tensor->serialize(qoffset, real_coffset);

    tensor->set_binary_model_qinfo_start(qoffset);
    for (auto s : str) {
      strs.push_back(s);
    }
    op_size += tensor->bm_size();
    coffset = op_binary_model_base + op_size;
    qoffset = op_binary_model_base + op_size;
  }
  return strs;
}

void CSINNOP::to_file(std::ofstream& file) {
  for (auto tensor : inputs) {
    // Reuse 'dumped' flags to avoid duplicate dump
    if (tensor->dumped) tensor->to_file(file);
    tensor->dumped = false;
  }
  for (auto tensor : outputs) {
    tensor->to_file(file);
  }
  for (auto tensor : consts) {
    tensor->to_file(file);
  }
}

size_t CSINNBMGraph::dump_params(std::string path) {
  std::ofstream file;
  file.open(path, std::ios::out | std::ios::binary);
  for (auto op : ops) {
    file.seekp(op->get_op_binary_model_base());
    op->to_file(file);
  }
  file.close();
  return graph_size;
}

size_t CSINNBMGraph::dump_graph_info(std::string path) {
  FILE* f = fopen(path.c_str(), "wb");

  sess->input_num = inputs.size();
  sess->input = static_cast<struct csinn_tensor**>(calloc(1, sizeof(void*) * sess->input_num));
  for (uint i = 0; i < inputs.size(); i++) {
    sess->input[i] = inputs[i]->tensor;
  }
  sess->output_num = outputs.size();
  sess->output = static_cast<struct csinn_tensor**>(calloc(1, sizeof(void*) * sess->output_num));
  for (uint i = 0; i < outputs.size(); i++) {
    sess->output[i] = outputs[i]->tensor;
  }
  shl_dump_bm_graph_info_section(f, sess);
  fclose(f);
  return 0;
}

struct shl_binary_model {
  std::string section_paths[127][4];
  struct shl_binary_model_section_info* sinfo;
  size_t graph_offset;
  size_t params_offset;
  size_t info_offset;
  size_t debug_offset;
};

void add_section(struct shl_binary_model* bm, int sec_index, string type, string path) {
  struct shl_binary_model_section_info* sinfo = bm->sinfo;
  FILE* b = fopen(path.c_str(), "rb");

  fseek(b, 0, SEEK_END);
  size_t size = ftell(b);
  size_t offset_size = (size + 4095) / 4096;

  if (type == "graph") {
    sinfo->sections[sec_index].graph_offset = bm->graph_offset;
    sinfo->sections[sec_index].graph_size = size;
    bm->section_paths[sec_index][0] = path;
  } else if (type == "params") {
    sinfo->sections[sec_index].params_offset = bm->params_offset;
    sinfo->sections[sec_index].params_size = size;
    bm->section_paths[sec_index][1] = path;
  } else if (type == "info") {
    sinfo->sections[sec_index].info_offset = bm->info_offset;
    sinfo->sections[sec_index].info_size = size;
    bm->section_paths[sec_index][2] = path;
  } else if (type == "debug") {
    sinfo->sections[sec_index].debug_offset = bm->debug_offset;
    sinfo->sections[sec_index].debug_size = size;
    bm->section_paths[sec_index][3] = path;
  } else {
    LOG(FATAL) << "Unsupported section type: " << type;
  }

  bm->graph_offset += offset_size;
  bm->params_offset += offset_size;
  bm->info_offset += offset_size;
  bm->debug_offset += offset_size;

  if (sec_index >= sinfo->section_num) {
    sinfo->section_num = sec_index + 1;
  }
  fclose(b);
}

void emit_section(FILE* b, const char* path) {
  FILE* fp = fopen(path, "rb");
  fseek(fp, 0, SEEK_END);
  size_t file_size = ftell(fp);
  char* buffer = reinterpret_cast<char*>(malloc(file_size));
  fseek(fp, 0, SEEK_SET);
  size_t read_size = fread(buffer, 1, file_size, fp);
  if (read_size == 0) {
    LOG(FATAL) << "Error size of section file";
  }
  fwrite(buffer, 1, file_size, b);
  free(buffer);
  fclose(fp);
}

void emit_binary_model(String output_path, Array<Array<String>> sections) {
  struct shl_binary_model* bm =
      static_cast<struct shl_binary_model*>(calloc(1, sizeof(struct shl_binary_model)));
  bm->sinfo = static_cast<struct shl_binary_model_section_info*>(
      calloc(1, sizeof(struct shl_binary_model_section_info)));
  bm->sinfo->section_info_size = 4096;
  bm->graph_offset = 2;
  bm->params_offset = 2;
  bm->info_offset = 2;
  bm->debug_offset = 2;
  for (auto section : sections) {
    int section_idx = std::stoi(section[0]);
    std::string type = section[1];
    std::string path = section[2];
    add_section(bm, section_idx, type, path);
  }
  FILE* fp = fopen(output_path.c_str(), "wb");
  shl_dump_bm_header(fp);
  shl_dump_bm_section_info(fp, bm->sinfo);

  for (int i = 0; i < 127; i++) {
    if (bm->section_paths[i][0] != "") {
      fseek(fp, bm->sinfo->sections[i].graph_offset * 4096, 0);
      emit_section(fp, bm->section_paths[i][0].c_str());
    }
    if (bm->section_paths[i][1] != "") {
      fseek(fp, bm->sinfo->sections[i].params_offset * 4096, 0);
      emit_section(fp, bm->section_paths[i][1].c_str());
    }
    if (bm->section_paths[i][2] != "") {
      fseek(fp, bm->sinfo->sections[i].info_offset * 4096, 0);
      emit_section(fp, bm->section_paths[i][2].c_str());
    }
    if (bm->section_paths[i][3] != "") {
      fseek(fp, bm->sinfo->sections[i].debug_offset * 4096, 0);
      emit_section(fp, bm->section_paths[i][3].c_str());
    }
  }
  fclose(fp);
}

TVM_REGISTER_GLOBAL("relay.ext.csinn.emit_binary_model").set_body_typed(emit_binary_model);

}  // namespace csinn
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
