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
 * \file src/relay/backend/contrib/csinn/format.h
 * \brief The class for c code emit.
 */
#ifndef TVM_RELAY_BACKEND_CONTRIB_CSINN_FORMAT_H_
#define TVM_RELAY_BACKEND_CONTRIB_CSINN_FORMAT_H_

#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "../codegen_c/codegen_c.h"
#include "csi_nn.h"
#include "shl_utils.h"

using std::string;

namespace tvm {
namespace relay {
namespace contrib {
namespace csinn {
struct CSIStructedSparsity {
  int8_t* index;
  std::vector<int32_t> shape;
  size_t size{0};
  enum csinn_mem_type_enum type;
};

/* memory block for binary model */
class CSIMemoryBlock {
 public:
  void set_align(int32_t align) { align_ = align; }

  size_t set_start(size_t start) {
    original_start_ = start;
    if (original_start_ % align_) {
      /* base unaligned */
      aligned_start_ = (original_start_ + align_) / align_ * align_;
    } else {
      aligned_start_ = original_start_;
    }
    return aligned_start_;
  }

  size_t get_start() { return aligned_start_ + offset_; }

  void set_offset(size_t offset) { offset_ = offset; }

  void set_size(size_t size) { size_ = size; }

  size_t get_size() { return size_; }

  size_t get_space_size() {
    space_size_ = aligned_start_ - original_start_ + size_ + offset_;
    return space_size_;
  }

 private:
  /* beginning address */
  size_t original_start_;
  /* algined beginning address */
  size_t aligned_start_;
  /* space size in DDR */
  size_t space_size_;
  /* real data size */
  size_t size_;
  /* alignment */
  int32_t align_{1};
  /* real data start offset from memory block beginning */
  size_t offset_{0};
};

class CSIConstant {
 public:
  CSIConstant(string dtype, std::vector<int> shape) {
    set_shape(shape);
    set_dtype(dtype);
    alloc_data_buf(byte_size());
    mtype = CSINN_MEM_TYPE_CPU_NOT_ALIGNED;
  }

  /* data layout e.g. NCHW, NHWC */
  int32_t layout;
  /* memory type */
  enum csinn_mem_type_enum mtype;

  /* only for sparse */
  struct CSIStructedSparsity sparse;

  void set_name(string name) { name_ = name; }

  string get_name() { return name_; }

  void alloc_data_buf(size_t size) {
    data_buf_size_ = size;
    data_buf_ = malloc(data_buf_size_);
  }

  void realloc_data_buf(size_t size) {
    data_buf_size_ = size;
    data_buf_ = realloc(data_buf_, size);
  }

  void* get_data_buf() { return data_buf_; }

  void set_data_buf(void* buf) { data_buf_ = buf; }

  size_t get_offset() { return offset_; }
  void set_offset(size_t offset) { offset_ = offset; }

  void set_shape(std::vector<int> shape) {
    shape_ = shape;
    for (uint i = 0; i < shape_.size(); i++) {
      dim_align_.push_back(1);
      stride_.push_back(0);
    }
  }

  std::vector<int> get_shape() { return shape_; }

  void set_byte_size(size_t size) { data_buf_size_ = size; }

  void set_align(int32_t align) { align_ = align; }

  int32_t get_align() { return align_; }

  void set_dim_n_align(int index, int align) {
    CHECK(static_cast<uint>(index) < shape_.size());
    dim_align_[index] = align;
    int stride_size = 0;
    if (align != 1) {
      if (shape_[index] % align) {
        stride_size = align - (shape_[index] % align);
      } else {
        stride_size = 0;
      }
      stride_[index] = stride_size;
    }
    if (stride_size != 0) {
      dim_n_extend(index, stride_size);
    }
  }

  void set_dtype(string dtype) { dtype_ = dtype; }

  string get_dtype() { return dtype_; }

  size_t element_number() {
    size_t size = 1;
    for (uint i = 0; i < shape_.size(); i++) {
      size = size * (shape_[i] + stride_[i]);
    }
    return size;
  }

  size_t element_bit() {
    size_t size = 8;
    if (dtype_ == "int4_t") {
      size = 4;
    } else if (dtype_ == "uint8_t" || dtype_ == "int8_t") {
      size = 8;
    } else if (dtype_ == "int16_t" || dtype_ == "float16" || dtype_ == "bfloat16") {
      size = 16;
    } else if (dtype_ == "float" || dtype_ == "int32_t") {
      size = 32;
    } else if (dtype_ == "float64" || dtype_ == "int64_t") {
      size = 64;
    } else {
      LOG(ERROR) << "Error dtype in bit size";
    }
    return size;
  }

  size_t byte_size() {
    size_t element = element_number();
    size_t byte = element_bit() * element;
    if (dtype_ == "int4_t" && (byte % 8)) {
      byte = (byte + 8) / 8;
    } else {
      byte = byte / 8;
    }
    return byte;
  }

 private:
  void dim_n_extend(int dim, uint extend_size) {
    size_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
      outer_size = outer_size * (shape_[i] + stride_[i]);
    }
    size_t inner_size = 1;
    for (uint i = dim + 1; i < shape_.size(); i++) {
      inner_size = inner_size * (shape_[i] + stride_[i]);
    }

    char* aligned_data_buf = static_cast<char*>(calloc(byte_size(), 1));

    for (uint i = 0; i < outer_size; i++) {
      memcpy(aligned_data_buf + i * inner_size * (shape_[dim] + extend_size),
             static_cast<char*>(data_buf_) + i * inner_size * shape_[dim],
             inner_size * shape_[dim]);
    }
    free(data_buf_);
    data_buf_ = aligned_data_buf;
    data_buf_size_ = byte_size();
  }

  /* constant name */
  string name_;
  /* data value in DDR */
  void* data_buf_;
  /* data buffer byte size */
  size_t data_buf_size_;
  /* constant type e.g. int8_t */
  string dtype_;
  /* real data start offset from data_buf beginning */
  size_t offset_{0};
  /* data_buf's address align */
  int32_t align_{1};

  /* n dimension */
  std::vector<int> shape_;
  /* stride for every dimension */
  std::vector<int> stride_;
  /* align for every dimension */
  std::vector<int> dim_align_;

  /* CSI Constant has a memory block in binary model, is not a memory block in binary model */
  CSIMemoryBlock* memory_;
};

struct Qinfo {
  int32_t zero_point;
  float scale;
  int32_t multiplier;
  int32_t shift;
  float min;
  float max;
};

struct QinfoNode : public Qinfo, public Object {
 public:
  QinfoNode() = default;
  explicit QinfoNode(Qinfo* qinfo) {
    zero_point = qinfo->zero_point;
    scale = qinfo->scale;
    multiplier = qinfo->multiplier;
    shift = qinfo->shift;
    min = qinfo->min;
    max = qinfo->max;
  }

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("zero_point", &zero_point);
    auto scale_f = FloatImm(DataType::Float(32), scale);
    v->Visit("scale", &scale_f);
    v->Visit("multiplier", &multiplier);
    v->Visit("shift", &shift);
    auto min_f = FloatImm(DataType::Float(32), min);
    v->Visit("min", &min_f);
    auto max_f = FloatImm(DataType::Float(32), max);
    v->Visit("max", &max_f);
  }

  static constexpr const char* _type_key = "relay.ext.csinn.Qinfo";
  TVM_DECLARE_FINAL_OBJECT_INFO(QinfoNode, Object);

  friend class QuantParams;
  friend class QuantParamsRef;
};

class QinfoRef : public ObjectRef {
 public:
  QinfoRef() {
    auto n = make_object<QinfoNode>();
    data_ = std::move(n);
  }

  /*!
   * \brief Construct from an object pointer.
   * \param n The object pointer.
   */
  explicit QinfoRef(ObjectPtr<Object> n) : ObjectRef(n) {}

  /*! \return Mutable pointers to the node. */
  QinfoNode* operator->() const {
    auto* ptr = get_mutable();
    ICHECK(ptr != nullptr);
    return static_cast<QinfoNode*>(ptr);
  }
};

struct QuantParams {
  struct Qinfo* qinfo;
  int32_t value_type;
  string name;
  std::vector<int> shape;
  int32_t q_size;
  int32_t offset;
  string dtype;
};

struct QuantParamsNode : public QuantParams, public Object {
 public:
  QuantParamsNode() = default;

  explicit QuantParamsNode(QuantParams* params) {
    value_type = params->value_type;
    name = params->name;
    shape = std::vector<int>(params->shape.begin(), params->shape.end());
    q_size = params->q_size;
    offset = params->offset;
    dtype = params->dtype;
    qinfo = params->qinfo;
  }

  QuantParams* GetQuantParams() {
    QuantParams* res = new QuantParams;
    res->dtype = dtype;
    res->name = name;
    res->offset = offset;
    res->q_size = q_size;
    res->qinfo = qinfo;
    res->shape = shape;
    res->value_type = value_type;

    return res;
  }

  void VisitAttrs(AttrVisitor* v) {
    Array<QinfoRef> qr_array;
    for (int i = 0; i < q_size; i++) {
      // auto n = make_object<Qinfo>(qinfo[i]);
      // qr_array.push_back(QinfoRef(n));

      QinfoNode qn(&qinfo[i]);
      auto n = make_object<QinfoNode>(qn);
      qr_array.push_back(QinfoRef(n));
    }
    v->Visit("qinfo", &qr_array);

    v->Visit("value_type", &value_type);
    v->Visit("name", &name);

    Array<Integer> arr_shape;
    for (auto s : shape) {
      arr_shape.push_back(Integer(s));
    }
    v->Visit("shape", &arr_shape);

    v->Visit("q_size", &q_size);
    v->Visit("offset", &offset);
    v->Visit("dtype", &dtype);
  }

  static constexpr const char* _type_key = "relay.ext.csinn.QuantParams";
  TVM_DECLARE_FINAL_OBJECT_INFO(QuantParamsNode, Object);

  friend class QuantParamsRef;
};

class QuantParamsRef : public ObjectRef {
 public:
  QuantParamsRef() {
    auto n = make_object<QuantParamsNode>();
    data_ = std::move(n);
  }

  /*!
   * \brief Construct from an object pointer.
   * \param n The object pointer.
   */
  explicit QuantParamsRef(ObjectPtr<Object> n) : ObjectRef(n) {}

  /*! \return Mutable pointers to the node. */
  QuantParamsNode* operator->() const {
    auto* ptr = get_mutable();
    ICHECK(ptr != nullptr);
    return static_cast<QuantParamsNode*>(ptr);
  }
};

class CSINNTensor {
 public:
  CSINNTensor() { tensor = csinn_alloc_tensor(NULL); }

  void append_str(std::ostringstream& decl) {
    decl << ";";
    astr.push_back(decl.str());
    decl.str("");
  }

  string to_mtype(enum csinn_mem_type_enum mtype);
  string to_dtype(enum csinn_dtype_enum dtype);
  string to_layout(int32_t layout);
  string to_layout() { return to_layout(tensor->layout); }
  std::vector<string> serialize(size_t qoffset, size_t coffset);
  virtual void to_file(std::ofstream& file) = 0;
  virtual size_t bm_size() = 0;

  void set_const(CSIConstant* cst) {
    const_data = cst->get_data_buf();
    tensor->mtype = cst->mtype;
    if (cst->sparse.size) {
      tensor->mtype = cst->sparse.type;
      sparse_index_data = reinterpret_cast<char*>(cst->sparse.index);
      binary_model_sparse_index_size = cst->sparse.size;
    }
    if (cst->layout == CSINN_LAYOUT_O32HWI32 || cst->layout == CSINN_LAYOUT_O32I32) {
      tensor->layout = cst->layout;
    }
    const_mem_ = new CSIMemoryBlock();
    const_mem_->set_align(cst->get_align());
    const_mem_->set_offset(cst->get_offset());
    const_mem_->set_size(cst->byte_size());
  }

  void set_quant(const struct QuantParams& quant) {
    quant_data = quant.qinfo;

    quant_mem_ = new CSIMemoryBlock();
    quant_mem_->set_size(quant.q_size * sizeof(Qinfo));
  }

  void set_const_data_size(size_t size) { const_mem_->set_size(size); }

  void set_binary_model_qinfo_start(size_t start) { quant_mem_->set_start(start); }

  size_t get_binary_model_qinfo_real_start() { return quant_mem_->get_start(); }

  size_t get_binary_model_qinfo_space_size() { return quant_mem_->get_space_size(); }

  void set_binary_model_const_start(size_t start) { const_mem_->set_start(start); }

  size_t get_binary_model_const_real_start() { return const_mem_->get_start(); }

  size_t get_binary_model_const_size() { return const_mem_->get_size(); }

  struct csinn_tensor* tensor;
  string name;
  void* const_data;
  /* real data start offset from const_data beginning */
  uint32_t const_data_offset;
  /* const_data's address align */
  int32_t const_data_align;
  char* sparse_index_data;
  struct Qinfo* quant_data;
  /* quant memory in binary model */
  CSIMemoryBlock* quant_mem_;
  /* const memory in binary model */
  CSIMemoryBlock* const_mem_;

  size_t binary_model_sparse_index_size{0};
  string org_name = "";

 private:
  void push_str(std::ostringstream& decl) {
    decl << ";";
    str.push_back(decl.str());
    decl.str("");
  }
  std::vector<string> str;
  std::vector<string> astr;
};

class CSINNConstantTensor : public CSINNTensor {
 public:
  size_t bm_size() {
    return quant_mem_->get_space_size() + const_mem_->get_space_size() +
           binary_model_sparse_index_size;
  }
  void to_file(std::ofstream& file);
};

class CSINNVarTensor : public CSINNTensor {
 public:
  size_t bm_size() { return quant_mem_->get_space_size(); }
  void to_file(std::ofstream& file);
  bool dumped = false;
};

class CSINNOP {
 public:
  CSINNTensor* get_tensor(string name);

  size_t size() { return op_size; }
  void set_bm_base(size_t base) {
    if (base % op_binary_model_base_align) {
      /* base unaligned */
      op_binary_model_base = (base + op_binary_model_base_align) / op_binary_model_base_align *
                             op_binary_model_base_align;
    } else {
      op_binary_model_base = base;
    }
  }

  void push_input(CSINNVarTensor* in);
  void push_output(CSINNVarTensor* out);
  void push_constant(CSINNConstantTensor* cst);
  CSINNConstantTensor* get_constant(uint32_t idx) { return consts[idx]; }

  std::vector<string> serialize();
  void to_file(std::ofstream& file);
  void set_name(string name) { layer_name = name; }
  string get_name() { return layer_name; }
  size_t get_op_binary_model_base() { return op_binary_model_base; }

 private:
  string layer_name;
  size_t op_size{0};
  /* base offset in binary model */
  size_t op_binary_model_base{0};
  /* base offset align in binary model, default is 4 */
  size_t op_binary_model_base_align{4};
  std::vector<CSINNVarTensor*> inputs;
  std::vector<CSINNVarTensor*> outputs;
  std::vector<CSINNConstantTensor*> consts;
  std::vector<string> strs;
};

class CSINNBMGraph {
 public:
  CSINNBMGraph() {
    sess = static_cast<struct csinn_session*>(calloc(1, sizeof(struct csinn_session)));
  }
  size_t push_op(CSINNOP* op) {
    ops.push_back(op);
    graph_size = op->get_op_binary_model_base() + op->size();
    return graph_size;
  }
  void set_layer_align(size_t align) { layer_align = align; }
  void set_input(CSINNTensor* tensor) { return inputs.push_back(tensor); }
  std::vector<CSINNTensor*> get_inputs() { return inputs; }
  void set_output(CSINNTensor* tensor) { return outputs.push_back(tensor); }
  std::vector<CSINNTensor*> get_outputs() { return outputs; }
  size_t dump_params(std::string path);
  size_t size() { return graph_size; }
  size_t dump_graph_info(std::string path);

  CSINNTensor* get_tensor(string name) {
    for (auto op : ops) {
      auto out = op->get_tensor(name);
      if (out) {
        return out;
      }
    }
    return NULL;
  }

  struct csinn_session* sess;

 private:
  std::vector<CSINNTensor*> inputs;
  std::vector<CSINNTensor*> outputs;
  std::vector<CSINNOP*> ops;
  size_t graph_base_size{0};
  size_t graph_size{0};
  size_t layer_align{32};
};

class CSINNCodeFormat {
 public:
  void EnterScope() { indent_ += 2; }

  void ExitScope() {
    CHECK_GE(indent_, 2U) << "Wrong ident found.";
    indent_ -= 2;
  }

  void Indents() {
    for (int i = 0; i < indent_; i++) {
      code_stream_ << ' ';
    }
  }

  void OneLine(string str) {
    Indents();
    code_stream_ << str << "\n";
  }

  void OneLine(std::ostringstream& str) {
    OneLine(str.str());
    str.str("");
  }

  void NewLine() { code_stream_ << "\n"; }

  void PushDecl(const std::vector<string>& decls) {
    for (string decl : decls) {
      buf_decl_.push_back(decl);
    }
  }

  void PushDecl(std::ostringstream& decl) {
    decl << ";";
    buf_decl_.push_back(decl.str());
    decl.str("");
  }

  void PushCall(std::ostringstream& call) {
    call << ";";
    buf_call_.push_back(call.str());
    call.str("");
  }

  void BufToCode() {
    for (auto decl : buf_decl_) {
      OneLine(decl);
    }
    NewLine();
    for (auto stmt : buf_call_) {
      OneLine(stmt);
    }
  }

  string str() { return code_stream_.str(); }

 private:
  std::vector<string> buf_decl_;
  std::vector<string> buf_call_;
  std::ostringstream code_stream_;
  int indent_{0};
};

}  // namespace csinn
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_BACKEND_CONTRIB_CSINN_FORMAT_H_
