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
 * \file src/relay/backend/contrib/csinn/backend.h
 * \brief The base class for backend.
 */
#ifndef TVM_RELAY_BACKEND_CONTRIB_CSINN_BACKEND_H_
#define TVM_RELAY_BACKEND_CONTRIB_CSINN_BACKEND_H_

#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/function.h>

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../../utils.h"

namespace tvm {
namespace relay {
namespace contrib {
namespace csinn {

using RelayFunction = tvm::relay::Function;
using RelayExpr = tvm::relay::Expr;
using RelayExprNode = tvm::relay::ExprNode;
using RelayExprVisitor = tvm::relay::ExprVisitor;
using RelayType = tvm::relay::Type;
using RelayCall = tvm::relay::Call;
using RelayCallNode = tvm::relay::CallNode;
using RelayVar = tvm::relay::Var;
using RelayVarNode = tvm::relay::VarNode;
using RelayConstant = tvm::relay::Constant;
using RelayConstantNode = tvm::relay::ConstantNode;
using RelayTuple = tvm::relay::Tuple;
using RelayTupleNode = tvm::relay::TupleNode;
using RelayTupleGetItem = tvm::relay::TupleGetItem;
using RelayTupleGetItemNode = tvm::relay::TupleGetItemNode;

struct HHBExprExtend {
  std::vector<int> shape;
  DataType dtype;
  void* target_data;
  std::string name;
  std::vector<std::string> output_names;
};

class ExprNode : public RelayExprNode {
 public:
  ExprNode() { hhb_expr_extend_ = new HHBExprExtend(); }
  std::vector<int> get_shape() const { return hhb_expr_extend_->shape; }
  ExprNode* get_next_expr(int index) const {
    if (index >= static_cast<int>(next_expr_.size())) {
      return NULL;
    }
    return next_expr_[index];
  }
  struct HHBExprExtend* hhb_expr_extend_;
  std::vector<ExprNode*> next_expr_;
  static constexpr const char* _type_key = "csinn.expr";
  static constexpr const uint32_t _type_child_slots = 0;
  TVM_DECLARE_BASE_OBJECT_INFO(ExprNode, RelayExprNode);
};

class Expr : public RelayExpr {
 public:
  void set_target_data(void* tdata) {
    auto data = static_cast<ExprNode*>(data_.get());
    data->hhb_expr_extend_->target_data = tdata;
  }
  void set_expr_name(std::string str) {
    auto data = static_cast<ExprNode*>(data_.get());
    data->hhb_expr_extend_->name = str;
  }
  std::string get_expr_name() {
    auto data = static_cast<ExprNode*>(data_.get());
    return data->hhb_expr_extend_->name;
  }
  void set_output_names(std::vector<std::string> name) {
    auto data = static_cast<ExprNode*>(data_.get());
    data->hhb_expr_extend_->output_names = name;
  }
  std::vector<std::string> get_output_names() {
    auto data = static_cast<ExprNode*>(data_.get());
    return data->hhb_expr_extend_->output_names;
  }
  void push_next_expr(ExprNode* e) {
    auto data = static_cast<ExprNode*>(data_.get());
    data->next_expr_.push_back(e);
  }

  void push_next_expr(ExprNode* e, uint index) {
    auto data = static_cast<ExprNode*>(data_.get());

    if (index >= data->next_expr_.size()) {
      std::vector<ExprNode*> new_next_expr_(index + 1, NULL);

      for (uint i = 0; i < data->next_expr_.size(); i++) {
        new_next_expr_[i] = data->next_expr_[i];
      }
      data->next_expr_ = new_next_expr_;
    }

    data->next_expr_[index] = e;
  }
  ExprNode* operator->() { return static_cast<ExprNode*>(data_.get()); }
  ExprNode* get() { return operator->(); }
  TVM_DEFINE_OBJECT_REF_METHODS(Expr, RelayExpr, ExprNode);
};

struct HHBCallExtend {
  struct QConfig_* quant_config;
  /* order: input, kernel, bias, output */
  std::vector<struct QuantParams*> op_quant;
};

class CallNode : public ExprNode {
 public:
  RelayExpr op;
  tvm::Array<Expr> args;
  Attrs attrs;
  tvm::Array<Type> type_args;

  struct HHBCallExtend* hhb_call_extend_;

  struct QConfig_* get_quant_config() const {
    return hhb_call_extend_->quant_config;
  }

  struct QuantParams* get_op_quant(int index) const {
    return hhb_call_extend_->op_quant[index];
  }

  int get_op_quant_size() const { return hhb_call_extend_->op_quant.size(); }

  static constexpr const char* _type_key = "csinn.call";
  TVM_DECLARE_FINAL_OBJECT_INFO(CallNode, ExprNode);
};

class Call : public Expr {
 public:
  Call(RelayExpr op, Array<Expr> args, struct HHBExprExtend* extend, Attrs attrs = Attrs(),
       Array<RelayType> type_args = Array<RelayType>(), Span span = Span());

  void set_quant_config(struct QConfig_* cfg) {
    auto data = static_cast<CallNode*>(data_.get());
    data->hhb_call_extend_->quant_config = cfg;
  }

  void push_op_quant(struct QuantParams* quant) {
    auto data = static_cast<CallNode*>(data_.get());
    data->hhb_call_extend_->op_quant.push_back(quant);
  }

  TVM_DEFINE_OBJECT_REF_METHODS(Call, Expr, CallNode);
};

class VarNode : public ExprNode {
 public:
  Id vid;
  Type type_annotation;
  const String& name_hint() const { return vid->name_hint; }
  static constexpr const char* _type_key = "csinn.var";
  TVM_DECLARE_FINAL_OBJECT_INFO(VarNode, ExprNode);
};

class Var : public Expr {
 public:
  Var(String name_hint, struct HHBExprExtend* extend, Type type_annotation, Span span = Span())
      : Var(Id(name_hint), extend, type_annotation, span) {}
  Var(Id vid, struct HHBExprExtend* extend, Type type_annotation, Span span = Span());
  TVM_DEFINE_OBJECT_REF_METHODS(Var, Expr, VarNode);
};

class ConstantNode : public ExprNode {
 public:
  runtime::NDArray data;
  static constexpr const char* _type_key = "csinn.constant";
  TVM_DECLARE_FINAL_OBJECT_INFO(ConstantNode, ExprNode);
};

class Constant : public Expr {
 public:
  Constant(runtime::NDArray data, struct HHBExprExtend* extend, Span span = Span());
  TVM_DEFINE_OBJECT_REF_METHODS(Constant, Expr, ConstantNode);
};

class TupleNode : public ExprNode {
 public:
  tvm::Array<Expr> fields;

  static constexpr const char* _type_key = "csinn.tuple";
  TVM_DECLARE_FINAL_OBJECT_INFO(TupleNode, ExprNode);
};

class Tuple : public Expr {
 public:
  explicit Tuple(tvm::Array<Expr> fields, Span span = Span());
  TupleNode* operator->() { return static_cast<TupleNode*>(data_.get()); }
  TupleNode* get() { return operator->(); }
  TVM_DEFINE_OBJECT_REF_METHODS(Tuple, Expr, TupleNode);
};

class TupleGetItemNode : public ExprNode {
 public:
  Expr tuple;
  int index;

  static constexpr const char* _type_key = "csinn.tuple_get_item";
  TVM_DECLARE_FINAL_OBJECT_INFO(TupleGetItemNode, ExprNode);
};

class TupleGetItem : public Expr {
 public:
  TupleGetItem(Expr tuple, int index, Span span = Span());
  TupleGetItemNode* operator->() { return static_cast<TupleGetItemNode*>(data_.get()); }
  TupleGetItemNode* get() { return operator->(); }
  TVM_DEFINE_OBJECT_REF_METHODS(TupleGetItem, Expr, TupleGetItemNode);
};

template <typename FType>
class CSINNExprFunctor;

template <typename R>
class CSINNExprFunctor<R(const Expr& n)> {
 private:
  using TSelf = CSINNExprFunctor<R(const Expr& n)>;
  using FType = tvm::NodeFunctor<R(const ObjectRef& n, TSelf* self)>;

 public:
  /*! \brief the result type of this functor */
  using result_type = R;
  /*! \brief virtual destructor */
  virtual ~CSINNExprFunctor() {}

  virtual R visit(const Expr& n) {
    ICHECK(n.defined()) << "Found null pointer node while traversing AST. The previous pass may "
                           "have generated invalid data.";
    static FType vtable = InitVTable();
    return vtable(n, this);
  }
  // Functions that can be overriden by subclass
  virtual R visit_expr(const ConstantNode* op) { return visit_expr_default(op); }
  virtual R visit_expr(const TupleNode* op) { return visit_expr_default(op); }
  virtual R visit_expr(const VarNode* op) { return visit_expr_default(op); }
  virtual R visit_expr(const GlobalVarNode* op) { return visit_expr_default(op); }
  virtual R visit_expr(const FunctionNode* op) { return visit_expr_default(op); }
  virtual R visit_expr(const CallNode* op) { return visit_expr_default(op); }
  virtual R visit_expr(const LetNode* op) { return visit_expr_default(op); }
  virtual R visit_expr(const IfNode* op) { return visit_expr_default(op); }
  virtual R visit_expr(const OpNode* op) { return visit_expr_default(op); }
  virtual R visit_expr(const TupleGetItemNode* op) { return visit_expr_default(op); }
  virtual R visit_expr(const RefCreateNode* op) { return visit_expr_default(op); }
  virtual R visit_expr(const RefReadNode* op) { return visit_expr_default(op); }
  virtual R visit_expr(const RefWriteNode* op) { return visit_expr_default(op); }
  virtual R visit_expr(const ConstructorNode* op) { return visit_expr_default(op); }
  virtual R visit_expr(const MatchNode* op) { return visit_expr_default(op); }
  virtual R visit_expr_default(const Object* op) {
    LOG(FATAL) << "Do not have a default for " << op->GetTypeKey();
    throw;
  }

  /*!
   * \brief Returns dtype string
   *
   * \param var Var to get the dtype of
   *
   * \return The dtype string.
   */
  std::string GetDtypeString(const Var& var) {
    auto ttype = var->checked_type().as<TensorTypeNode>();
    ICHECK(ttype) << "Expect TensorTypeNode";
    return GetDtypeString(ttype);
  }

  /*!
   * \brief Returns dtype string
   *
   * \param ttype TensorTypeNode* to get the dtype of
   *
   * \return The dtype string.
   */
  std::string GetDtypeString(const TensorTypeNode* ttype) { return GetDtypeString(ttype->dtype); }

  std::string GetDtypeString(DataType i_dtype) {
    std::string dtype;
    if (runtime::TypeMatch(i_dtype, kDLFloat, 32)) {
      dtype = "float";
    } else if (runtime::TypeMatch(i_dtype, kDLFloat, 16)) {
      dtype = "half";
    } else if (runtime::TypeMatch(i_dtype, kDLInt, 32)) {
      dtype = "int32_t";
    } else if (runtime::TypeMatch(i_dtype, kDLInt, 64)) {
      dtype = "int64_t";
    } else if (runtime::TypeMatch(i_dtype, kDLInt, 8)) {
      dtype = "int8_t";
    } else if (runtime::TypeMatch(i_dtype, kDLUInt, 8)) {
      dtype = "uint8_t";
    } else if (runtime::TypeMatch(i_dtype, kDLUInt, 1)) {
      dtype = "bool";
    } else {
      LOG(FATAL) << "Unsupported dtype " << i_dtype;
    }

    return dtype;
  }

 private:
  // initialize the vtable.
  static FType InitVTable() {
    FType vtable;

    vtable.template set_dispatch<ConstantNode>([](const ObjectRef& n, TSelf* self) {
      return self->visit_expr(static_cast<const ConstantNode*>(n.get()));
    });
    vtable.template set_dispatch<TupleNode>([](const ObjectRef& n, TSelf* self) {
      return self->visit_expr(static_cast<const TupleNode*>(n.get()));
    });
    vtable.template set_dispatch<VarNode>([](const ObjectRef& n, TSelf* self) {
      return self->visit_expr(static_cast<const VarNode*>(n.get()));
    });
    vtable.template set_dispatch<GlobalVarNode>([](const ObjectRef& n, TSelf* self) {
      return self->visit_expr(static_cast<const GlobalVarNode*>(n.get()));
    });
    vtable.template set_dispatch<FunctionNode>([](const ObjectRef& n, TSelf* self) {
      return self->visit_expr(static_cast<const FunctionNode*>(n.get()));
    });
    vtable.template set_dispatch<CallNode>([](const ObjectRef& n, TSelf* self) {
      return self->visit_expr(static_cast<const CallNode*>(n.get()));
    });
    vtable.template set_dispatch<LetNode>([](const ObjectRef& n, TSelf* self) {
      return self->visit_expr(static_cast<const LetNode*>(n.get()));
    });
    vtable.template set_dispatch<IfNode>([](const ObjectRef& n, TSelf* self) {
      return self->visit_expr(static_cast<const IfNode*>(n.get()));
    });
    vtable.template set_dispatch<OpNode>([](const ObjectRef& n, TSelf* self) {
      return self->visit_expr(static_cast<const OpNode*>(n.get()));
    });
    vtable.template set_dispatch<TupleGetItemNode>([](const ObjectRef& n, TSelf* self) {
      return self->visit_expr(static_cast<const TupleGetItemNode*>(n.get()));
    });
    vtable.template set_dispatch<RefCreateNode>([](const ObjectRef& n, TSelf* self) {
      return self->visit_expr(static_cast<const RefCreateNode*>(n.get()));
    });
    vtable.template set_dispatch<RefReadNode>([](const ObjectRef& n, TSelf* self) {
      return self->visit_expr(static_cast<const RefReadNode*>(n.get()));
    });
    vtable.template set_dispatch<RefWriteNode>([](const ObjectRef& n, TSelf* self) {
      return self->visit_expr(static_cast<const RefWriteNode*>(n.get()));
    });
    vtable.template set_dispatch<ConstructorNode>([](const ObjectRef& n, TSelf* self) {
      return self->visit_expr(static_cast<const ConstructorNode*>(n.get()));
    });
    vtable.template set_dispatch<MatchNode>([](const ObjectRef& n, TSelf* self) {
      return self->visit_expr(static_cast<const MatchNode*>(n.get()));
    });
    return vtable;
  }
};

class FromRelay : public ExprFunctor<Expr(const RelayExpr&)> {
 public:
  Expr expr(const RelayExpr& expr);

  virtual Expr VisitRelayExpr(const RelayExpr& expr);
  virtual Expr VisitExpr_(const RelayCallNode* call);
  virtual Expr VisitExpr_(const RelayVarNode* op);
  virtual Expr VisitExpr_(const RelayConstantNode* op);
  // virtual Expr VisitExpr_(const RelayGlobalVarNode* op);
  // virtual Expr VisitExpr_(const RelayOpNode* op);
  virtual Expr VisitExpr_(const RelayTupleNode* op);
  // virtual Expr VisitExpr_(const RelayFunctionNode* op);
  // virtual Expr VisitExpr_(const RelayLetNode* op);
  // virtual Expr VisitExpr_(const RelayIfNode* op);
  virtual Expr VisitExpr_(const RelayTupleGetItemNode* op);
  // virtual Expr VisitExpr_(const RelayRefCreateNode* op);
  // virtual Expr VisitExpr_(const RelayRefReadNode* op);
  // virtual Expr VisitExpr_(const RelayRefWriteNode* op);
  // virtual Expr VisitExpr_(const RelayConstructorNode* op);
  // virtual Expr VisitExpr_(const RelayMatchNode* op);

 protected:
  /*! \brief Internal map used for memoization. */
  std::unordered_map<RelayExpr, Expr, ObjectPtrHash, ObjectPtrEqual> memo_;
};

class ToRelay : public CSINNExprFunctor<RelayExpr(const Expr&)> {
 public:
  RelayExpr relay(const Expr& expr);

  virtual RelayExpr visit(const Expr& expr);
  virtual RelayExpr visit_expr(const ConstantNode* call);
  virtual RelayExpr visit_expr(const CallNode* call);
  virtual RelayExpr visit_expr(const VarNode* call);
  virtual RelayExpr visit_expr(const TupleNode* call);
  virtual RelayExpr visit_expr(const TupleGetItemNode* call);

 protected:
  /*! \brief Internal map used for memoization. */
  std::unordered_map<Expr, RelayExpr, ObjectPtrHash, ObjectPtrEqual> memo_;
};

class HHBExprVisitor : public CSINNExprFunctor<void(const Expr& n)> {
 public:
  void visit(const Expr& expr) override;
  void visit_expr(const VarNode* op) override;
  // void visit_expr(const GlobalVarNode* op) override;
  void visit_expr(const ConstantNode* op) override;
  void visit_expr(const TupleNode* op) override;
  // void visit_expr(const FunctionNode* op) override;
  void visit_expr(const CallNode* op) override;
  // void visit_expr(const LetNode* op) override;
  // void visit_expr(const IfNode* op) override;
  // void visit_expr(const OpNode* op) override;
  void visit_expr(const TupleGetItemNode* op) override;
  // void visit_expr(const RefCreateNode* op) override;
  // void visit_expr(const RefReadNode* op) override;
  // void visit_expr(const RefWriteNode* op) override;
  // void visit_expr(const ConstructorNode* op) override;
  // void visit_expr(const MatchNode* op) override;
  // virtual void VisitType(const Type& t);
  // virtual void VisitClause(const Clause& c);
  // virtual void VisitPattern(const Pattern& c);
  // virtual void VisitSpan(const Span& span);

 protected:
  // Internal visiting counter
  std::unordered_map<const Object*, size_t> visit_counter_;
};

class HHBExprMutator : public CSINNExprFunctor<Expr(const Expr&)> {
 public:
  virtual Expr visit_expr(const ConstantNode* call);
  virtual Expr visit_expr(const CallNode* call);
  virtual Expr visit_expr(const VarNode* call);
  virtual Expr visit_expr(const TupleNode* call);
  virtual Expr visit_expr(const TupleGetItemNode* call);
  virtual Expr visit(const Expr& expr);

 protected:
  std::unordered_map<Expr, Expr, ObjectPtrHash, ObjectPtrEqual> memo_;
};

class Optimize {
 public:
  virtual void optimization();
  virtual void phase0();
  virtual void phase1();
  virtual void phase2();
  virtual void phase3();
  virtual void target_define_phase0() {}
  virtual void target_define_phase1() {}
  virtual void target_define_phase2() {}
  virtual void target_define_phase3() {}
};

class Pass {
 public:
  virtual void to_target_graph() {}
  virtual void from_target_graph() {}
  virtual void import_realy_expr(const RelayExpr& func);
  virtual RelayExpr export_realy_expr();

 protected:
  Expr expr;

 private:
  class FromRelay get_expr;
  class ToRelay export_relay;
};

/*!
 * \brief Check if a call has the provided name.
 * \param call A Relay call node.
 * \param op_name The name of the expected call.
 * \return true if the call's name is equivalent to the given name. Otherwise,
 * false.
 */
inline bool IsOp(const CallNode* call, const std::string& op_name) {
  const auto* op_node = call->op.as<OpNode>();
  ICHECK(op_node) << "Expects a single op.";
  Op op = GetRef<Op>(op_node);
  return op == Op::Get(op_name);
}

}  // namespace csinn
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_BACKEND_CONTRIB_CSINN_BACKEND_H_
