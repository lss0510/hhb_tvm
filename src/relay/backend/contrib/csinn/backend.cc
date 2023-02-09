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
 * \file src/relay/backend/contrib/csinn/backend.cc
 * \brief Implementation of CSINN backend codegen APIs.
 */

#include "backend.h"

namespace tvm {
namespace relay {
namespace contrib {
namespace csinn {

Call::Call(RelayExpr op, Array<Expr> args, struct HHBExprExtend* extend, Attrs attrs,
           Array<RelayType> type_args, Span span) {
  ObjectPtr<CallNode> n = make_object<CallNode>();
  n->op = std::move(op);
  n->args = std::move(args);
  n->attrs = std::move(attrs);
  n->type_args = std::move(type_args);
  n->span = std::move(span);
  n->hhb_expr_extend_ = extend;
  n->hhb_call_extend_ = new HHBCallExtend();
  data_ = std::move(n);
}

Var::Var(Id vid, struct HHBExprExtend* extend, Type type_annotation, Span span) {
  ObjectPtr<VarNode> n = make_object<VarNode>();
  n->vid = std::move(vid);
  n->type_annotation = std::move(type_annotation);
  n->span = std::move(span);
  n->hhb_expr_extend_ = extend;
  data_ = std::move(n);
}

Constant::Constant(runtime::NDArray data, struct HHBExprExtend* extend, Span span) {
  ObjectPtr<ConstantNode> n = make_object<ConstantNode>();
  n->data = std::move(data);
  n->span = std::move(span);
  n->hhb_expr_extend_ = extend;
  data_ = std::move(n);
}

Tuple::Tuple(tvm::Array<Expr> fields, Span span) {
  ObjectPtr<TupleNode> n = make_object<TupleNode>();
  n->fields = std::move(fields);
  n->span = std::move(span);
  data_ = std::move(n);
}

TupleGetItem::TupleGetItem(Expr tuple, int index, Span span) {
  ObjectPtr<TupleGetItemNode> n = make_object<TupleGetItemNode>();
  n->tuple = std::move(tuple);
  n->index = index;
  n->span = std::move(span);
  data_ = std::move(n);
}

Expr FromRelay::VisitExpr_(const RelayCallNode* op) {
  tvm::Array<Expr> call_args;
  for (auto arg : op->args) {
    auto new_arg = VisitRelayExpr(arg);
    call_args.push_back(new_arg);
  }
  auto type = op->checked_type();
  std::vector<int> shape;
  DataType dtype;
  if (type.as<TensorTypeNode>()) {
    shape = tvm::relay::backend::GetShape(type);
    dtype = tvm::relay::backend::GetType(type);
  }
  struct HHBExprExtend* extend = new HHBExprExtend();
  extend->shape = shape;
  extend->dtype = dtype;
  auto ret = Call(op->op, call_args, extend, op->attrs, op->type_args, op->span);
  ret->checked_type_ = op->checked_type_;
  return ret;
}

Expr FromRelay::VisitExpr_(const RelayVarNode* op) {
  std::vector<int> shape;
  DataType dtype;
  auto type = op->checked_type();
  if (type.as<TensorTypeNode>()) {
    shape = tvm::relay::backend::GetShape(type);
    dtype = tvm::relay::backend::GetType(type);
  }
  struct HHBExprExtend* extend = new HHBExprExtend();
  extend->shape = shape;
  extend->dtype = dtype;
  auto ret = Var(op->vid, extend, op->type_annotation, op->span);
  ret->checked_type_ = op->checked_type_;
  return ret;
}

Expr FromRelay::VisitExpr_(const RelayConstantNode* op) {
  auto shape = tvm::relay::backend::GetShape(op->checked_type());
  auto dtype = tvm::relay::backend::GetType(op->checked_type());
  struct HHBExprExtend* extend = new HHBExprExtend();
  extend->shape = shape;
  extend->dtype = dtype;
  auto ret = Constant(op->data, extend, op->span);
  ret->checked_type_ = op->checked_type_;
  return ret;
}

Expr FromRelay::VisitExpr_(const RelayTupleNode* op) {
  tvm::Array<Expr> fields;
  for (auto field : op->fields) {
    auto new_field = VisitRelayExpr(field);
    fields.push_back(new_field);
  }
  auto ret = Tuple(fields, op->span);
  ret->checked_type_ = op->checked_type_;
  return ret;
}

Expr FromRelay::VisitExpr_(const RelayTupleGetItemNode* get_item) {
  auto new_tuple = VisitRelayExpr(get_item->tuple);
  auto shape = tvm::relay::backend::GetShape(get_item->checked_type());
  auto dtype = tvm::relay::backend::GetType(get_item->checked_type());
  struct HHBExprExtend* extend = new HHBExprExtend();
  extend->shape = shape;
  extend->dtype = dtype;
  auto ret = TupleGetItem(new_tuple, get_item->index, get_item->span);
  ret->hhb_expr_extend_ = extend;
  ret->checked_type_ = get_item->checked_type_;
  return ret;
}

Expr FromRelay::VisitRelayExpr(const RelayExpr& expr) {
  auto it = this->memo_.find(expr);
  if (it != this->memo_.end()) {
    return it->second;
  } else {
    Expr new_expr = ExprFunctor::VisitExpr(expr);
    memo_[expr] = new_expr;
    return new_expr;
  }
}

Expr FromRelay::expr(const RelayExpr& expr) {
  Expr ret = ExprFunctor::VisitExpr(expr);
  return ret;
}

RelayExpr ToRelay::visit_expr(const CallNode* op) {
  tvm::Array<RelayExpr> call_args;
  for (auto arg : op->args) {
    auto new_arg = visit(arg);
    call_args.push_back(new_arg);
  }
  auto ret = RelayCall(op->op, call_args, op->attrs, op->type_args, op->span);
  ret->checked_type_ = op->checked_type_;
  return ret;
}

RelayExpr ToRelay::visit_expr(const VarNode* op) {
  auto ret = RelayVar(op->vid, op->type_annotation, op->span);
  ret->checked_type_ = op->checked_type_;
  return ret;
}

RelayExpr ToRelay::visit_expr(const ConstantNode* op) {
  auto ret = RelayConstant(op->data, op->span);
  ret->checked_type_ = op->checked_type_;
  return ret;
}

RelayExpr ToRelay::visit_expr(const TupleNode* op) {
  tvm::Array<RelayExpr> fields;
  for (auto field : op->fields) {
    auto new_field = visit(field);
    fields.push_back(new_field);
  }
  auto ret = RelayTuple(fields, op->span);
  ret->checked_type_ = op->checked_type_;
  return ret;
}

RelayExpr ToRelay::visit_expr(const TupleGetItemNode* op) {
  auto new_tuple = visit(op->tuple);
  auto ret = RelayTupleGetItem(new_tuple, op->index, op->span);
  ret->checked_type_ = op->checked_type_;
  return ret;
}

RelayExpr ToRelay::visit(const Expr& expr) {
  auto it = this->memo_.find(expr);
  if (it != this->memo_.end()) {
    return it->second;
  } else {
    RelayExpr new_expr = CSINNExprFunctor::visit(expr);
    memo_[expr] = new_expr;
    return new_expr;
  }
}

RelayExpr ToRelay::relay(const Expr& expr) {
  RelayExpr ret = visit(expr);
  return ret;
}

void HHBExprVisitor::visit(const Expr& expr) {
  auto it = visit_counter_.find(expr.get());
  if (it != visit_counter_.end()) {
    ++it->second;
  } else {
    using TParent = CSINNExprFunctor<void(const Expr&)>;
    TParent::visit(expr);
    visit_counter_.insert({expr.get(), 1});
  }
}

void HHBExprVisitor::visit_expr(const VarNode* op) {
  // this->VisitSpan(op->span);
  // if (op->type_annotation.defined()) {
  //   this->VisitType(op->type_annotation);
  // }
}

// void HHBExprVisitor::visit_expr(const GlobalVarNode* op) { this->VisitSpan(op->span); }

void HHBExprVisitor::visit_expr(const ConstantNode* op) {  // this->VisitSpan(op->span);
}

void HHBExprVisitor::visit_expr(const TupleNode* op) {
  // this->VisitSpan(op->span);
  for (auto field : op->fields) {
    this->visit(field);
  }
}

// void HHBExprVisitor::visit_expr(const FunctionNode* op) {
//   this->VisitSpan(op->span);
//   for (auto param : op->params) {
//     this->VisitExpr(param);
//   }

//   this->VisitExpr(op->body);
// }

void HHBExprVisitor::visit_expr(const CallNode* op) {
  // this->VisitSpan(op->span);
  // this->visit(op->op);

  // for (auto ty_arg : op->type_args) {
  //   this->VisitType(ty_arg);
  // }

  for (auto arg : op->args) {
    this->visit(arg);
  }
}

void HHBExprVisitor::visit_expr(const TupleGetItemNode* op) {
  // this->VisitSpan(op->span);
  this->visit(op->tuple);
}

// void HHBExprVisitor::visit_expr(const RefCreateNode* op) {
//   this->VisitSpan(op->span);
//   this->visit(op->value);
// }

// void HHBExprVisitor::visit_expr(const RefReadNode* op) {
//   this->VisitSpan(op->span);
//   this->visit(op->ref);
// }

// void HHBExprVisitor::visit_expr(const RefWriteNode* op) {
//   this->VisitSpan(op->span);
//   this->visit(op->ref);
//   this->visit(op->value);
// }

// void HHBExprVisitor::visit_expr(const ConstructorNode* op) {
//   for (const Type& t : op->inputs) {
//     this->VisitType(t);
//   }
//   this->VisitType(op->belong_to);
// }

// void HHBExprVisitor::visit_expr(const MatchNode* op) {
//   this->VisitSpan(op->span);
//   this->visit(op->data);
//   for (const Clause& c : op->clauses) {
//     this->VisitClause(c);
//   }
// }

// void HHBExprVisitor::VisitClause(const Clause& op) {
//   this->VisitPattern(op->lhs);
//   this->visit(op->rhs);
// }

// void HHBExprVisitor::VisitPattern(const Pattern& p) { return; }

// void HHBExprVisitor::VisitType(const Type& t) { return; }

// void HHBExprVisitor::VisitSpan(const Span& span) { return; }

Expr HHBExprMutator::visit_expr(const CallNode* op) {
  for (auto arg : op->args) {
    visit(arg);
  }
  return GetRef<Expr>(op);
}

Expr HHBExprMutator::visit_expr(const VarNode* op) { return GetRef<Expr>(op); }

Expr HHBExprMutator::visit_expr(const ConstantNode* op) { return GetRef<Expr>(op); }

Expr HHBExprMutator::visit_expr(const TupleNode* op) {
  tvm::Array<Expr> fields;
  fields.reserve(op->fields.size());
  for (auto field : op->fields) {
    auto new_field = visit(field);
    fields.push_back(new_field);
  }
  auto tuple = Tuple(fields, op->span);
  for (auto f : fields) {
    f.push_next_expr(tuple.get());
  }
  return tuple;
}

Expr HHBExprMutator::visit_expr(const TupleGetItemNode* op) {
  auto t = visit(op->tuple);
  if (op->tuple == t) {
    return GetRef<Expr>(op);
  } else {
    auto tuple = TupleGetItem(t, op->index, op->span);
    tuple->hhb_expr_extend_ = op->hhb_expr_extend_;
    t.push_next_expr(tuple.get(), op->index);
    return tuple;
  }
}

Expr HHBExprMutator::visit(const Expr& expr) {
  auto it = this->memo_.find(expr);
  if (it != this->memo_.end()) {
    return it->second;
  } else {
    Expr new_expr = CSINNExprFunctor::visit(expr);
    memo_[expr] = new_expr;
    return new_expr;
  }
}

void Pass::import_realy_expr(const RelayExpr& func) { expr = get_expr.expr(func); }

RelayExpr Pass::export_realy_expr() {
  RelayExpr ret = export_relay.relay(expr);
  return ret;
}

void Optimize::phase0() { /* TODO */
}
void Optimize::phase1() { /* TODO */
}
void Optimize::phase2() { /* TODO */
}
void Optimize::phase3() { /* TODO */
}

void Optimize::optimization() {
  // PHASE 0
  phase0();
  // user-defined phase-0
  target_define_phase0();
  // PHASE 1
  phase1();
  // user-defined phase-1
  target_define_phase1();
  // PHASE 2
  phase2();
  // user-defined phase-2
  target_define_phase2();
  // PHASE 3
  phase3();
  // user-defined phase-3
  target_define_phase3();
}
}  // namespace csinn
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
