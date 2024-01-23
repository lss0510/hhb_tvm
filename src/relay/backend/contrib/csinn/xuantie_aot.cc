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
 * \file src/relay/backend/contrib/csinn/xuantie_aot.cc
 * \brief Implementation of XuanTie CPU codegen APIs.
 */

#include "xuantie_aot.h"

#include <tvm/relay/runtime.h>

#include <algorithm>
#include <string>

namespace tvm {
namespace relay {
namespace contrib {
namespace csinn {
using namespace tvm::relay;

static string replace_illegal_string(string a) {
  std::string new_name = a;
  int pos;
  std::vector<char> illegal_str = {'.', '/', ':', '@'};
  for (size_t i = 0; i < illegal_str.size(); i++) {
    pos = new_name.find(illegal_str[i]);
    while (pos != -1) {
      new_name.replace(pos, 1, "_");
      pos = new_name.find(illegal_str[i]);
    }
  }

  return new_name;
}

static string callnode_name(const CallNode* call) {
  string ret;
  /* align to intrinsic supported node */
  if (IsOp(call, "qnn.csi.conv2d")) {
    auto attr = call->attrs.as<relay::qnn::QnnCSIConv2DAttrs>();
    ret = attr->layer_name.c_str();
  } else if (IsOp(call, "qnn.csi.tanh") || IsOp(call, "qnn.csi.relu") ||
             IsOp(call, "qnn.csi.relu6") || IsOp(call, "qnn.csi.sigmoid")) {
    auto attr = call->attrs.as<relay::qnn::QnnCSIUnaryAttrs>();
    ret = attr->layer_name.c_str();
  } else if (IsOp(call, "qnn.csi.add") || IsOp(call, "qnn.csi.div") || IsOp(call, "qnn.csi.mul") ||
             IsOp(call, "qnn.csi.subtract")) {
    auto attr = call->attrs.as<relay::qnn::QnnBinaryOpAttrs>();
    ret = attr->layer_name.c_str();
  } else if (IsOp(call, "qnn.csi.dense")) {
    auto attr = call->attrs.as<relay::qnn::QnnCSIDenseAttrs>();
    ret = attr->layer_name.c_str();
  } else if (IsOp(call, "qnn.csi.deconv2d")) {
    auto attr = call->attrs.as<relay::qnn::QnnCSIDeConv2DAttrs>();
    ret = attr->layer_name.c_str();
  } else if (IsOp(call, "qnn.csi.avgpool2d")) {
    auto attr = call->attrs.as<relay::qnn::QnnCSIAvgPool2DAttrs>();
    ret = attr->layer_name.c_str();
  } else if (IsOp(call, "qnn.csi.global_avgpool2d")) {
    auto attr = call->attrs.as<relay::qnn::QnnCSIGlobalAvgPoolAttrs>();
    ret = attr->layer_name.c_str();
  } else if (IsOp(call, "qnn.csi.concatenate")) {
    auto attr = call->attrs.as<relay::qnn::QnnConcatenateAttrs>();
    ret = attr->layer_name.c_str();
  } else if (IsOp(call, "qnn.csi.clip")) {
    auto attr = call->attrs.as<relay::qnn::QnnCSIClipAttrs>();
    ret = attr->layer_name.c_str();
  } else if (IsOp(call, "qnn.csi.maxpool2d")) {
    auto attr = call->attrs.as<relay::qnn::QnnCSIMaxPool2DAttrs>();
    ret = attr->layer_name.c_str();
  } else if (IsOp(call, "qnn.csi.global_maxpool2d")) {
    auto attr = call->attrs.as<relay::qnn::QnnCSIGlobalMaxPoolAttrs>();
    ret = attr->layer_name.c_str();
  } else if (IsOp(call, "qnn.csi.lrn")) {
    auto attr = call->attrs.as<relay::qnn::QnnCSILRNAttrs>();
    ret = attr->layer_name.c_str();
  } else if (IsOp(call, "qnn.csi.leaky_relu")) {
    auto attr = call->attrs.as<relay::qnn::QnnCSILeakyReluAttrs>();
    ret = attr->layer_name.c_str();
  } else if (IsOp(call, "qnn.csi.mean") || IsOp(call, "qnn.csi.maximum") ||
             IsOp(call, "qnn.csi.minimum")) {
    auto attr = call->attrs.as<relay::qnn::QnnCSIReduceAttrs>();
    ret = attr->layer_name.c_str();
  } else if (IsOp(call, "qnn.csi.prelu")) {
    auto attr = call->attrs.as<relay::qnn::QnnCSIPReluAttrs>();
    ret = attr->layer_name.c_str();
  } else if (IsOp(call, "qnn.csi.reshape")) {
    auto attr = call->attrs.as<relay::qnn::QnnCSIReshapeAttrs>();
    ret = attr->layer_name.c_str();
  } else if (IsOp(call, "qnn.csi.softmax")) {
    auto attr = call->attrs.as<relay::qnn::QnnCSIAxisAttrs>();
    ret = attr->layer_name.c_str();
  } else if (IsOp(call, "qnn.csi.space_to_depth")) {
    auto attr = call->attrs.as<relay::qnn::QnnCSISubPixelAttrs>();
    ret = attr->layer_name.c_str();
  } else if (IsOp(call, "qnn.csi.split")) {
    auto attr = call->attrs.as<relay::qnn::QnnCSISplitAttrs>();
    ret = attr->layer_name.c_str();
  } else if (IsOp(call, "qnn.csi.take")) {
    auto attr = call->attrs.as<relay::qnn::QnnCSITakeAttrs>();
    ret = attr->layer_name.c_str();
  } else if (IsOp(call, "qnn.csi.transpose")) {
    auto attr = call->attrs.as<relay::qnn::QnnCSITransposeAttrs>();
    ret = attr->layer_name.c_str();
  } else if (IsOp(call, "qnn.csi.upsampling")) {
    auto attr = call->attrs.as<relay::qnn::QnnCSIUpSamplingAttrs>();
    ret = attr->layer_name.c_str();
  } else if (IsOp(call, "qnn.csi.matmul")) {
    auto attr = call->attrs.as<relay::qnn::QnnCSIMatMulAttrs>();
    ret = attr->layer_name.c_str();
  } else {
    LOG(WARNING) << "Skip invalid call node, rename as: Unsupport_call_node_name";
    ret = "Unsupport_call_node_name";
  }
  ret = replace_illegal_string(ret);
  return ret;
}

static Var create_input(const Expr& arg, string name) {
  auto cshape = arg->get_shape();
  Array<PrimExpr> input_shape;
  for (auto s : cshape) {
    input_shape.push_back(PrimExpr(s));
  }
  auto tensor_type = TensorType(input_shape, DataType::Float(32));
  struct HHBExprExtend* extend = arg->hhb_expr_extend_;
  auto input = Var(name, extend, tensor_type);
  return input;
}

void GraphPartition::create_partition(const CallNode* call) {
  struct HHBExprExtend* extend = call->hhb_expr_extend_;
  string call_name = callnode_name(call);
  Array<Expr> args = call->args;
  for (uint i = 0; i < call->args.size(); i++) {
    auto input = create_input(call->args[i], call_name + to_string(i));
    args.Set(i, input);
  }

  auto subgraph = Call(call->op, args, extend, call->attrs, call->type_args, call->span);
  partitions_.push_back(static_cast<Expr>(subgraph));
}

void GraphPartition::visit_expr(const CallNode* call) {
  for (auto arg : call->args) {
    this->visit(arg);
  }

  create_partition(call);
}

void GraphPartition::graph_partition(const Expr& expr) {
  expr_ = expr;
  visit(expr);
}

tvm::Array<relay::Expr> change_args_dtype(const tvm::Array<relay::Expr> args, DataType base_dtype) {
  tvm::Array<relay::Expr> ret;

  for (uint i = 0; i < args.size(); i++) {
    if (auto var_node = args[i].as<relay::VarNode>()) {
      auto type = var_node->type_annotation.as<TensorTypeNode>();
      auto shape = type->shape;
      auto new_type = TensorType(shape, base_dtype);
      auto new_arg = relay::Var(var_node->vid, new_type, type->span);
      ret.push_back(new_arg);
    } else {
      LOG(FATAL) << "Unsupported Node: " << AsText(args[i], false);
    }
  }

  return ret;
}

RelayExpr XuanTie_AOT::tvm_realy_expr(const RelayExpr& expr, string* name) {
  auto subgraph = static_cast<const RelayCallNode*>(expr.get());
  tvm::Op op;
  RelayExpr ret;
  Array<relay::Expr> args = change_args_dtype(subgraph->args, base_dtype);
  if (relay::backend::IsOp(subgraph, "qnn.csi.conv2d")) {
    op = relay::Op::Get("nn.conv2d");
    auto attr = subgraph->attrs.as<relay::qnn::QnnCSIConv2DAttrs>();
    auto conv2d_attr = make_object<Conv2DAttrs>();
    auto weight_node = args[1].as<relay::VarNode>();
    auto weight_type = weight_node->type_annotation.as<TensorTypeNode>();
    conv2d_attr->strides = attr->strides;
    conv2d_attr->padding = attr->padding;
    conv2d_attr->dilation = attr->dilation;
    conv2d_attr->groups = attr->groups;
    conv2d_attr->channels = attr->channels;
    conv2d_attr->kernel_size = attr->kernel_size;
    conv2d_attr->data_layout = attr->data_layout;
    conv2d_attr->kernel_layout = attr->kernel_layout;
    conv2d_attr->out_layout = attr->out_layout;
    conv2d_attr->out_dtype = weight_type->dtype;
    *name = attr->layer_name.c_str();
    ret = relay::Call(op, {args[0], args[1]}, Attrs(conv2d_attr));

    auto bias_node = args[2].as<relay::VarNode>();
    auto bias_type = bias_node->type_annotation.as<TensorTypeNode>();
    auto bias_shape = bias_type->shape;
    if (bias_shape.size() != 0) {
      auto bias_attr = make_object<BiasAddAttrs>();
      bias_attr->axis = 1;
      op = relay::Op::Get("nn.bias_add");
      ret = relay::Call(op, {ret, args[2]}, Attrs(bias_attr));
    }
  } else if (relay::backend::IsOp(subgraph, "qnn.csi.conv2d_relu")) {
    op = relay::Op::Get("nn.conv2d");
    auto attr = subgraph->attrs.as<relay::qnn::QnnCSIConv2DAttrs>();
    auto conv2d_attr = make_object<Conv2DAttrs>();
    auto weight_node = args[1].as<relay::VarNode>();
    auto weight_type = weight_node->type_annotation.as<TensorTypeNode>();
    conv2d_attr->strides = attr->strides;
    conv2d_attr->padding = attr->padding;
    conv2d_attr->dilation = attr->dilation;
    conv2d_attr->groups = attr->groups;
    conv2d_attr->channels = attr->channels;
    conv2d_attr->kernel_size = attr->kernel_size;
    conv2d_attr->data_layout = attr->data_layout;
    conv2d_attr->kernel_layout = attr->kernel_layout;
    conv2d_attr->out_layout = attr->out_layout;
    conv2d_attr->out_dtype = weight_type->dtype;
    *name = attr->layer_name.c_str();
    ret = relay::Call(op, {args[0], args[1]}, Attrs(conv2d_attr));

    auto bias_node = args[2].as<relay::VarNode>();
    auto bias_type = bias_node->type_annotation.as<TensorTypeNode>();
    auto bias_shape = bias_type->shape;
    if (bias_shape.size() != 0) {
      auto bias_attr = make_object<BiasAddAttrs>();
      bias_attr->axis = 1;
      op = relay::Op::Get("nn.bias_add");
      ret = relay::Call(op, {ret, args[2]}, Attrs(bias_attr));
    }

    op = relay::Op::Get("nn.relu");
    ret = relay::Call(op, {ret});
  } else if (relay::backend::IsOp(subgraph, "qnn.csi.conv2d_relu6")) {
    op = relay::Op::Get("nn.conv2d");
    auto attr = subgraph->attrs.as<relay::qnn::QnnCSIConv2DAttrs>();
    auto conv2d_attr = make_object<Conv2DAttrs>();
    auto weight_node = args[1].as<relay::VarNode>();
    auto weight_type = weight_node->type_annotation.as<TensorTypeNode>();

    conv2d_attr->strides = attr->strides;
    conv2d_attr->padding = attr->padding;
    conv2d_attr->dilation = attr->dilation;
    conv2d_attr->groups = attr->groups;
    conv2d_attr->channels = attr->channels;
    conv2d_attr->kernel_size = attr->kernel_size;
    conv2d_attr->data_layout = attr->data_layout;
    conv2d_attr->kernel_layout = attr->kernel_layout;
    conv2d_attr->out_layout = attr->out_layout;
    conv2d_attr->out_dtype = weight_type->dtype;
    *name = attr->layer_name.c_str();
    ret = relay::Call(op, {args[0], args[1]}, Attrs(conv2d_attr));

    auto bias_node = args[2].as<relay::VarNode>();
    auto bias_type = bias_node->type_annotation.as<TensorTypeNode>();
    auto bias_shape = bias_type->shape;
    if (bias_shape.size() != 0) {
      auto bias_attr = make_object<BiasAddAttrs>();
      bias_attr->axis = 1;
      op = relay::Op::Get("nn.bias_add");
      ret = relay::Call(op, {ret, args[2]}, Attrs(bias_attr));
    }

    op = relay::Op::Get("clip");
    auto clip_attr = make_object<ClipAttrs>();
    clip_attr->a_min = 0;
    clip_attr->a_max = 6;
    ret = relay::Call(op, {ret}, Attrs(clip_attr));
  } else if (relay::backend::IsOp(subgraph, "qnn.csi.dense")) {
    op = relay::Op::Get("nn.dense");
    auto weight_node = args[1].as<relay::VarNode>();
    auto weight_type = weight_node->type_annotation.as<TensorTypeNode>();

    auto attr = subgraph->attrs.as<relay::qnn::QnnCSIDenseAttrs>();
    auto dense_attr = make_object<DenseAttrs>();
    dense_attr->units = attr->units;
    dense_attr->out_dtype = weight_type->dtype;
    *name = attr->layer_name.c_str();
    ret = relay::Call(op, {args[0], args[1]}, Attrs(dense_attr));

    auto bias_node = args[2].as<relay::VarNode>();
    auto bias_type = bias_node->type_annotation.as<TensorTypeNode>();
    auto bias_shape = bias_type->shape;
    if (bias_shape.size() != 0) {
      auto bias_attr = make_object<BiasAddAttrs>();
      bias_attr->axis = 1;
      op = relay::Op::Get("nn.bias_add");
      ret = relay::Call(op, {ret, args[2]}, Attrs(bias_attr));
    }
  } else if (relay::backend::IsOp(subgraph, "qnn.csi.relu")) {
    auto attr = subgraph->attrs.as<relay::qnn::QnnCSIUnaryAttrs>();
    *name = attr->layer_name.c_str();
    op = relay::Op::Get("nn.relu");
    ret = relay::Call(op, args);
  } else if (relay::backend::IsOp(subgraph, "qnn.csi.relu6")) {
    auto attr = subgraph->attrs.as<relay::qnn::QnnCSIUnaryAttrs>();
    *name = attr->layer_name.c_str();
    op = relay::Op::Get("clip");
    auto clip_attr = make_object<ClipAttrs>();
    clip_attr->a_min = 0;
    clip_attr->a_max = 6;
    ret = relay::Call(op, args, Attrs(clip_attr));
  } else if (relay::backend::IsOp(subgraph, "qnn.csi.avgpool2d")) {
    auto attr = subgraph->attrs.as<relay::qnn::QnnCSIAvgPool2DAttrs>();
    auto pool2d_attr = make_object<AvgPool2DAttrs>();
    pool2d_attr->pool_size = attr->pool_size;
    pool2d_attr->strides = attr->strides;
    pool2d_attr->dilation = attr->dilation;
    pool2d_attr->padding = attr->padding;
    pool2d_attr->layout = attr->layout;
    pool2d_attr->ceil_mode = attr->ceil_mode;
    pool2d_attr->count_include_pad = attr->count_include_pad;
    *name = attr->layer_name.c_str();
    op = relay::Op::Get("nn.avg_pool2d");
    ret = relay::Call(op, args, Attrs(pool2d_attr));
  } else if (relay::backend::IsOp(subgraph, "qnn.csi.global_avgpool2d")) {
    auto attr = subgraph->attrs.as<relay::qnn::QnnCSIGlobalAvgPoolAttrs>();
    auto pool2d_attr = make_object<GlobalPool2DAttrs>();
    pool2d_attr->layout = attr->layout;
    *name = attr->layer_name.c_str();
    op = relay::Op::Get("nn.global_avg_pool2d");
    ret = relay::Call(op, args, Attrs(pool2d_attr));
  } else if (relay::backend::IsOp(subgraph, "qnn.csi.maxpool2d")) {
    auto attr = subgraph->attrs.as<relay::qnn::QnnCSIMaxPool2DAttrs>();
    auto pool2d_attr = make_object<MaxPool2DAttrs>();
    pool2d_attr->pool_size = attr->pool_size;
    pool2d_attr->strides = attr->strides;
    pool2d_attr->dilation = attr->dilation;
    pool2d_attr->padding = attr->padding;
    pool2d_attr->layout = attr->layout;
    pool2d_attr->ceil_mode = attr->ceil_mode;
    *name = attr->layer_name.c_str();
    op = relay::Op::Get("nn.max_pool2d");
    ret = relay::Call(op, args, Attrs(pool2d_attr));
  } else if (relay::backend::IsOp(subgraph, "qnn.csi.global_maxpool2d")) {
    auto attr = subgraph->attrs.as<relay::qnn::QnnCSIGlobalMaxPoolAttrs>();
    auto pool2d_attr = make_object<GlobalPool2DAttrs>();
    pool2d_attr->layout = attr->layout;
    *name = attr->layer_name.c_str();
    op = relay::Op::Get("nn.global_max_pool2d");
    ret = relay::Call(op, args, Attrs(pool2d_attr));
  } else if (relay::backend::IsOp(subgraph, "qnn.csi.add")) {
    auto attr = subgraph->attrs.as<relay::qnn::QnnBinaryOpAttrs>();
    *name = attr->layer_name.c_str();
    op = relay::Op::Get("add");
    ret = relay::Call(op, args);
  } else if (relay::backend::IsOp(subgraph, "qnn.csi.reshape")) {
    auto attr = subgraph->attrs.as<relay::qnn::QnnCSIReshapeAttrs>();
    auto reshape_attr = make_object<ReshapeAttrs>();
    reshape_attr->newshape = attr->newshape;
    *name = attr->layer_name.c_str();
    op = relay::Op::Get("reshape");
    ret = relay::Call(op, args, Attrs(reshape_attr));
  } else if (relay::backend::IsOp(subgraph, "qnn.csi.softmax")) {
    auto attr = subgraph->attrs.as<relay::qnn::QnnCSIAxisAttrs>();
    auto softmax_attr = make_object<SoftmaxAttrs>();
    softmax_attr->axis = attr->axis;
    *name = attr->layer_name.c_str();
    op = relay::Op::Get("nn.softmax");
    ret = relay::Call(op, args, Attrs(softmax_attr));
  } else if (relay::backend::IsOp(subgraph, "qnn.csi.transpose")) {
    auto attr = subgraph->attrs.as<relay::qnn::QnnCSITransposeAttrs>();
    auto transpose_attr = make_object<TransposeAttrs>();
    transpose_attr->axes = attr->axes;
    *name = attr->layer_name.c_str();
    op = relay::Op::Get("transpose");
    ret = relay::Call(op, args, Attrs(transpose_attr));
  } else if (relay::backend::IsOp(subgraph, "qnn.csi.subtract")) {
    auto attr = subgraph->attrs.as<relay::qnn::QnnBinaryOpAttrs>();
    *name = attr->layer_name.c_str();
    op = relay::Op::Get("subtract");
    ret = relay::Call(op, args);
  } else if (relay::backend::IsOp(subgraph, "qnn.csi.mul")) {
    auto attr = subgraph->attrs.as<relay::qnn::QnnBinaryOpAttrs>();
    *name = attr->layer_name.c_str();
    op = relay::Op::Get("multiply");
    ret = relay::Call(op, args);
  } else if (relay::backend::IsOp(subgraph, "qnn.csi.div")) {
    auto attr = subgraph->attrs.as<relay::qnn::QnnBinaryOpAttrs>();
    *name = attr->layer_name.c_str();
    op = relay::Op::Get("divide");
    ret = relay::Call(op, args);
  } else if (relay::backend::IsOp(subgraph, "qnn.csi.matmul")) {
    auto attr = subgraph->attrs.as<relay::qnn::QnnCSIMatMulAttrs>();
    auto batch_matmul_attr = make_object<BatchMatmulAttrs>();
    batch_matmul_attr->transpose_a = attr->transpose_a;
    batch_matmul_attr->transpose_b = attr->transpose_b;
    *name = attr->layer_name.c_str();
    op = relay::Op::Get("nn.batch_matmul");
    ret = relay::Call(op, {args[0], args[1]}, Attrs(batch_matmul_attr));
  } else {
    LOG(WARNING) << "Skip invalid qnn subgraph, start with op: " << AsText(subgraph->op, false);
    *name = "Unsupport subgraph";
  }
  return ret;
}

void XuanTie_AOT::generate_c_intrinsic() {
  Target llvm_tgt = Target("c -keys=riscv -device=rvv --system-lib --runtime=c");
  Array<Target> targets = {llvm_tgt};
  auto pfb = tvm::runtime::Registry::Get("relay.build_module._BuildModule");
  auto module_get_import = tvm::runtime::Registry::Get("runtime.ModuleGetImport");
  auto module_get_source = tvm::runtime::Registry::Get("runtime.ModuleGetSource");

  for (auto partition : partitions_) {
    tvm::runtime::Module build_mod = (*pfb)();
    auto build_f = build_mod.GetFunction("build", false);
    auto mod_f = build_mod.GetFunction("get_module", false);

    auto qnn_expr = export_relay_.relay(partition);

    string expr_name;

    auto rexpr = tvm_realy_expr(qnn_expr, &expr_name);
    if (expr_name == "Unsupport subgraph") {
      /* skip unsupport partition */
      continue;
    }
    auto relay_mod = tvm::IRModule::FromExpr(rexpr);
    ICHECK(relay_mod.defined()) << "Module must be defined";
    build_f(relay_mod, targets, llvm_tgt, Executor::Create("aot"), Runtime::Create("crt"),
            WorkspaceMemoryPools(), ConstantMemoryPools(), replace_illegal_string(expr_name));
    tvm::runtime::Module mod = mod_f();

    tvm::runtime::Module c_source_mod = (*module_get_import)(mod, 0);
    std::string c_source_code = (*module_get_source)(c_source_mod, "c");
    c_intrinsic_[expr_name] = c_source_code;
  }
}

void XuanTie_AOT::compile(const Expr& expr) {
  expr_ = expr;

  graph_partition_ = new GraphPartition();
  graph_partition_->graph_partition(expr_);
  partitions_ = graph_partition_->get_partitions();

  generate_c_intrinsic();
}

string emit_alloc() {
  std::ostringstream code_stream;
  code_stream << "#include \"csi_nn.h\"\n";
  code_stream << "#include \"shl_memory.h\"\n";
  code_stream << "void* TVMBackendAllocWorkspace(int device_type, int device_id, uint64_t nbytes, "
                 "int dtype_code_hint, int dtype_bits_hint) {";
  code_stream << "return shl_mem_alloc(nbytes);\n";
  code_stream << "}\n";
  code_stream << "int TVMBackendFreeWorkspace(int device_type, int device_id, void* ptr) {";
  code_stream << "shl_mem_free(ptr);\n";
  code_stream << "return 0;\n";
  code_stream << "}\n";
  return code_stream.str();
}

string XuanTie_AOT::get_ccode() {
  std::ostringstream code_stream;

  code_stream << emit_alloc();

  for (auto s : c_intrinsic_) {
    code_stream << s.second;
  }

  code_stream << "#include \"shl_tvmgen.h\"\n";
  code_stream << "static struct shl_tvmgen_name_func hhb_gen_func_map[" << c_intrinsic_.size()
              << "];\n";
  code_stream << "void hhb_gen_register() {\n";
  int index = 0;

  for (auto s : c_intrinsic_) {
    code_stream << "hhb_gen_func_map[" << index << "].name = \"" << s.first << "\";\n";
    code_stream << "hhb_gen_func_map[" << index << "].ptr = " << replace_illegal_string(s.first)
                << "___tvm_main__;\n";
    code_stream << "hhb_gen_func_map[" << index << "].opt_method = CSINN_OPT_TVMGEN;\n";
    index++;
  }

  code_stream << "shl_tvmgen_map_reg(hhb_gen_func_map, " << c_intrinsic_.size() << ");\n";
  code_stream << "}";

  return code_stream.str();
}

}  // namespace csinn
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
