# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name, unused-argument, too-many-lines, import-outside-toplevel
# pylint: disable=consider-using-enumerate, no-else-return, unused-variable
# pylint: disable=inconsistent-return-statements, logging-not-lazy, arguments-differ
# pylint: disable=too-many-nested-blocks, no-else-continue
"""Optimization passess for qnn."""
import logging
import numpy as np

from tvm import relay
from tvm.relay.transform import function_pass
from tvm.relay.expr import Var, Call, TupleGetItem, Constant, Tuple, const
from tvm.relay.frontend.common import infer_shape as _infer_shape
from tvm.relay import function
from tvm.ir import transform

from ..ir.base import _qnn_attrs, _get_csi_op, csi_op
from .calibrate import CONST, USE_MINMAX, USE_SCALE

LOG = 25
logger = logging.getLogger("HHB")


def is_valid(param):
    """param = [1, 1, 0, scale, zp], if scale==0 and zp==0
    this param is not initialized or valid.
    """
    res = False
    if tuple(param[-2:]) != (0, 0):
        res = True
    return res


@function_pass(opt_level=1)
class CurrentInputAndPreOutput:
    """Ensure the input's quant params of current op is the same with
    the output's quant params of previous op.
    """

    def transform_funciton(self, func, mod, ctx):
        """_summary_"""
        func = relay.Function(func.params, func.body, None, func.type_params, func.attrs)

        class CurrentInputAndPreOutputMutator(relay.ExprMutator):
            """_summary_

            Args:
                relay (_type_): _description_
            """

            def visit_call(self, call):
                op_args = [self.visit(arg) for arg in call.args]
                new_op_attrs = _qnn_attrs(call.attrs)

                for i, arg in enumerate(op_args):
                    if isinstance(arg, Call):
                        arg_attrs = _qnn_attrs(arg.attrs)

                        if is_valid(arg_attrs["q_params"][-1]):
                            new_op_attrs["q_params"][i] = arg_attrs["q_params"][-1]
                        elif is_valid(new_op_attrs["q_params"][i]):
                            arg_attrs["q_params"][-1] = new_op_attrs["q_params"][i]
                            op_args[i] = csi_op().all_handle[arg.op.name](*(arg.args), **arg_attrs)
                    # (Fixme@chenf): TupleGetItem and Tuple node's params should be fixed.
                    elif isinstance(arg, TupleGetItem):
                        arg_attrs = _qnn_attrs(arg.tuple_value.attrs)

                        in_num = len(arg.tuple_value.args)
                        if tuple(arg_attrs["q_params"][in_num + arg.index][-2:]) != (0, 0):
                            new_op_attrs["q_params"][i] = arg_attrs["q_params"][in_num + arg.index]
                    elif isinstance(arg, Tuple):
                        for j in range(len(arg)):
                            curr_node = arg.field[j]
                            if isinstance(curr_node, Call):
                                curr_attrs = _qnn_attrs(curr_node.attrs)
                                if tuple(curr_attrs["q_params"][-1][-2:]) != (0, 0):
                                    new_op_attrs["q_params"][j] = curr_attrs["q_params"][-1]
                            elif isinstance(curr_node, TupleGetItem):
                                curr_attrs = _qnn_attrs(curr_node.attrs)

                                in_num = len(arg.tuple_value.args)
                                if tuple(curr_attrs["q_params"][in_num + curr_node.index][-2:]) != (
                                    0,
                                    0,
                                ):
                                    new_op_attrs["q_params"][j] = curr_attrs["q_params"][
                                        in_num + curr_node.index
                                    ]
                return csi_op().all_handle[call.op.name](*op_args, **new_op_attrs)

        return CurrentInputAndPreOutputMutator().visit(func)


@function_pass(opt_level=1)
class CurrentInputAndCurrentOutput:
    """Ensure the input's quant params of current op is the same with
    the output's quant params of current op."""

    def transform_funciton(self, func, mod, ctx):
        """_summary_"""
        func = relay.Function(func.params, func.body, None, func.type_params, func.attrs)

        class CurrentInputAndCurrentOutputMutator(relay.ExprMutator):
            """_summary_

            Args:
                relay (_type_): _description_
            """

            def visit_call(self, call):
                op_args = [self.visit(arg) for arg in call.args]
                new_op_attrs = _qnn_attrs(call.attrs)

                out2in_list = [
                    "qnn.csi.reshape",
                    "qnn.csi.upsampling",
                    "qnn.csi.transpose",
                    "qnn.csi.mean",
                    "qnn.csi.relu",
                    "qnn.csi.relu6",
                    "qnn.csi.avgpool2d",
                    "qnn.csi.maxpool2d",
                    "qnn.csi.global_avgpool2d",
                    "qnn.csi.global_maxpool2d",
                    "qnn.csi.strided_slice",
                ]

                if call.op.name in out2in_list:
                    if (
                        is_valid(new_op_attrs["q_params"][-1])
                        and is_valid(new_op_attrs["q_params"][0])
                        and tuple(new_op_attrs["q_params"][-1][-2:])
                        != tuple(new_op_attrs["q_params"][0][-2:])
                    ):
                        logger.warning(
                            "%s:%s has different quant info for input:%s/output:%s"
                            % (
                                call.op.name,
                                new_op_attrs["layer_name"],
                                tuple(new_op_attrs["q_params"][0][-2:]),
                                tuple(new_op_attrs["q_params"][-1][-2:]),
                            )
                        )
                    if is_valid(new_op_attrs["q_params"][-1]) and (
                        not is_valid(new_op_attrs["q_params"][0])
                    ):
                        new_op_attrs["q_params"][0] = new_op_attrs["q_params"][-1]
                        logger.warning(
                            "The quant info of output is copied to input in %s:%s"
                            % (call.op.name, new_op_attrs["layer_name"])
                        )
                    elif is_valid(new_op_attrs["q_params"][0]) and (
                        not is_valid(new_op_attrs["q_params"][-1])
                    ):
                        new_op_attrs["q_params"][-1] = new_op_attrs["q_params"][0]
                        logger.warning(
                            "The quant info of input is copied to output in %s:%s"
                            % (call.op.name, new_op_attrs["layer_name"])
                        )

                return csi_op().all_handle[call.op.name](*op_args, **new_op_attrs)

        return CurrentInputAndCurrentOutputMutator().visit(func)


def unify_quant_params(mod):
    old_level = logger.getEffectiveLevel()
    logger.setLevel(logging.ERROR)
    mod = transform.Sequential([CurrentInputAndCurrentOutput()])(mod)
    logger.setLevel(old_level)
    mod = transform.Sequential([CurrentInputAndPreOutput()])(mod)
    mod = transform.Sequential([CurrentInputAndCurrentOutput()])(mod)
    mod = transform.Sequential([CurrentInputAndPreOutput()])(mod)
    return mod


@function_pass(opt_level=1)
class OptimizeShapeCheck:
    """Optimize shape check layer"""

    def transform_function(self, func, mod, ctx):
        """patten and convert op"""
        func = relay.Function(func.params, func.body, None, func.type_params, func.attrs)

        class OptimizeShapeCheckMutator(relay.ExprMutator):
            """optimize shape"""

            def visit_call(self, call):
                op_args = [self.visit(arg) for arg in call.args]

                if call.op.name in [
                    "qnn.csi.add",
                    "qnn.csi.mul",
                    "qnn.csi.subtract",
                    "qnn.csi.div",
                    "qnn.csi.power",
                    "qnn.csi.minimum",
                    "qnn.csi.maximum",
                ]:
                    if isinstance(op_args[1], Constant) and len(_infer_shape(op_args[1])) == 0:
                        dtype = (
                            op_args[1]._checked_type_.dtype
                            if op_args[1]._checked_type_
                            else "float32"
                        )
                        value = op_args[1].data.asnumpy().tolist()
                        op_args[1] = const(np.array([value]).astype(dtype), dtype)
                new_call = Call(call.op, op_args, call.attrs, call.type_args, call.span)
                return new_call

            def visit_function(self, fn):
                new_params = [self.visit(x) for x in fn.params]
                new_body = self.visit(fn.body)
                return function.Function(list(new_params), new_body)

        return OptimizeShapeCheckMutator().visit(func)


@function_pass(opt_level=1)
class UpdataQparams:
    """update attr for layers"""

    def __init__(self, indexd_graph):
        self.indexd_graph = indexd_graph

    def transform_function(self, func, mod, ctx):
        """patten and convert op"""
        func = relay.Function(func.params, func.body, None, func.type_params, func.attrs)
        updata_qparams = self

        class UpdataQparamsMutator(relay.ExprMutator):
            """_summary_

            Args:
                relay (_type_): _description_
            """

            def visit_call(self, call):
                op_args = [self.visit(arg) for arg in call.args]
                node_name = hash(call)
                if node_name in updata_qparams.indexd_graph:
                    attrs = updata_qparams.indexd_graph[node_name].attr
                else:
                    attrs = _qnn_attrs(call.attrs)
                new_call = _get_csi_op(call.op.name)(*op_args, **attrs)
                return new_call

        return UpdataQparamsMutator().visit(func)


@function_pass(opt_level=1)
class InsertAddBeforeConcat:
    """Optimize concat layer"""

    def __init__(self, op_list):
        self.insert_list = ["qnn.csi." + op for op in op_list]
        self.concat_input = []

    def insert_add(self, inputs, q_params):
        """insert op"""

        in_shape = _infer_shape(inputs)
        zeros = np.ones(in_shape, np.float32)
        zeros = relay.expr.const(zeros, dtype="float32")
        add_q = [1, 0, 1, 1.0, 1.0]
        new_q_params = [q_params[-1], add_q, q_params[-1]]
        return relay.qnn.op.csi_mul(inputs, zeros, new_q_params)

    def transform_function(self, func, mod, ctx):
        """patten and convert op"""
        func = relay.Function(func.params, func.body, None, func.type_params, func.attrs)
        insert_add_ = self

        class InsertAddBeforeConcatMutator(relay.ExprMutator):
            """_summary_

            Args:
                relay (_type_): _description_
            """

            def visit_call(self, call):
                op_args = [self.visit(arg) for arg in call.args]
                current_attrs = _qnn_attrs(call.attrs)
                if call.op.name == "qnn.csi.concatenate":
                    new_tuple_args = [[] for _ in op_args[0]]
                    for idx, pre_call in enumerate(op_args[0]):
                        new_tuple_args[idx] = op_args[0][idx]
                        if isinstance(pre_call, Call):
                            if pre_call.op.name in insert_add_.insert_list:
                                pre_attrs = _qnn_attrs(pre_call.attrs)
                                new_pre_call = insert_add_.insert_add(
                                    pre_call, pre_attrs["q_params"]
                                )
                                new_tuple_args[idx] = new_pre_call
                            elif pre_call in insert_add_.concat_input:
                                pre_attrs = _qnn_attrs(pre_call.attrs)
                                new_pre_call = insert_add_.insert_add(
                                    pre_call, pre_attrs["q_params"]
                                )
                                new_tuple_args[idx] = new_pre_call
                            insert_add_.concat_input.append(pre_call)

                    new_current_call = relay.qnn.op.csi_concatenate(
                        Tuple(new_tuple_args), **current_attrs
                    )
                    return new_current_call

                return Call(call.op, op_args, call.attrs, call.type_args, call.span)

        return InsertAddBeforeConcatMutator().visit(func)


def optimize_quantization(mod, broadcast_quantization=False, target=""):
    """Optimize quantization for th1520"""

    class Node:
        """Indexed node"""

        def __init__(self, name, op_name, attr, inputs):
            self.name = name
            self.op_name = op_name
            self.attr = attr
            self.call = None  # for debug
            self.inputs = inputs
            self.outputs = list()
            self.change_in = dict()
            self.change_out = dict()

        def get_input_idx(self, input_name):
            for i, j in enumerate(self.inputs):
                if j == input_name:
                    return i
            raise Exception("Can't find input!.")

    class CreateIndexedGraph(relay.ExprVisitor):
        """create indexed graph"""

        def __init__(self, mod, target):
            super(CreateIndexedGraph, self).__init__()
            self.target = target
            self.indexd_graph = dict()
            self.mod = mod
            self.need_change = set()
            self.visit(self.mod["main"])

        def visit_call(self, call):
            _ = [self.visit(arg) for arg in call.args]
            attrs = _qnn_attrs(call.attrs)
            node_name = hash(call)

            pre_layers = call.args if call.op.name != "qnn.csi.concatenate" else call.args[0]

            inputs = []
            for pre_layer in pre_layers:
                if isinstance(pre_layer, (Var, Constant)):
                    continue
                if isinstance(pre_layer, TupleGetItem):
                    hash_pre = hash(pre_layer.tuple_value)
                else:
                    hash_pre = hash(pre_layer)
                inputs.append(hash_pre)
                if hash_pre not in self.indexd_graph:
                    raise Exception("Can't find pre node.")
                in_node = self.indexd_graph[hash_pre]
                in_node.outputs.append(node_name)
            self.indexd_graph[node_name] = Node(node_name, call.op.name, attrs, inputs)
            self.indexd_graph[node_name].call = call

        def get_graph(self):
            """return indexed graph"""
            return self.indexd_graph

        def update_node_in(self, node_name, in_name, qinfo):
            node = self.indexd_graph[node_name]
            node.change_in[in_name] = qinfo
            self.need_change.add(node_name)

        def update_node_out(self, node_name, in_name, qinfo):
            node = self.indexd_graph[node_name]
            node.change_out[in_name] = qinfo
            self.need_change.add(node_name)

        def th1520_qinfo_mutator(self, in2out_list, out2in_list):
            """qinfo mutator for th1520"""
            for node_name, node in self.indexd_graph.items():
                op_name = node.op_name
                if op_name in in2out_list:
                    if node.attr["q_params"][1] == node.attr["q_params"][0]:
                        continue
                    # in to out
                    node.attr["q_params"][1] = node.attr["q_params"][0]
                    # register for all out
                    for out_name in node.outputs:
                        self.update_node_in(out_name, node.name, node.attr["q_params"][0])

                elif op_name in out2in_list:
                    if op_name == "qnn.csi.concatenate":
                        for idx, in_name in enumerate(node.inputs):
                            node.attr["q_params"][idx] = node.attr["q_params"][-1]
                            in_node = self.indexd_graph[in_name]
                            if in_node.op_name == "qnn.csi.concatenate" and self.target in (
                                "th1520",
                                "hth1520",
                            ):
                                raise Exception("concat try to modifly pre concat out!")
                            if in_node.op_name == "qnn.csi.concatenate":
                                continue
                            self.update_node_out(in_name, node.name, node.attr["q_params"][-1])
                    else:
                        if node.attr["q_params"][1] == node.attr["q_params"][0]:
                            continue
                        # out to in
                        node.attr["q_params"][0] = node.attr["q_params"][1]
                        # for ops in first layer
                        if not node.inputs:
                            continue
                        # register for all inputs
                        for in_name in node.inputs:
                            self.update_node_out(in_name, node.name, node.attr["q_params"][0])

            while self.need_change:
                node_name = self.need_change.pop()
                node = self.indexd_graph[node_name]
                in_changed = False
                out_changed = False
                if len(node.change_out) > 1:
                    if node.op_name != "qnn.csi.split":
                        raise Exception(
                            "Multiple nodes attempt to modify the current node at the same time！"
                        )
                if node.change_in:
                    for in_node_name, qinfo in node.change_in.items():
                        in_idx = node.get_input_idx(in_node_name)
                        node.attr["q_params"][in_idx] = qinfo
                    in_changed = True
                    node.change_in.clear()

                if node.change_out:
                    if node.op_name == "qnn.csi.split":
                        for out_node_name, qinfo in node.change_out.items():
                            out_node = self.indexd_graph[out_node_name]

                            out_node_ins = []
                            for arg in out_node.call.args:
                                if isinstance(arg, Tuple):
                                    for a in arg:
                                        out_node_ins.append(a)
                                else:
                                    out_node_ins.append(arg)
                            for i_out_node in out_node_ins:
                                if not isinstance(i_out_node, TupleGetItem):
                                    continue
                                split_index = i_out_node.index
                                break
                            node.attr["q_params"][1 + split_index] = qinfo

                            for out_name in node.outputs:
                                if out_node_name == out_name:
                                    continue
                                if out_node.op_name in out2in_list:
                                    continue
                                out_node = self.indexd_graph[out_name]

                                out_node_ins = []
                                for arg in out_node.call.args:
                                    if isinstance(arg, Tuple):
                                        for a in arg:
                                            out_node_ins.append(a)
                                    else:
                                        out_node_ins.append(arg)

                                need_change = False
                                for o in out_node_ins:
                                    if not isinstance(o, TupleGetItem):
                                        continue
                                    if split_index == o.index:
                                        need_change = True
                                        break
                                if need_change:
                                    self.update_node_in(
                                        out_name, node.name, node.attr["q_params"][1 + split_index]
                                    )
                        node.change_out.clear()
                        continue
                    else:
                        for _, qinfo in node.change_out.items():
                            node.attr["q_params"][-1] = qinfo
                            break

                    # updat outputs
                    for out_name in node.outputs:
                        out_node = self.indexd_graph[out_name]
                        if out_node.op_name in out2in_list:
                            continue
                        self.update_node_in(out_name, node.name, node.attr["q_params"][-1])
                        out_changed = True
                    node.change_out.clear()

                if in_changed and out_changed:
                    if node.op_name in in2out_list + out2in_list:
                        raise Exception("Input and output qinfo can't be changed at the same time.")
                if node.op_name in ["qnn.csi.concatenate"]:
                    if in_changed:
                        new_min = np.min(node.attr["q_params"])
                        new_max = np.max(node.attr["q_params"])
                        node.attr["q_params"][-1] = node.attr["q_params"][-1][:3] + [
                            new_min,
                            new_max,
                        ]
                        # updata inputs
                        for idx, in_name in enumerate(node.inputs):
                            in_node = self.indexd_graph[in_name]
                            if in_node.op_name == "qnn.csi.concatenate" and self.target in (
                                "th1520",
                                "hth1520",
                            ):
                                raise Exception("concat try to modifly pre concat out!")
                            if in_node.op_name == "qnn.csi.concatenate":
                                break
                            in_node.change_out[node.name] = node.attr["q_params"][-1]
                            node.attr["q_params"][idx] = node.attr["q_params"][-1]
                            self.need_change.add(in_name)

                        # updat outputs
                        for out_name in node.outputs:
                            self.update_node_in(out_name, node.name, node.attr["q_params"][-1])

                    if out_changed:
                        raise Exception("Concat output cannot be modified！")

                elif node.op_name in in2out_list + out2in_list:
                    if in_changed:
                        # in to out
                        if node.attr["q_params"][-1] != node.attr["q_params"][0]:
                            node.attr["q_params"][-1] = node.attr["q_params"][0]
                            for out_name in node.outputs:
                                self.update_node_in(out_name, node.name, node.attr["q_params"][-1])

                    if out_changed:
                        # out to in
                        if node.attr["q_params"][0] != node.attr["q_params"][-1]:
                            node.attr["q_params"][0] = node.attr["q_params"][-1]
                            for in_name in node.inputs:
                                self.update_node_out(in_name, node.name, node.attr["q_params"][-1])

        def cpu_qinfo_mutator(self, in2out_list, out2in_list):
            """qinfo mutator for cpus"""
            for node_name, node in self.indexd_graph.items():
                op_name = node.op_name
                if op_name in in2out_list:
                    if node.attr["q_params"][1] == node.attr["q_params"][0]:
                        continue
                    # in to out
                    node.attr["q_params"][1] = node.attr["q_params"][0]
                    # register for all out
                    for out_name in node.outputs:
                        self.update_node_in(out_name, node.name, node.attr["q_params"][0])

            while self.need_change:
                node_name = self.need_change.pop()
                node = self.indexd_graph[node_name]
                in_changed = False
                out_changed = False
                if len(node.change_out) > 1:
                    if node.op_name != "qnn.csi.split":
                        raise Exception(
                            "Multiple nodes attempt to modify the current node at the same time！"
                        )
                if node.change_in:
                    for in_node_name, qinfo in node.change_in.items():
                        in_idx = node.get_input_idx(in_node_name)
                        node.attr["q_params"][in_idx] = qinfo
                    in_changed = True
                    node.change_in.clear()

                if node.change_out:
                    if node.op_name == "qnn.csi.split":
                        for out_node_name, qinfo in node.change_out.items():
                            out_node = self.indexd_graph[out_node_name]

                            out_node_ins = []
                            for arg in out_node.call.args:
                                if isinstance(arg, Tuple):
                                    for a in arg:
                                        out_node_ins.append(a)
                                else:
                                    out_node_ins.append(arg)
                            for i_out_node in out_node_ins:
                                if not isinstance(i_out_node, TupleGetItem):
                                    continue
                                split_index = i_out_node.index
                                break
                            node.attr["q_params"][1 + split_index] = qinfo

                            for out_name in node.outputs:
                                if out_node_name == out_name:
                                    continue
                                if out_node.op_name in out2in_list:
                                    continue
                                out_node = self.indexd_graph[out_name]

                                out_node_ins = []
                                for arg in out_node.call.args:
                                    if isinstance(arg, Tuple):
                                        for a in arg:
                                            out_node_ins.append(a)
                                    else:
                                        out_node_ins.append(arg)

                                need_change = False
                                for o in out_node_ins:
                                    if not isinstance(o, TupleGetItem):
                                        continue
                                    if split_index == o.index:
                                        need_change = True
                                        break
                                if need_change:
                                    self.update_node_in(
                                        out_name, node.name, node.attr["q_params"][1 + split_index]
                                    )
                        node.change_out.clear()
                        continue
                    else:
                        for _, qinfo in node.change_out.items():
                            node.attr["q_params"][-1] = qinfo
                            break

                    # updat outputs
                    for out_name in node.outputs:
                        out_node = self.indexd_graph[out_name]
                        if out_node.op_name in out2in_list:
                            continue
                        self.update_node_in(out_name, node.name, node.attr["q_params"][-1])
                        out_changed = True
                    node.change_out.clear()

                if in_changed and out_changed:
                    if node.op_name in in2out_list + out2in_list:
                        raise Exception("Input and output qinfo can't be changed at the same time.")

                elif node.op_name in in2out_list + out2in_list:
                    if in_changed:
                        # in to out
                        if node.attr["q_params"][-1] != node.attr["q_params"][0]:
                            node.attr["q_params"][-1] = node.attr["q_params"][0]
                            for out_name in node.outputs:
                                self.update_node_in(out_name, node.name, node.attr["q_params"][-1])
                    if out_changed:
                        # out to in
                        if node.attr["q_params"][0] != node.attr["q_params"][-1]:
                            node.attr["q_params"][0] = node.attr["q_params"][-1]
                            for in_name in node.inputs:
                                self.update_node_out(in_name, node.name, node.attr["q_params"][-1])

        def qinfo_exchange(self, in2out_list, out2in_list, target):
            """change node quant params"""
            in2out_list = ["qnn.csi." + op for op in in2out_list]
            out2in_list = ["qnn.csi." + op for op in out2in_list]
            if target in ("th1520", "hth1520"):
                self.th1520_qinfo_mutator(in2out_list, out2in_list)
            else:
                self.cpu_qinfo_mutator(in2out_list, out2in_list)

    mod = transform.Sequential([OptimizeShapeCheck()])(mod)
    if broadcast_quantization:
        if target in ["th1520", "hth1520"]:
            out2in_list = ["concatenate"]
            in2out_list = [
                "reshape",
                "upsampling",
                "transpose",
                "mean",
                "relu",
                "relu6",
                "avgpool2d",
                "maxpool2d",
                "global_avgpool2d",
                "global_maxpool2d",
                "strided_slice",
            ]
            mod = transform.Sequential([InsertAddBeforeConcat(out2in_list + in2out_list)])(mod)
        else:
            out2in_list = []
            in2out_list = ["transpose", "reshape", "upsampling", "maxpool2d", "strided_slice"]
        index_graph_creater = CreateIndexedGraph(mod, target)
        index_graph_creater.qinfo_exchange(in2out_list, out2in_list, target)
        mod = transform.Sequential([UpdataQparams(index_graph_creater.get_graph())])(mod)

    return mod


def create_qnn_diso_with_data(op_name, in_call, rhs_data, in_q, out_q, layer_name=""):
    """Create double input and single output qnn node with prepared quantization params."""
    from tvm.relay.backend.contrib.csinn_backend import QnnConfig, QuantCalculator

    q_config = QnnConfig()
    qc = QuantCalculator(q_config)

    def _get_scale_zp_value(cal_qinfo):
        scale = []
        zp = []
        for i in range(cal_qinfo.q_size):
            scale.append(float(cal_qinfo.qinfo[i].scale))
            zp.append(cal_qinfo.qinfo[i].zero_point)
        if cal_qinfo.q_size == 1:
            scale = scale[0]
            zp = zp[0]
        return scale, zp

    rhs_q_param = [CONST, USE_MINMAX, in_q[2]]

    channel_num = (len(in_q) - 3) // 2
    for _ in range(channel_num):
        rhs_q_param += [np.min(rhs_data), np.max(rhs_data)]
    rhs_quant_info = qc.get_quant_params(rhs_q_param, "input2")
    rhs_quantized_data = qc.quantize_weight(rhs_data, rhs_quant_info)
    rhs_node = relay.expr.const(rhs_quantized_data)
    rhs_scale, rhs_zp = _get_scale_zp_value(rhs_quant_info)
    true_rhs_q_param = [CONST, USE_SCALE, in_q[2]]
    if isinstance(rhs_scale, (tuple, list)):
        for s, zp in zip(rhs_scale, rhs_zp):
            true_rhs_q_param += [s, zp]
    else:
        true_rhs_q_param += [rhs_scale, rhs_zp]

    new_q_params = [in_q, true_rhs_q_param, out_q]

    new_call = csi_op().diso_handle[op_name](in_call, rhs_node, new_q_params, layer_name)
    return new_call


@function_pass(opt_level=1)
class QNNTh1520InsertAddBetweenLeakyReluAndAdd:
    """Due to accuracy issues, we should insert add op between leakyrely and and in the following
        situation:

    .. code-block:: text

        conv2d
          | \
          | leakyrelu
          |   /
           add

    Would become:

    .. code-block:: text

        conv2d
          | \
          | leakyrelu
          |   |
          |  add
          |   /
           add
            |
           add

    """

    def transform_function(self, func, mod, ctx):
        """Helper function to convert qnn ir."""
        func = relay.Function(func.params, func.body, None, func.type_params, func.attrs)

        class BetweenLeakyReLUAndAdd(relay.ExprMutator):
            """insert add between leakyrelu and and"""

            def visit_call(self, call):
                op_args = [self.visit(arg) for arg in call.args]
                call_attrs = _qnn_attrs(call.attrs)
                if call.op.name == "qnn.csi.add":
                    new_pre_list = []
                    is_match = False
                    for pre in op_args:
                        if isinstance(pre, Call) and pre.op.name == "qnn.csi.leaky_relu":
                            pre_attrs = _qnn_attrs(pre.attrs)
                            pre_shape = _infer_shape(pre)
                            add_data = np.ones(pre_shape, np.float32) * 2.0
                            add_data = add_data.astype(np.float32)
                            layer_name = f"after_" + pre_attrs["layer_name"]
                            add_call = create_qnn_diso_with_data(
                                "qnn.csi.add",
                                pre,
                                add_data,
                                pre_attrs["q_params"][-1],
                                pre_attrs["q_params"][-1],
                                layer_name,
                            )
                            new_pre_list.append(add_call)
                            is_match = True
                        else:
                            new_pre_list.append(pre)
                    new_add_call = csi_op().all_handle[call.op.name](*new_pre_list, **call_attrs)
                    if is_match:
                        pre_shape = _infer_shape(new_add_call)
                        add_data = np.ones(pre_shape, np.float32) * -2.0
                        add_data = add_data.astype(np.float32)
                        layer_name = f"after_" + call_attrs["layer_name"]
                        add_call = create_qnn_diso_with_data(
                            "qnn.csi.add",
                            new_add_call,
                            add_data,
                            call_attrs["q_params"][-1],
                            call_attrs["q_params"][-1],
                            layer_name,
                        )
                        return add_call
                    return new_add_call

                new_call = Call(call.op, op_args, call.attrs, call.type_args, call.span)
                return new_call

        return BetweenLeakyReLUAndAdd().visit(func)


@function_pass(opt_level=1)
class QNNTh1520InsertReluBetweenSigmoidAndMul:
    """Due to accuracy issues, we should insert relu op between sigmoid and mul in the following
        situation:

    .. code-block:: text

        conv2d
          | \
          | sigmoid
          |   /
           mul

    Would become:

    .. code-block:: text

        conv2d
          | \
          | sigmoid
          |   |
          |  relu
          |   /
           mul

    """

    def transform_function(self, func, mod, ctx):
        """Helper function to convert qnn ir."""
        func = relay.Function(func.params, func.body, None, func.type_params, func.attrs)

        class BetweenSigmoidAndMul(relay.ExprMutator):
            """insert relu between simoid and mul"""

            def visit_call(self, call):
                op_args = [self.visit(arg) for arg in call.args]
                if call.op.name == "qnn.csi.mul":
                    new_pre_list = []
                    for pre in op_args:
                        if isinstance(pre, Call) and pre.op.name == "qnn.csi.sigmoid":
                            sigmoid_attrs = _qnn_attrs(pre.attrs)
                            relay_params = [
                                sigmoid_attrs["q_params"][-1],
                                sigmoid_attrs["q_params"][-1],
                            ]
                            new_call = relay.qnn.op.csi_relu(
                                pre,
                                "float32",
                                q_params=relay_params,
                                layer_name="after_" + sigmoid_attrs["layer_name"] + "relu",
                            )

                            new_pre_list.append(new_call)
                        else:
                            new_pre_list.append(pre)
                    new_call = Call(call.op, new_pre_list, call.attrs, call.type_args, call.span)
                    return new_call
                new_call = Call(call.op, op_args, call.attrs, call.type_args, call.span)
                return new_call

        return BetweenSigmoidAndMul().visit(func)
