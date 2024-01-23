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
# pylint: disable=invalid-name, wildcard-import, unused-wildcard-import
# pylint: disable=not-callable, import-outside-toplevel, unused-argument
"""Optimization passess for qnn."""
import collections
import json
import logging
import math

import numpy as np

import tvm
from tvm import relay
from tvm.relay.expr_functor import ExprVisitor
from tvm.relay.expr import RelayExpr
from tvm.relay.expr import Call, Var, Tuple, Constant, TupleGetItem
from ..dataflow_pattern import DFPatternCallback
from ..dataflow_pattern import is_constant
from ..dataflow_pattern import wildcard, is_op
from ..dataflow_pattern import rewrite
from ..frontend.common import infer_shape, infer_type
from ..transform import function_pass
from ._convert_to_csi import _qnn_attrs, csi_op
from ._convert_to_csi import CONST, ACTIVATION, PER_CHANNEL, PER_TENSOR, USE_MINMAX, USE_SCALE

LOG = 25
logger = logging.getLogger("HHB")


def is_invalid_q_params(q_param):
    """Support for per-channel quantization detection."""
    res = False
    if not q_param:
        return res

    channel_num = len(q_param) - 3

    assert channel_num % 2 == 0, f"Invalid_q_params: {q_param}"
    channel_num = channel_num // 2
    for i in range(channel_num):
        if tuple(q_param[i + 3 : i + 5]) == (0.0, 0.0):
            res = True
            break
    return res


def get_qnn_call_io_num(call: Call):
    """Get the numbers of input/output for specified call."""
    assert isinstance(call, Call), f"Only Support for Call, but get {type(call)}"
    call_attrs = _qnn_attrs(call.attrs)
    in_num = 0
    for arg in call.args:
        if isinstance(arg, Tuple):
            in_num += len(arg)
        else:
            in_num += 1
    out_num = len(call_attrs["q_params"]) - in_num
    assert out_num > 0, f"The number of call's inputs should be no less than 1, but get {out_num}"
    return in_num, out_num


def create_qnn_diso_with_data(op_name, in_call, rhs_data, in_q, out_q, layer_name=""):
    """Create double input and single output qnn node with prepared quantization params."""
    from .qnn2onnx import QnnConfig, QuantCalculator

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


class QNNNodeMap(object):
    """Convert original qnn module into node map that holds keys information."""

    class Node(object):
        """Internal class holds node info."""

        def __init__(self) -> None:
            # [(call_hash, index), ...]
            self.ins = []
            self.in_q_params = []
            self.out_q_params = []
            self.hash_value = None
            self.call_name = None

        def __eq__(self, __value: object) -> bool:
            return self.hash_value == __value.hash_value

        def __hash__(self) -> int:
            return self.hash_value

    def __init__(self) -> None:
        self.nodes = []
        self.hash2nodes = {}

    def create_empty_node(self) -> Node:
        return self.Node()

    def find_out_node(self, call_hash, index):
        """Get the specified output node of current call."""
        for node in self.nodes:
            for hash_index in node.ins:
                if (call_hash, index) == hash_index:
                    return node
        return None

    def create_map_from_module(self, mod):
        """Create node map with specified qnn module."""
        class_obj = self

        class CreateHashMap(relay.ExprVisitor):
            """Convert QNN ir into hash map"""

            def visit_call(self, call):
                _ = [self.visit(arg) for arg in call.args]
                call_attrs = _qnn_attrs(call.attrs)
                qnn_map_node = class_obj.create_empty_node()
                qnn_map_node.hash_value = hash(call)
                qnn_map_node.call_name = call.op.name

                in_num = 0
                for i, arg in enumerate(call.args):
                    if isinstance(arg, Tuple):
                        in_num += len(arg)
                        for a in arg:
                            if isinstance(a, TupleGetItem):
                                qnn_map_node.ins.append((hash(a.tuple_value), a.index))
                            else:
                                qnn_map_node.ins.append((hash(a), 0))
                    elif isinstance(arg, TupleGetItem):
                        qnn_map_node.ins.append((hash(arg.tuple_value), arg.index))
                        in_num += 1
                    else:
                        qnn_map_node.ins.append((hash(arg), 0))
                        in_num += 1

                for i, q_param in enumerate(call_attrs["q_params"]):
                    true_value = q_param if not is_invalid_q_params(q_param) else None
                    if i < in_num:
                        qnn_map_node.in_q_params.append(true_value)
                    else:
                        qnn_map_node.out_q_params.append(true_value)

                class_obj.nodes.append(qnn_map_node)
                class_obj.hash2nodes[qnn_map_node.hash_value] = qnn_map_node

        chm = CreateHashMap()
        chm.visit(mod["main"])


class QNNQuantizationSpec(object):
    """Define some quantization restrictions for different target in QNN."""

    def __init__(self, board) -> None:
        self._out2in_list = []
        self._in2out_list = [
            "qnn.csi.transpose",
            "qnn.csi.reshape",
            "qnn.csi.upsampling",
            "qnn.csi.maxpool2d",
            "qnn.csi.strided_slice",
        ]

        if board in ["th1520", "hth1520"]:
            _th1520 = [
                "qnn.csi.mean",
                "qnn.csi.relu",
                "qnn.csi.relu6",
                "qnn.csi.avgpool2d",
                "qnn.csi.global_avgpool2d",
                "qnn.csi.global_maxpool2d",
            ]
            self._in2out_list = self._in2out_list + _th1520
            self._out2in_list = self._out2in_list + ["qnn.csi.concatenate"]

    @property
    def out2in(self):
        return self._out2in_list

    @property
    def in2out(self):
        return self._in2out_list

    @property
    def miso(self):
        return ["qnn.csi.concatenate"]

    @property
    def simo(self):
        return ["qnn.csi.split"]

    @property
    def ignore_check(self):
        return ["qnn.csi.relu", "qnn.csi.relu6", "qnn.csi.mean"]


@function_pass(opt_level=1)
class QNNSeparateRepeatedQDQ:
    """Separate repeated QDQ structure with specified op.

    .. code-block:: text

        op1 -> quantize -> dequantize -> quantize -> dequantize -> op2

    Would become:

    .. code-block:: text

        op1 -> quantize -> dequantize -> [op] -> quantize -> dequantize -> op2
    """

    def __init__(self, op_name="qnn.csi.mul") -> None:
        self.op_name = op_name

    def create_specified_op(self, in_node: Call, dtype: str):
        """Generate specified qnn op."""
        if self.op_name == "qnn.csi.mul":
            rhs_value = np.array([1]).astype(dtype)
            dq_in_node = relay.const(rhs_value)
            scale_node = relay.const(1.0, dtype="float32")
            zp_node = relay.const(0, dtype="int32")

            rhs = relay.qnn.op.csi_dequantize(
                dq_in_node, scale_node, zp_node, axis=1, out_dtype="float32", q_params=[]
            )

            out = relay.qnn.op.csi_mul(
                in_node, rhs, q_params=[], layer_name="after_" + in_node.attrs.layer_name + "mul"
            )
        else:
            raise ValueError(f"Unsupport op: {self.op_name}")

        return out

    def transform_function(self, func, mod, ctx):
        """Helper function to convert qnn ir."""
        func = relay.Function(func.params, func.body, None, func.type_params, func.attrs)
        class_obj = self

        class InsertSpecifiedOp(DFPatternCallback):
            """Insert specified op between dequanitze and quantize ops."""

            def __init__(self, require_type=False, rewrite_once=False):
                super().__init__(require_type, rewrite_once)
                self.in_op = wildcard()

                # dequantize op
                self.scale1 = is_constant()
                self.zp1 = is_constant()
                self.dequantize = is_op("qnn.csi.dequantize")(self.in_op, self.scale1, self.zp1)

                # quantize op
                self.scale2 = is_constant()
                self.zp2 = is_constant()
                self.quantize = is_op("qnn.csi.quantize")(self.dequantize, self.scale2, self.zp2)

                self.pattern = self.quantize

            def callback(
                self, pre: RelayExpr, post: RelayExpr, node_map: tvm.ir.container.Map
            ) -> RelayExpr:
                dequantize_node = node_map[self.dequantize][0]
                quantize_node = node_map[self.quantize][0]
                dtype = infer_type(dequantize_node.args[0])

                inserted_op = class_obj.create_specified_op(dequantize_node, dtype)

                out = relay.qnn.op.csi_quantize(
                    inserted_op, self.scale2, self.zp2, **_qnn_attrs(quantize_node.attrs)
                )
                return out

        out = rewrite(InsertSpecifiedOp(), mod["main"].body)
        res = tvm.IRModule.from_expr(out)
        return res["main"]


@function_pass(opt_level=1)
class QNNFuseQDQ:
    """Fuse QDQ nodes into qnn ir.

    .. code-block:: text

        input -> quantize -> dequantize -> qnn_layer1 -> quantize -> dequantize -> qnn_layer2 ->
        quantize -> dequantize -> output

    Would become:

    .. code-block:: text

        input -> qnn_layer1 -> qnn_layer2 -> output

    """

    def __init__(self, config) -> None:
        self.config = config

    def transform_function(self, func, mod, ctx):
        """Helper function to convert qnn ir."""
        func = relay.Function(func.params, func.body, None, func.type_params, func.attrs)

        def _get_qdq_params(scale, zp, tensor_type):
            """Get the quantization params that meet the requirements of qnn."""
            q_param = [tensor_type, USE_SCALE]

            assert (
                scale.size == zp.size
            ), f"Mismatch size between scale:{scale.size} and zero_point:{zp.size}"
            if scale.size == 1:
                q_param += [PER_TENSOR]
            else:
                q_param += [PER_CHANNEL]
            scale = scale.tolist()
            zp = zp.tolist()
            if isinstance(scale, (tuple, list)):
                for s_zp in zip(scale, zp):
                    q_param += list(s_zp)
            else:
                q_param += [scale, zp]
            return q_param

        class FuseActivatiionQDQ(DFPatternCallback):
            r"""Extract quant info from quantize/dequantize ops and fuse them into previous op.

            op
            |
            quantize    ->   op with output quantization info (scale, zero_point)
            |
            dequantize

            """

            def __init__(self):
                super(FuseActivatiionQDQ, self).__init__()

                # any call
                self.call_patten = wildcard()(None)

                # quantize op
                self.scale1 = is_constant()
                self.zp1 = is_constant()
                self.quantize = is_op("qnn.csi.quantize")(self.call_patten, self.scale1, self.zp1)

                # dequantize op
                self.scale2 = is_constant()
                self.zp2 = is_constant()
                self.dequantize = is_op("qnn.csi.dequantize")(self.quantize, self.scale2, self.zp2)

                self.pattern = self.dequantize

            def callback(
                self, pre: RelayExpr, post: RelayExpr, node_map: tvm.ir.container.Map
            ) -> RelayExpr:
                call_node = node_map[self.call_patten][0]
                scale1_node = node_map[self.scale1][0]
                zp1_node = node_map[self.zp1][0]

                scale1_val = scale1_node.data.numpy()
                zp1_val = zp1_node.data.numpy()

                call_attrs = _qnn_attrs(call_node.attrs)
                # modify output quant params of call_node
                call_attrs["q_params"][-1] = _get_qdq_params(scale1_val, zp1_val, ACTIVATION)

                new_node = csi_op().all_handle[call_node.op.name](*call_node.args, **call_attrs)

                return new_node

        class FuseQDQActivation(relay.ExprMutator):
            r"""Extract quant info of input node from quantize/dequantize ops
                and fuse them into subsequent op.

            input          input
            |              |
            quantize   ->   op with input quantization info (scale, zero_point)
            |
            dequantize
            |
            op

            """

            def visit_call(self, call):
                op_args = [self.visit(arg) for arg in call.args]
                new_op_attrs = _qnn_attrs(call.attrs)
                new_args = list(op_args)

                for i, arg in enumerate(op_args):
                    if isinstance(arg, Call):
                        if arg.op.name == "qnn.csi.dequantize":
                            quant_node = arg.args[0]
                            if (
                                quant_node
                                and isinstance(quant_node, Call)
                                and quant_node.op.name == "qnn.csi.quantize"
                            ):
                                pre_node = quant_node.args[0]
                                scale_val = arg.args[1].data.numpy()
                                zp_val = arg.args[2].data.numpy()

                                new_op_attrs["q_params"][i] = _get_qdq_params(
                                    scale_val, zp_val, ACTIVATION
                                )
                                new_args[i] = pre_node
                    elif isinstance(arg, Tuple):
                        new_tuple = []
                        for j in range(len(arg)):
                            dequant_node = arg.fields[j]
                            if (
                                dequant_node
                                and isinstance(dequant_node, Call)
                                and dequant_node.op.name == "qnn.csi.dequantize"
                            ):
                                quant_node = dequant_node.args[0]
                                if (
                                    quant_node
                                    and isinstance(quant_node, Call)
                                    and quant_node.op.name == "qnn.csi.quantize"
                                ):
                                    pre_node = quant_node.args[0]
                                    scale_val = dequant_node.args[1].data.numpy()
                                    zp_val = dequant_node.args[2].data.numpy()

                                    new_op_attrs["q_params"][i + j] = _get_qdq_params(
                                        scale_val, zp_val, ACTIVATION
                                    )
                                    new_tuple.append(pre_node)
                                    continue
                            new_tuple.append(dequant_node)
                        new_args[i] = Tuple(new_tuple)
                return csi_op().all_handle[call.op.name](*new_args, **new_op_attrs)

        class FuseDequantize(relay.ExprMutator):
            r"""Fuse dequantize into op.

            dequantize
            |
            op         ->   op

            """

            def visit_call(self, call):
                op_args = [self.visit(arg) for arg in call.args]
                new_op_attrs = _qnn_attrs(call.attrs)
                new_args = list(op_args)

                for i, arg in enumerate(op_args):
                    if isinstance(arg, Call):
                        if arg.op.name == "qnn.csi.dequantize":
                            const_node = arg.args[0]
                            if const_node and isinstance(const_node, Constant):
                                scale_val = arg.args[1].data.numpy()
                                zp_value = arg.args[2].data.numpy()

                                new_op_attrs["q_params"][i] = _get_qdq_params(
                                    scale_val, zp_value, CONST
                                )
                                new_args[i] = const_node
                    elif isinstance(arg, Tuple):
                        new_tuple = []
                        for j in range(len(arg)):
                            dequant_node = arg.fields[j]
                            if (
                                dequant_node
                                and isinstance(dequant_node, Call)
                                and dequant_node.op.name == "qnn.csi.dequantize"
                            ):
                                const_node = dequant_node.args[0]
                                if const_node and isinstance(const_node, Constant):
                                    scale_val = dequant_node.args[1].data.numpy()
                                    zp_value = dequant_node.args[2].data.numpy()

                                    new_op_attrs["q_params"][i + j] = _get_qdq_params(
                                        scale_val, zp_value, CONST
                                    )
                                    continue
                            new_tuple.append(dequant_node)
                        new_args[i] = Tuple(new_tuple)
                return csi_op().all_handle[call.op.name](*new_args, **new_op_attrs)

        class AlignCurrentInputAndPreOutput(relay.ExprMutator):
            """Ensure the input's quant params of current op is the same with
            the output's quant params of previous op.
            """

            def __init__(self, qnn_map: QNNNodeMap):
                super().__init__()

                self.qnn_map = qnn_map

            def visit_call(self, call):
                op_args = [self.visit(arg) for arg in call.args]
                call_attrs = _qnn_attrs(call.attrs)

                call_node = self.qnn_map.hash2nodes[hash(call)]

                # deal with input quantization params
                new_in_q_params = []
                for i, in_q in enumerate(call_node.in_q_params):
                    if in_q is not None:
                        new_in_q_params.append(in_q)
                        continue
                    in_hash, out_idx = call_node.ins[i]
                    if in_hash not in self.qnn_map.hash2nodes:
                        new_in_q_params.append(call_attrs["q_params"][i])
                        continue
                    in_node = self.qnn_map.hash2nodes[in_hash]
                    if in_node.out_q_params[out_idx] is None:
                        new_in_q_params.append(call_attrs["q_params"][i])
                    else:
                        new_in_q_params.append(in_node.out_q_params[out_idx])

                # deal with output quantization params
                new_out_q_params = []
                for i, out_q in enumerate(call_node.out_q_params):
                    if out_q is not None:
                        new_out_q_params.append(out_q)
                        continue
                    out_node = self.qnn_map.find_out_node(call_node.hash_value, i)
                    if out_node:
                        for in_idx, (in_out_hash, _) in enumerate(out_node.ins):
                            if in_out_hash == call_node.hash_value:
                                new_out_q_params.append(out_node.in_q_params[in_idx])
                    else:
                        in_num = len(new_in_q_params)
                        new_out_q_params.append(call_attrs["q_params"][in_num + i])

                # create new call
                call_attrs["q_params"] = new_in_q_params + new_out_q_params
                return csi_op().all_handle[call.op.name](*op_args, **call_attrs)

        out = rewrite(FuseActivatiionQDQ(), mod["main"].body)
        res = tvm.IRModule.from_expr(out)

        res["main"] = FuseQDQActivation().visit(res["main"])
        res["main"] = FuseDequantize().visit(res["main"])

        # fuse biasadd
        from ._convert_to_csi import fuse_layer

        res = fuse_layer(res, self.config)

        qnn_map = QNNNodeMap()
        qnn_map.create_map_from_module(res)

        res["main"] = AlignCurrentInputAndPreOutput(qnn_map).visit(res["main"])
        return res["main"]


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


class QNNConvertDict(ExprVisitor):
    """Internal helper class to dump json."""

    def __init__(self):
        super().__init__()
        self.qnn_data = collections.OrderedDict()
        self.qnn_data["input_names"] = []
        self.qnn_data["layers"] = []

    def visit_var(self, var):
        self.qnn_data["input_names"].append(var.name_hint)

    def visit_call(self, call):
        _ = [self.visit(arg) for arg in call.args]
        call_attrs = _qnn_attrs(call.attrs)

        layer_data = collections.OrderedDict()
        layer_data["op_type"] = call.op.name
        layer_data["name"] = call_attrs.pop("layer_name")
        layer_data["hash_value"] = hash(call)

        q_params = call_attrs.pop("q_params")
        layer_data["attrs"] = collections.OrderedDict(sorted(call_attrs.items()))

        # input tensor
        input_data = []
        for i, arg in enumerate(call.args):
            arg_data = collections.OrderedDict()
            if isinstance(arg, Var):
                arg_data["name"] = arg.name_hint
                arg_data["dim"] = infer_shape(arg)
                arg_data["hash_value"] = hash(arg)
                arg_data["index"] = 0
                arg_data["q_param"] = q_params[i]
            elif isinstance(arg, Call):
                arg_data["name"] = arg.attrs.layer_name
                arg_data["dim"] = infer_shape(arg)
                arg_data["hash_value"] = hash(arg)
                arg_data["index"] = 0
                arg_data["q_param"] = q_params[i]
            elif isinstance(arg, TupleGetItem):
                true_call = arg.tuple_value
                arg_data["name"] = true_call.attrs.layer_name
                arg_data["dim"] = infer_shape(true_call).fields[arg.index].concrete_shape
                arg_data["hash_value"] = hash(true_call)
                arg_data["index"] = arg.index
                arg_data["q_param"] = q_params[i]
            elif isinstance(arg, Constant):
                data = arg.data.numpy()
                arg_data["name"] = arg.span.source_name.name if arg.span else "const_" + str(i)
                arg_data["dim"] = data.shape
                arg_data["hash_value"] = hash(arg)
                arg_data["index"] = 0
                arg_data["q_param"] = q_params[i]
                arg_data["data"] = data.tolist()
            elif isinstance(arg, Tuple):
                for j, a in enumerate(arg):
                    arg_data = collections.OrderedDict()
                    if isinstance(a, Var):
                        arg_data["name"] = a.name_hint
                        arg_data["dim"] = infer_shape(a)
                        arg_data["hash_value"] = hash(a)
                        arg_data["index"] = 0
                        arg_data["q_param"] = q_params[i + j]
                    elif isinstance(a, Call):
                        arg_data["name"] = a.attrs.layer_name
                        arg_data["dim"] = infer_shape(a)
                        arg_data["hash_value"] = hash(a)
                        arg_data["index"] = 0
                        arg_data["q_param"] = q_params[i + j]
                    elif isinstance(a, TupleGetItem):
                        true_call = a.tuple_value
                        arg_data["name"] = true_call.attrs.layer_name
                        arg_data["dim"] = infer_shape(true_call).fields[a.index].concrete_shape
                        arg_data["hash_value"] = hash(true_call)
                        arg_data["index"] = a.index
                        arg_data["q_param"] = q_params[i + j]
                    elif isinstance(a, Constant):
                        data = a.data.numpy()
                        arg_data["name"] = a.span.source_name.name if a.span else "const_" + str(i)
                        arg_data["dim"] = data.shape
                        arg_data["hash_value"] = hash(a)
                        arg_data["index"] = 0
                        arg_data["q_param"] = q_params[i + j]
                        arg_data["data"] = data.tolist()
                    input_data.append(arg_data)
                continue
            input_data.append(arg_data)
        layer_data["inputs"] = input_data

        # output tensor
        output_data = []
        o_shape = infer_shape(call)
        if isinstance(o_shape, (tuple, list)):
            data = collections.OrderedDict()
            data["name"] = ""
            data["dim"] = list(o_shape)
            data["is_const"] = 0
            output_data.append(data)
        else:
            for i in range(len(o_shape.fields)):
                data = collections.OrderedDict()
                data["name"] = ""
                data["dim"] = list(o_shape.fields[i].concrete_shape)
                data["is_const"] = 0
                output_data.append(data)
        layer_data["outputs"] = output_data

        self.qnn_data["layers"].append(layer_data)


@function_pass(opt_level=1)
class QNNDumpToJson:
    """Dump qnn ir into json file."""

    def __init__(self, tofile) -> None:
        self.tofile = tofile

    def transform_function(self, func, mod, ctx):
        """Helper function to convert qnn ir."""
        func = relay.Function(func.params, func.body, None, func.type_params, func.attrs)

        dtj = QNNConvertDict()
        dtj.visit(func)
        with open(self.tofile, "w") as f:
            try:
                import jsbeautifier

                options = jsbeautifier.default_options()
                options.indent_size = 2
                res = jsbeautifier.beautify(json.dumps(dtj.qnn_data), options)
                f.write(res)
            except ImportError:
                logging.warning(
                    "Recommend installing jsbeautifier to get better formatted JSON file."
                )
                json.dump(dtj.qnn_data, f, indent=2)
        return func


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
                            pre_shape = infer_shape(pre)
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
                        pre_shape = infer_shape(new_add_call)
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
class QNNCheckValidQuantParams:
    """Check whether the quantization params is valid. For examples;

    1. ensure that every tensor in the layer has quantization params;
    2. ensure that the quantization params of the input tensor in the current layer are consistent
        with the quantization params of the output tensor in the previous layer;
    3. some ops should meet the restriction of quantization.

    """

    def __init__(self, board) -> None:
        self.qnn_spec = QNNQuantizationSpec(board)

    def transform_function(self, func, mod, ctx):
        """Helper function to convert qnn ir."""
        func = relay.Function(func.params, func.body, None, func.type_params, func.attrs)
        class_obj = self

        class CheckQuantParams(relay.ExprVisitor):
            """Internal helper class for check."""

            def __init__(self):
                super(CheckQuantParams, self).__init__()
                self.no_complete_quant_param = []
                self.mismatch = []
                self.not_meet_restrict = []

            def visit_call(self, call):
                _ = [self.visit(arg) for arg in call.args]
                call_attrs = _qnn_attrs(call.attrs)

                # deal with case 1
                for idx, q_param in enumerate(call_attrs["q_params"]):
                    if is_invalid_q_params(q_param):
                        if (
                            call.op.name in [*csi_op().conv_handle.keys(), "qnn.csi.dense"]
                            and idx == 2
                        ):
                            # ignore bias
                            continue
                        self.no_complete_quant_param.append(call_attrs["layer_name"])
                        break

                # deal with case 2
                q_param_idx = 0
                for i, arg in enumerate(call.args):
                    if isinstance(arg, (Constant, Var)):
                        q_param_idx += 1
                    elif isinstance(arg, Call):
                        arg_attrs = _qnn_attrs(arg.attrs)
                        if tuple(call_attrs["q_params"][q_param_idx]) != tuple(
                            arg_attrs["q_params"][-1]
                        ):
                            self.mismatch.append(call_attrs["layer_name"])
                            break
                        q_param_idx += 1
                    elif isinstance(arg, TupleGetItem):
                        true_arg = arg.tuple_value
                        true_arg_in_num, _ = get_qnn_call_io_num(true_arg)
                        pre_attrs = _qnn_attrs(true_arg.attrs)
                        if tuple(pre_attrs["q_params"][true_arg_in_num + arg.index]) != tuple(
                            call_attrs["q_params"][q_param_idx]
                        ):
                            self.mismatch.append(call_attrs["layer_name"])
                            break
                        q_param_idx += 1
                    elif isinstance(arg, Tuple):
                        for j, a in enumerate(arg):
                            if isinstance(a, TupleGetItem):
                                true_a = a.tuple_value
                                true_a_in_num, _ = get_qnn_call_io_num(true_a)
                                true_a_attrs = _qnn_attrs(true_a.attrs)
                                if tuple(
                                    true_a_attrs["q_params"][true_a_in_num + a.index]
                                ) != tuple(call_attrs["q_params"][q_param_idx + j]):
                                    self.mismatch.append(call_attrs["layer_name"])
                                    break
                            elif isinstance(a, Call):
                                a_attrs = _qnn_attrs(a.attrs)
                                if tuple(call_attrs["q_params"][q_param_idx + j]) != tuple(
                                    a_attrs["q_params"][-1]
                                ):
                                    self.mismatch.append(call_attrs["layer_name"])
                                    break
                        q_param_idx += len(arg)
                    else:
                        q_param_idx += 1

                # deal with case 3
                in_num = 0
                for arg in enumerate(call.args):
                    if isinstance(arg, Tuple):
                        in_num += len(arg)
                    else:
                        in_num += 1
                out_num = len(call.args)
                if call.op.name in class_obj.qnn_spec.out2in:
                    assert out_num == 1, f"The num of output should be 1, but get {out_num}"
                    for i in range(in_num):
                        if tuple(call_attrs["q_params"][i]) != tuple(call_attrs["q_params"][-1]):
                            self.not_meet_restrict.append(call_attrs["layer_name"])
                            break
                elif (
                    call.op.name in class_obj.qnn_spec.in2out
                    and call.op.name not in class_obj.qnn_spec.ignore_check
                ):
                    assert in_num == 1, f"The num of input should be 1, but get {in_num}"
                    for i in range(out_num):
                        if tuple(call_attrs["q_params"][in_num + i]) != tuple(
                            call_attrs["q_params"][0]
                        ):
                            self.not_meet_restrict.append(call_attrs["layer_name"])
                            break

        cqp = CheckQuantParams()
        cqp.visit(func)
        if cqp.no_complete_quant_param:
            raise ValueError(
                f"There is incomplete quantization params in {cqp.no_complete_quant_param}"
            )
        if cqp.mismatch:
            raise ValueError(
                f"The quantization params of current layer mismatch that of the previous "
                f"layer: {cqp.mismatch}"
            )
        if cqp.not_meet_restrict:
            raise ValueError(
                f"The quantization params of these layers do not meet the "
                f"restrictions: {cqp.not_meet_restrict}"
            )

        return func


@function_pass(opt_level=1)
class QNNConvertReshapeToFlatten:
    """Convert reshape into flatten.

    .. code-block:: text

        input(n, 3, 2, 2) -> reshape(n, 12) -> output(n, 12)

    Or

    .. code-block:: text

        input(n, 3, 2, 2) -> reshape(n, -1) -> output(n, 12)

    Would become:

    .. code-block:: text

        input(n, 3, 2, 2) -> flatten -> output(n, 12)

    """

    def transform_function(self, func, mod, ctx):
        """Helper function to convert qnn ir."""
        func = relay.Function(func.params, func.body, None, func.type_params, func.attrs)

        class InterHelper(relay.ExprMutator):
            """Helper class"""

            def visit_call(self, call):
                op_args = [self.visit(arg) for arg in call.args]
                call_attrs = _qnn_attrs(call.attrs)

                if call.op.name == "qnn.csi.reshape":
                    in_shape = infer_shape(op_args[0])
                    newshape = call_attrs["newshape"]
                    if len(newshape) == 2 and in_shape[0] == newshape[0]:
                        new_call = relay.qnn.op.csi_flatten(
                            op_args[0],
                            out_dtype=call_attrs["out_dtype"],
                            q_params=call_attrs["q_params"],
                            layer_name=call_attrs["layer_name"],
                        )
                        return new_call
                return csi_op().all_handle[call.op.name](*op_args, **call_attrs)

        return InterHelper().visit(func)


@function_pass(opt_level=1)
class QNNFuseConvDepthtospace:
    """Fuse conv2d+depth2space into deconv."""

    def transform_function(self, func, mod, ctx):
        """Helper function to convert qnn ir."""
        func = relay.Function(func.params, func.body, None, func.type_params, func.attrs)

        class InterHelper(DFPatternCallback):
            """Helper class"""

            def __init__(self, require_type=False, rewrite_once=False):
                super().__init__(require_type, rewrite_once)

                self.input = wildcard()
                self.weight = is_constant()
                self.bias = is_constant()
                self.conv = is_op("qnn.csi.conv2d")(self.input, self.weight, self.bias).has_attr(
                    {"strides": [1, 1], "groups": 1}
                )
                self.d2s = is_op("qnn.csi.depth_to_space")(self.conv)

                self.pattern = self.d2s

            def callback(self, pre, post, node_map) -> RelayExpr:
                in_call = node_map[self.input][0]

                weight = node_map[self.weight][0]
                bias = node_map[self.bias][0]
                conv_call = node_map[self.conv][0]

                d2s_call = node_map[self.d2s][0]

                conv_attrs = _qnn_attrs(conv_call.attrs)
                d2s_attrs = _qnn_attrs(d2s_call.attrs)
                block_size = d2s_attrs["block_size"]

                hk, wk = conv_attrs["kernel_size"]
                deconv_kernel_size = [hk * block_size, wk * block_size]
                deconv_strides = [block_size, block_size]
                deconv_pad = [
                    (hk - 1 - conv_attrs["padding"][2]) * block_size,
                    (wk - 1 - conv_attrs["padding"][3]) * block_size,
                    (hk - 1 - conv_attrs["padding"][0]) * block_size,
                    (wk - 1 - conv_attrs["padding"][1]) * block_size,
                ]

                Fk, Pk, Hk, Wk = infer_shape(weight)
                Fk_deconv = Fk // (block_size * block_size)
                Pk_deconv = Pk
                Hk_deconv = Hk * block_size
                Wk_deconv = Wk * block_size
                weight_data = weight.data.numpy()
                # flip weight in x, y
                flipped_weight_data = np.zeros(weight_data.shape, dtype=weight_data.dtype)
                for f in range(Fk):
                    for c in range(Pk):
                        for h in range(Hk):
                            for w in range(Wk):
                                flipped_weight_data[f, c, h, w] = weight_data[
                                    f, c, Hk - 1 - h, Wk - 1 - w
                                ]

                # interleave the weight
                interleaved_weight_data = np.zeros(weight_data.shape, dtype=weight_data.dtype)
                for f in range(Fk):
                    for c in range(Pk):
                        for h in range(Hk):
                            for w in range(Wk):
                                idx = (f % Fk_deconv) * block_size * block_size + f // Fk_deconv
                                interleaved_weight_data[idx, c, h, w] = flipped_weight_data[
                                    f, c, h, w
                                ]

                #  combine weight into deconv weight
                deconv_weight_data_oihw = np.zeros(
                    (Fk_deconv, Pk_deconv, Hk_deconv, Wk_deconv), dtype=weight_data.dtype
                )
                for f in range(Fk):
                    for c in range(Pk):
                        for h in range(Hk):
                            for w in range(Wk):
                                deconv_weight_data_oihw[
                                    f // (block_size * block_size),
                                    c,
                                    h * block_size + f // block_size % block_size,
                                    w * block_size + f % block_size,
                                ] = interleaved_weight_data[f, c, h, w]

                deconv_weight_data_iohw = np.transpose(deconv_weight_data_oihw, (1, 0, 2, 3))
                deconv_weight = relay.const(deconv_weight_data_iohw)

                bias_data = bias.data.numpy()
                if isinstance(bias_data.tolist(), float) and math.isclose(bias_data.tolist(), 0.0):
                    deconv_bias = bias
                else:
                    interleaved_bias_data = np.zeros(bias_data.shape, bias_data.dtype)
                    for f in range(Fk):
                        idx = (f % Fk_deconv) * (block_size * block_size) + f // Fk_deconv
                        interleaved_bias_data[idx] = bias_data[f]
                    deconv_bias = relay.const(interleaved_bias_data)

                deconv_call = relay.qnn.op.csi_deconv2d(
                    in_call,
                    deconv_weight,
                    deconv_bias,
                    strides=deconv_strides,
                    padding=deconv_pad,
                    dilation=(1, 1),
                    groups=1,
                    channels=Fk_deconv,
                    kernel_size=deconv_kernel_size,
                    data_layout="NCHW",
                    kernel_layout="IOHW",
                    out_layout="",
                    output_padding=(0, 0),
                    out_dtype="float32",
                    q_params=conv_attrs["q_params"],
                    layer_name="deconv_" + conv_attrs["layer_name"],
                )
                return deconv_call

        out = rewrite(InterHelper(), mod["main"].body)
        res = tvm.IRModule.from_expr(out)

        return res["main"]
