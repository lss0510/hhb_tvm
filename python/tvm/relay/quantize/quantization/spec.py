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
# pylint: disable=missing-docstring, unused-import
# pylint: disable=too-many-nested-blocks, no-else-continue
"""Optimization spec for qnn."""
import collections
import json
import logging
import numpy as np

import tvm
from tvm import relay
from tvm.relay.expr_functor import ExprVisitor
from tvm.relay.expr import RelayExpr
from tvm.relay.expr import Call, Var, Tuple, Constant, TupleGetItem
from tvm.relay.dataflow_pattern import (
    DFPatternCallback,
    is_constant,
    wildcard,
    is_op,
    rewrite,
)
from tvm.relay.frontend.common import infer_shape, infer_type
from tvm.relay.transform import function_pass

from ..ir.base import _qnn_attrs, csi_op


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
            json.dump(dtj.qnn_data, f, indent=2)
        return func


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
