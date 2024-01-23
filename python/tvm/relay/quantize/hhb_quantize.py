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
# pylint: disable=not-callable
"""Automatic quantization toolkit."""
import logging
import os

from tvm import relay
from tvm.ir import transform
from tvm.relay import transform as _transform
from tvm.relay import expr as _expr
from tvm.relay.frontend.common import create_span

from .ir.convert_to_relay import convert_to_relay
from .ir.relay_qnn import convert_to_csi_qnn
from .ir.base import csi_op, rename_call
from .optimization.qnn_layout_convert import csi_layout_convert
from .optimization.relay_opt import (
    FuseCacheMatMul,
    FuseLayerNormal,
    TConv1dAddT,
    FuseCacheConv1d,
    InsertNOp,
    InsertRelu,
)
from .optimization.qnn_fuse import (
    FuseActivateQuantInfo,
    FuseInputQuantInfo,
    FuseDequantizeOp,
    QNNFuseQDQ,
    QNNFuseConvDepthtospace,
    FuseWhereSoftmax,
    Resume4DimsMatMul,
    Conv2dSqueezeAdd,
    fuse_layer,
)
from .optimization.qnn_split import ConvSpliter
from .quantization.spec import QNNCheckValidQuantParams, QNNDumpToJson, QNNSeparateRepeatedQDQ
from .quantization.optimize import (
    QNNTh1520InsertReluBetweenSigmoidAndMul,
    QNNTh1520InsertAddBetweenLeakyReluAndAdd,
)
from .quantization.calibrate import calibration
from .quantization.optimize import unify_quant_params, optimize_quantization
from .quantization.hybird_quantize import DumpLayerOutput, ModelQuantizationInfo, to_json


LOG = 25
logger = logging.getLogger("HHB")


def _check_unsupported_ops(target, model):
    x86_op_list = [
        "abs",
        "acos",
        "acosh",
        "add",
        "argmax",
        "argmin",
        "asin",
        "asinh",
        "atan",
        "atanh",
        "broadcast_to",
        "cast",
        "ceil",
        "clip",
        "clip",
        "concatenate",
        "cos",
        "cosh",
        "divide",
        "equal",
        "erf",
        "exp",
        "expand_dims",
        "floor",
        "floor_divide",
        "floor_mod",
        "full",
        "image.dilation2d",
        "image.resize2d",
        "left_shift",
        "less",
        "log",
        "max",
        "maximum",
        "mean",
        "min",
        "minimum",
        "mod",
        "multiply",
        "negative",
        "nn.adaptive_avg_pool1d",
        "nn.avg_pool2d",
        "nn.avg_pool3d",
        "nn.batch_flatten",
        "nn.batch_matmul",
        "nn.bias_add",
        "nn.conv2d",
        "nn.conv1d",
        "nn.conv2d_transpose",
        "nn.conv3d",
        "nn.conv3d_transpose",
        "nn.dense",
        "nn.depth_to_space",
        "nn.fsmn",
        "nn.global_avg_pool2d",
        "nn.global_max_pool2d",
        "nn.layer_norm",
        "nn.leaky_relu",
        "nn.log_softmax",
        "nn.lrn",
        "nn.max_pool2d",
        "nn.max_pool2d_with_argmax",
        "nn.max_pool3d",
        "nn.pad",
        "nn.prelu",
        "nn.relu",
        "nn.softmax",
        "nn.space_to_depth",
        "nn.upsampling",
        "one_hot",
        "power",
        "prod",
        "reshape",
        "reverse",
        "right_shift",
        "round",
        "rsqrt",
        "scatter_nd",
        "sigmoid",
        "sign",
        "sin",
        "sinh",
        "split",
        "sqrt",
        "squeeze",
        "strided_slice",
        "subtract",
        "sum",
        "take",
        "tan",
        "tanh",
        "tile",
        "transpose",
        "vision.max_pool2d_location",
        "vision.proposal",
        "vision.psroipooling",
        "vision.roi_pool",
        "segment_max",
        "segment_mean",
        "segment_min",
        "segment_prod",
        "segment_sum",
        "vision.unpooling",
        "where",
        "qnn.quantize",
        "qnn.dequantize",
    ]
    th1520_op_list = [
        "add",
        "cast",
        "clip",
        "concatenate",
        "divide",
        "exp",
        "expand_dims",
        "image.resize2d",
        "mean",
        "multiply",
        "nn.avg_pool2d",
        "nn.batch_flatten",
        "nn.bias_add",
        "nn.conv2d",
        "nn.conv2d_transpose",
        "nn.dense",
        "nn.depth_to_space",
        "nn.global_avg_pool2d",
        "nn.global_max_pool2d",
        "nn.leaky_relu",
        "nn.lrn",
        "nn.max_pool2d",
        "nn.max_pool2d_with_argmax",
        "nn.pad",
        "nn.prelu",
        "nn.relu",
        "nn.softmax",
        "nn.upsampling",
        "minimum",
        "maximum",
        "reshape",
        "sigmoid",
        "split",
        "squeeze",
        "strided_slice",
        "subtract",
        "transpose",
        "vision.max_pool2d_location",
        "vision.proposal",
        "vision.psroipooling",
        "vision.roi_pool",
        "vision.unpooling",
        "qnn.quantize",
        "qnn.dequantize",
    ]

    qnn_op_list = [
        "qnn.csi.add",
        "qnn.csi.avgpool2d",
        "qnn.csi.concatenate",
        "qnn.csi.conv2d",
        "qnn.csi.depth_to_space",
        "qnn.csi.dense",
        "qnn.csi.minimum",
        "qnn.csi.relu6",
        "qnn.csi.relu",
        "qnn.csi.reshape",
        "qnn.csi.softmax",
    ]

    custom_op_list = [
        "cache_matmul",
        "cache_conv1d",
    ]

    op_maps = {
        "x86_ref": x86_op_list,
        "th1520": th1520_op_list,
        "e907": x86_op_list,
        "c906": x86_op_list,
        "rvm": x86_op_list,
        "c908": x86_op_list,
        "c920": x86_op_list,
        "c920v2": x86_op_list,
        "hth1520": x86_op_list,
    }

    class GetModelOps(relay.ExprVisitor):
        """Get the operation name of the input model used"""

        def __init__(self):
            super(GetModelOps, self).__init__()
            self.op_lists = []

        def visit_call(self, call):
            _ = [self.visit(arg) for arg in call.args]
            op_name = call.op.name
            if op_name not in self.op_lists:
                self.op_lists.append(op_name)

    if target not in op_maps:
        raise Exception(f'Unspported this target "{target}"')

    get_model_ops = GetModelOps()
    get_model_ops.visit(model["main"])
    model_ops = get_model_ops.op_lists
    unsupported_ops = []
    quanted_model = False
    for op_name in model_ops:
        if op_name not in op_maps[target] + qnn_op_list + custom_op_list:
            unsupported_ops.append(op_name)
        if op_name in qnn_op_list:
            quanted_model = True
    if len(unsupported_ops) > 0:
        raise Exception(f"Unspported ops {unsupported_ops} in target {target}")
    return quanted_model


def _bind_params(func, params):
    """Bind the params to the expression."""
    name_dict = {}
    for arg in func.params:
        name = arg.name_hint
        if name in name_dict:
            name_dict[name] = None
        else:
            name_dict[name] = arg
    bind_dict = {}
    for k, v in params.items():
        if k not in name_dict:
            continue
        arg = name_dict[k]
        if arg is None:
            raise ValueError("Multiple args in the function have name %s" % k)
        bind_dict[arg] = _expr.const(v)
    return _expr.bind(func, bind_dict)


def check_bn_variance(model):
    "Make sure data in variance is not negtive"

    class CheckBNVar(relay.ExprMutator):
        def visit_call(self, call):
            new_fn = self.visit(call.op)
            new_args = [self.visit(arg) for arg in call.args]
            if call.op.name == "nn.batch_norm":
                var = new_args[4].data.asnumpy()
                var[var < 0] = 0
                new_args[4] = _expr.const(var)

            return _expr.Call(new_fn, new_args, call.attrs, call.type_args, call.span)

    model["main"] = CheckBNVar().visit(model["main"])
    return model


def get_count_call(mod):
    """Get the count of call in relay ir"""

    class GetCountVisitor(relay.ExprVisitor):
        """Counting the number of call"""

        def __init__(self):
            super(GetCountVisitor, self).__init__()
            self.memo_map = {}
            self.call_count = 0

        def visit_call(self, call):
            self.call_count += 1
            _ = [self.visit(arg) for arg in call.args]

    gc = GetCountVisitor()
    gc.visit(mod["main"])
    return gc.call_count


def save_const_output(mod, output_dir):
    """Save and remove const output"""

    class save_output_in_tuple(relay.ExprMutator):
        """Save and remove const output in tuple"""

        first_visit_expr = True
        idx = 0

        def visit_call(self, call):
            self.first_visit_expr = False
            new_fn = self.visit(call.op)
            new_args = [self.visit(arg) for arg in call.args]
            return _expr.Call(new_fn, new_args, call.attrs, call.type_args, call.span)

        def visit_tuple(self, tup):
            if self.first_visit_expr:
                self.first_visit_expr = False
                new_fup = []
                for field in tup.fields:
                    if isinstance(field, _expr.Constant):
                        const_output = field.data.asnumpy()
                        const_output.tofile(
                            os.path.join(output_dir, "_const_output.{}.tensor".format(self.idx)),
                            "\n",
                        )
                        self.idx = self.idx + 1
                    else:
                        new_fup.append(self.visit(field))
                return _expr.Tuple(new_fup, tup.span)

            return _expr.Tuple([self.visit(field) for field in tup.fields], tup.span)

    mod["main"] = save_output_in_tuple().visit(mod["main"])

    return mod


def rename_constant(mod):
    """Specify name for constant node."""

    def _get_new_const(node, new_name):
        new_span = create_span(new_name)
        new_node = _expr.const(node.data.asnumpy(), dtype=node.checked_type.dtype, span=new_span)
        return new_node

    class RenameConstant(relay.ExprMutator):
        """Specify name for constant node."""

        def visit_call(self, call):
            op_args = [self.visit(arg) for arg in call.args]

            if call.op.name in csi_op().conv_handle.keys():
                weight = op_args[1]
                bias = op_args[2]
                op_args[1] = _get_new_const(weight, call.attrs.layer_name + ":weight")

                if bias and isinstance(bias, _expr.Constant):
                    op_args[2] = _get_new_const(bias, call.attrs.layer_name + ":bias")
            else:
                for idx, arg in enumerate(op_args):
                    if isinstance(arg, _expr.Constant):
                        op_args[idx] = _get_new_const(
                            arg, call.attrs.layer_name + ":const_" + str(idx)
                        )
            new_call = _expr.Call(call.op, op_args, call.attrs, call.type_args, call.span)
            return new_call

    mod["main"] = RenameConstant().visit(mod["main"])
    return mod


def detect_quantized_model(mod):
    """Check whether the model is quantitative model."""

    class InterHelper(relay.ExprVisitor):
        """Internal helper class"""

        def __init__(self):
            super(InterHelper, self).__init__()
            self.memo_map = {}
            self.quant_schema = set()

        def visit_call(self, call):
            _ = [self.visit(arg) for arg in call.args]
            if call.op.name == "qnn.quantize":
                self.quant_schema.add(call.attrs.out_dtype)

    ih = InterHelper()
    ih.visit(mod["main"])
    return ih.quant_schema


def quantize_hhb(module, params=None, curr_qconfig=None, dataset=None, target="x86_ref"):
    """The quantization procedure.

    Parameters
    ---------
    module: Module
        The original module.

    params : dict of str to NDArray
        Input parameters to the graph that do not change
        during inference time. Used for constant folding.

    dataset: list of dict of Var -> NDArray
        The calibration dataset.

    Returns
    -------
    ret: Function
        The graph after quantization
    """
    detected_quant_type = detect_quantized_model(module)

    if target in ("th1520", "hth1520") and curr_qconfig["quantization_scheme"] not in [
        "int16_sym",
        "int8_sym",
    ]:
        module = InsertNOp(module)

    if target in ("th1520", "hth1520"):
        # fix sigmoid + mul acc bug in th1520 npu
        module = InsertRelu(module)

    if params:
        module["main"] = _bind_params(module["main"], params)

    module = check_bn_variance(module)

    call_count = get_count_call(module)
    opt_seq = [
        _transform.SimplifyInference(),
        _transform.DynamicToStatic(),
        _transform.FoldConstant(),
        # _transform.FoldScaleAxis(),
        # _transform.CanonicalizeOps(),
        # _transform.FoldConstant(),
        # user-define passes
        # _transform.SpaceToBatch2AtrousConv(),
    ]
    if call_count > 1:
        opt_seq.insert(2, _transform.SimplifyExpr())
    if curr_qconfig["use_custom_fusion"]:
        logger.warning("Using custom fusion.")
        opt_seq += [FuseCacheMatMul(), FuseLayerNormal(), TConv1dAddT(), FuseCacheConv1d()]
    optimizer = transform.Sequential(opt_seq)
    logger.log(LOG, "Start optimization.")
    module = optimizer(module)
    logger.debug("Optimized model:")
    logger.debug(module["main"])
    logger.log(LOG, "Optimization completed!")
    module = save_const_output(module, os.path.dirname(curr_qconfig["params_path"]))
    logger.debug("save const output")

    quanted_model = _check_unsupported_ops(target, module)

    dtype_float = False
    if curr_qconfig["dtype_weight"] in ("float16", "bfloat16") or (
        (target not in ("th1520", "hth1520")) and curr_qconfig["dtype_weight"] == "float32"
    ):
        if not (
            target in ("c906", "rvm", "c908", "c920", "c920v2")
            and curr_qconfig["calibrate_mode"] == "scale"
        ):
            dtype_float = True

    if curr_qconfig["quantization_scheme"] == "float16_w_int8":
        dtype_float = False

    if curr_qconfig["convert_to_relay"] and quanted_model:
        convert_to_relay(module)
        quanted_model = False

    # original relay model includes quantize/dequantize nodes.
    orig_quantized_model = False
    if detected_quant_type and len(detected_quant_type) == 1:
        orig_quantized_model = True
    if dtype_float or orig_quantized_model:
        logger.log(LOG, "Start conversion to csinn.")
        if dataset:
            if orig_quantized_model:
                logger.log(LOG, "Ignore calibrate dataset in quantized model.")
            else:
                logger.log(LOG, "Ignore calibrate dataset in f16/bf16/f32 conversion.")
        module = convert_to_csi_qnn(
            module,
            None,
            curr_qconfig["channel_quantization"],
            curr_qconfig["channel_quantization_ratio_threshold"],
        )
        logger.debug("Converted model:")
        logger.debug(module["main"])
        logger.log(LOG, "Conversion completed!")
    elif dataset and not quanted_model:
        quant_params = calibration(module, dataset, curr_qconfig)
        logger.log(LOG, "Start conversion to csinn.")
        module = convert_to_csi_qnn(
            module,
            quant_params,
            curr_qconfig["channel_quantization"],
            curr_qconfig["channel_quantization_ratio_threshold"],
        )
        logger.debug("Converted model:")
        logger.debug(module["main"])
        logger.log(LOG, "Conversion completed!")
    else:
        if not quanted_model:
            raise Exception("Can't find calibration dataset!")

    logger.log(LOG, "Start operator fusion.")
    fuse_pass = [Conv2dSqueezeAdd()]
    if curr_qconfig["use_custom_fusion"]:
        logger.warning("Using custom fusion.")
        fuse_pass += [FuseWhereSoftmax(), Resume4DimsMatMul()]
    fuser = transform.Sequential(fuse_pass)
    module = fuser(module)

    # fuse quantization info
    if orig_quantized_model:
        logger.log(LOG, "Fuse quantize/dequantize nodes into ops.")
        fuse_pass = [FuseActivateQuantInfo(), FuseInputQuantInfo(), FuseDequantizeOp()]
        fuser = transform.Sequential(fuse_pass)
        module = fuser(module)
        logger.debug(module["main"])

    csi_module = fuse_layer(module, curr_qconfig)
    csi_module = QNNFuseConvDepthtospace()(csi_module)

    logger.debug("Fused model:")
    logger.debug(csi_module["main"])
    logger.log(LOG, "Operator fusion completed!")

    if orig_quantized_model:
        csi_module = unify_quant_params(csi_module)

    csi_module = optimize_quantization(
        csi_module, curr_qconfig["broadcast_quantization"], target=curr_qconfig["target"]
    )

    logger.log(LOG, "Start operator split.")
    split_pass = [_transform.InferType(), ConvSpliter(curr_qconfig)]
    spliter = transform.Sequential(split_pass)
    csi_module = spliter(csi_module)
    logger.log(LOG, "Operator split completed!")

    csi_module = relay.transform.InferType()(csi_module)

    logger.log(LOG, "Start layout convert.")
    csi_module = csi_layout_convert(
        csi_module, dest_layout=curr_qconfig["layout"], align=curr_qconfig["h_align"]
    )
    logger.log(LOG, "Layout convert completed!")

    logger.debug("Start specify name for constant node.")
    csi_module = relay.transform.InferType()(csi_module)
    csi_module = rename_constant(csi_module)
    logger.debug("specify name for constant node completed!")

    logger.debug("Start specify name for call node.")
    csi_module = rename_call(csi_module, call_count)
    logger.debug("specify name for call node completed!")

    if curr_qconfig["dump_quantization_loss"] or curr_qconfig["auto_hybrid_quantization"]:
        logger.log(LOG, "Start quantization analysis.")
        target_dir = os.path.dirname(curr_qconfig["params_path"])

        if curr_qconfig["from_quant_file"]:
            logger.log(
                LOG,
                "Get quantization loss directly from file: %s",
                os.path.join(target_dir, "model.quant.json"),
            )
        else:
            dlo = DumpLayerOutput(dataset, curr_qconfig)
            dlo.visit(csi_module["main"])

            mqi = ModelQuantizationInfo()
            mqi.update_layer_info(
                dlo.float_outs_map, dlo.qnn_outs_map, dlo.quant_info, curr_qconfig
            )

            if curr_qconfig["auto_hybrid_quantization"]:
                mqi.update_hybrid_layers(
                    curr_qconfig["quantization_loss_algorithm"],
                    curr_qconfig["quantization_loss_threshold"],
                    curr_qconfig["loss_threshold_type"],
                )
            json_data = mqi.to_dict()
            to_json(json_data, os.path.join(target_dir, "model.quant.json"))
            logger.log(
                LOG,
                "Quantization information can be found in %s",
                os.path.join(target_dir, "model.quant.json"),
            )
        logger.log(LOG, "Quantization analysis completed!")

    csi_module = relay.transform.InferType()(csi_module)
    logger.info("Quantized model:")
    logger.info(csi_module["main"])

    return csi_module


def execute_qnn_pass_with_log(log_str, func, qnn, *arg, **kwargs):
    """Wrapper for optimization pass"""
    start_log = "Start " + log_str
    logger.log(LOG, start_log)
    res = func(qnn, *arg, **kwargs)
    logger.debug("Model: %s", qnn["main"])
    end_log = "End " + log_str
    logger.log(LOG, end_log)
    return res


def optimization_phase0(module):
    """Optimization procedures in native relay"""
    opt_module = check_bn_variance(module)

    call_count = get_count_call(opt_module)
    opt_seq = [
        _transform.SimplifyInference(),
        _transform.DynamicToStatic(),
        _transform.FoldConstant(),
        # _transform.FoldScaleAxis(),
        # _transform.CanonicalizeOps(),
        # _transform.FoldConstant(),
        # user-define passes
        # _transform.SpaceToBatch2AtrousConv(),
    ]
    if call_count > 1:
        opt_seq.insert(2, _transform.SimplifyExpr())
    optimizer = transform.Sequential(opt_seq)
    opt_module = optimizer(opt_module)
    return opt_module


def optimization_phase1(module, use_custom_fusion):
    """Optimization procedures for transformer in relay."""

    opt_seq = [FuseLayerNormal()]
    if use_custom_fusion:
        opt_seq += [FuseCacheMatMul(), TConv1dAddT(), FuseCacheConv1d()]
    optimizer = transform.Sequential(opt_seq)
    opt_module = optimizer(module)
    return opt_module


def optimization_th1520(module):
    """Optimization procedures for th1520 in qnn."""
    opt_seq = [
        QNNTh1520InsertReluBetweenSigmoidAndMul(),
        QNNTh1520InsertAddBetweenLeakyReluAndAdd(),
    ]
    optimizer = transform.Sequential(opt_seq)
    opt_module = optimizer(module)
    return opt_module


def get_quantized_model(module, params=None, curr_config=None, target="x86_ref"):
    """Convert quantized model into qnn ir.

    Parameters
    ---------
    module: Module
        The original module.

    params : dict of str to NDArray
        Input parameters to the graph that do not change
        during inference time. Used for constant folding.

    target : str
        Target platform.

    Returns
    -------
    ret: Function
        The graph after quantization
    """
    if params:
        module["main"] = _bind_params(module["main"], params)

    module = execute_qnn_pass_with_log("optimization for native relay", optimization_phase0, module)

    module = execute_qnn_pass_with_log(
        "optimization for transformer ops",
        optimization_phase1,
        module,
        curr_config["use_custom_fusion"],
    )

    module = execute_qnn_pass_with_log(
        "save const output", save_const_output, module, os.path.dirname(curr_config["params_path"])
    )
    _ = _check_unsupported_ops(target, module)

    qnn_module = execute_qnn_pass_with_log(
        "conversion to csinn",
        convert_to_csi_qnn,
        module,
        None,
        curr_config["channel_quantization"],
        curr_config["channel_quantization_ratio_threshold"],
    )

    qdq_fuse_pass = transform.Sequential([QNNSeparateRepeatedQDQ(), QNNFuseQDQ(curr_config)])
    qnn_module = execute_qnn_pass_with_log(
        "fuse quantize/dequantize nodes into ops", qdq_fuse_pass, qnn_module
    )

    qnn_module = execute_qnn_pass_with_log(
        "graph fusion for qnn", fuse_layer, qnn_module, curr_config
    )

    if target in ("th1520", "hth1520"):
        # fix acc bug in th1520 npu
        qnn_module = execute_qnn_pass_with_log(
            "optimize for th1520", optimization_th1520, qnn_module
        )

    # final check
    qnn_module = QNNCheckValidQuantParams(board=target)(qnn_module)

    if logger.level <= logging.DEBUG:
        json_file = os.path.dirname(curr_config["params_path"])
        json_file = os.path.join(json_file, "model_qnn.json")
        qnn_module = QNNDumpToJson(json_file)(qnn_module)
        logger.debug("save qnn ir structure into %s", json_file)

    qnn_module = relay.transform.InferType()(qnn_module)
    logger.info("Final qnn model:")
    logger.info(qnn_module["main"])

    return qnn_module
