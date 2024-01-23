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
# pylint: disable=too-many-nested-blocks, no-else-continue, unused-argument
"""Automatic hybrid quantization toolkit."""
import os
import copy
import collections
import json
import re
from typing import List
import numpy as np
from scipy import stats

import tvm
from tvm import relay
from tvm.relay.frontend.common import infer_shape
from tvm.relay.op.contrib import csinn
from tvm.contrib import graph_executor
from tvm.relay.expr import Var, Call, TupleGetItem, Constant, Tuple
from tvm.ir import transform, IRModule
from tvm.relay.backend.contrib.csinn_backend import collect_quant_info

from .calibrate import get_out_params, get_weight_params


def to_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def prob_distribution(data):
    abs_data = np.abs(data)
    return abs_data / (np.sum(abs_data) + 1e-5)


class LossBase(object):
    """Base loss function define

    Parameters
    ----------
    per_tensor: Optional[bool]
        Whether calculate loss with per-tensor or per-channel

    axis: Optional[int]
        If per_tensor is False, we should specify axis for per-channel computation.
    """

    def __init__(self, per_tensor=True, axis=0):
        self.per_tensor = per_tensor
        self.axis = axis

        self.avg_values = []
        self.min_values = []
        self.max_values = []

        self._data = []

    def name(self):
        return "loss_base"

    def to_dict(self):
        """Convert object to dict"""
        res = collections.OrderedDict()
        res["avg_values"] = self.avg_values
        res["min_values"] = self.min_values
        res["max_values"] = self.max_values
        return res

    def loss_function(self, lhs, rhs=None):
        """loss implemention"""
        raise NotImplementedError("loss_function is not implemented.")

    def cal_loss(self, lhs, rhs=None):
        """Batch computation."""
        if lhs is not None and not isinstance(lhs, np.ndarray):
            raise ValueError("Need numpy array.")
        if rhs is not None and not isinstance(rhs, np.ndarray):
            raise ValueError("Need numpy array.")

        if self.per_tensor:
            self._data.append([self.loss_function(lhs, rhs)])
        else:  # per-channel
            shape = lhs.shape
            split_lhs = np.split(lhs, shape[self.axis], self.axis)
            split_rhs = None
            if rhs is not None:
                split_rhs = np.split(rhs, shape[self.axis], self.axis)
            all_loss = []
            for idx, d in enumerate(split_lhs):
                if split_rhs is None:
                    curr_loss = self.loss_function(d)
                else:
                    curr_loss = self.loss_function(d, split_rhs[idx])
                all_loss.append(curr_loss)
            self._data.append(all_loss)

    def update_results(self):
        """Get statistic data."""
        if len(self._data) == 0:
            return
        data = np.array(self._data, dtype=np.float32)
        assert len(data.shape) == 2, "The dim of data is not 2."

        self.min_values = np.min(data, axis=0).tolist()
        self.max_values = np.max(data, axis=0).tolist()
        self.avg_values = np.mean(data, axis=0).tolist()


class GiniLoss(LossBase):
    """Gini loss, refer to: https://neuroplausible.com/gini"""

    def name(self):
        return "gini"

    def loss_function(self, lhs, rhs=None):
        assert lhs is not None, "lhs is None."
        data = lhs.flatten()
        data = np.abs(data)
        data[data < 1e-4] = 0
        nonzero = np.nonzero(data)[0]
        data = data[nonzero]
        data = np.sort(data)
        index = np.arange(1, data.shape[0] + 1)
        n = data.size
        gini = float(((np.sum((2 * index - n - 1) * data)) / (n * np.sum(data) + 1e-5)))
        return gini


class CosineSimilarity(LossBase):
    """Cosine loss"""

    def name(self):
        return "cos_similarity"

    def loss_function(self, lhs, rhs=None):
        assert lhs is not None, "lhs is None."
        assert rhs is not None, "rhs is None."

        lhs_flatten = lhs.flatten()
        rhs_flatten = rhs.flatten()
        return np.dot(lhs_flatten, rhs_flatten) / (
            np.linalg.norm(lhs_flatten) * (np.linalg.norm(rhs_flatten)) + 1e-5
        )


class MSELoss(LossBase):
    """Mean squared error."""

    def name(self):
        return "mse"

    def loss_function(self, lhs, rhs=None):
        assert lhs is not None, "lhs is None."
        assert rhs is not None, "rhs is None."
        res = np.abs(lhs - rhs)
        res = np.square(res)
        res = np.mean(res)
        return res


class CrossEntropy(LossBase):
    """Cross entropy loss"""

    def name(self):
        return "cross_entropy"

    def loss_function(self, lhs, rhs=None):
        assert lhs is not None, "lhs is None."
        assert rhs is not None, "rhs is None."

        lhs_p = prob_distribution(lhs.flatten())
        rhs_p = prob_distribution(rhs.flatten())

        lhs_p[lhs_p < 1e-9] = 1e-9
        rhs_p[rhs_p < 1e-9] = 1e-9

        return np.sum(-lhs_p * np.log2(rhs_p))


class KLDivergence(LossBase):
    """Kullback–Leibler divergence loss"""

    def name(self):
        return "kl_divergence"

    def loss_function(self, lhs, rhs=None):
        assert lhs is not None, "lhs is None."
        assert rhs is not None, "rhs is None."

        lhs_p = prob_distribution(lhs.flatten())
        rhs_p = prob_distribution(rhs.flatten())

        lhs_p[lhs_p < 1e-9] = 1e-9
        rhs_p[rhs_p < 1e-9] = 1e-9

        return stats.entropy(lhs_p, rhs_p)


class QuantParams(object):
    """Hold quantizaiton parameters."""

    def __init__(self) -> None:
        self.min_values = []
        self.max_values = []
        self.scales = []
        self.zero_points = []
        self.origin_min_values = []
        self.origin_max_values = []

    def to_dict(self):
        res = collections.OrderedDict()
        res["min_values"] = self.min_values
        res["max_values"] = self.max_values
        res["scales"] = self.scales
        res["zero_points"] = self.zero_points
        res["origin_min_values"] = self.origin_min_values
        res["origin_max_values"] = self.origin_max_values
        return res


class LossInfo(object):
    """Holds all loss"""

    def __init__(self, per_tensor=True, axis=0) -> None:
        self.per_tensor = per_tensor
        self.axis = axis

        self._all_loss = collections.OrderedDict()

    def initialize(self):
        """Create all loss object."""
        self.gini = GiniLoss(self.per_tensor, self.axis)
        self._all_loss[self.gini.name()] = self.gini

        self.cos_sim = CosineSimilarity(self.per_tensor, self.axis)
        self._all_loss[self.cos_sim.name()] = self.cos_sim

        self.mse = MSELoss(self.per_tensor, self.axis)
        self._all_loss[self.mse.name()] = self.mse

        self.cross_entropy = CrossEntropy(self.per_tensor, self.axis)
        self._all_loss[self.cross_entropy.name()] = self.cross_entropy

        self.kl = KLDivergence(self.per_tensor, self.axis)
        self._all_loss[self.kl.name()] = self.kl

    def to_dict(self):
        res = collections.OrderedDict()
        res[self.gini.name()] = self.gini.to_dict()
        res[self.cos_sim.name()] = self.cos_sim.to_dict()
        res[self.mse.name()] = self.mse.to_dict()
        res[self.cross_entropy.name()] = self.cross_entropy.to_dict()
        res[self.kl.name()] = self.kl.to_dict()
        return res

    def cal_loss(self, lhs, rhs):
        self.gini.cal_loss(lhs)
        self.cos_sim.cal_loss(lhs, rhs)
        self.mse.cal_loss(lhs, rhs)
        self.cross_entropy.cal_loss(lhs, rhs)
        self.kl.cal_loss(lhs, rhs)

    def update_results(self):
        self.gini.update_results()
        self.cos_sim.update_results()
        self.mse.update_results()
        self.cross_entropy.update_results()
        self.kl.update_results()

    def get_loss_by_name(self, name):
        return self._all_loss[name]


class LayerQuantizatonInfo(object):
    """Quantization information for every tensor."""

    def __init__(self, name, tensor_type):
        self.name = name
        self.tensor_type = tensor_type
        self.quantization_schema = "uint8_asym"
        self.quantization_algorithm = "minmax"
        self.quantization_parameters = QuantParams()
        self.float_result_path = None
        self.quant_result_path = None
        self.loss = LossInfo()

    def to_dict(self):
        """convert obj into dict"""
        res = collections.OrderedDict()
        res["name"] = self.name
        res["tensor_type"] = self.tensor_type
        res["quantization_schema"] = self.quantization_schema
        res["quantization_algorithm"] = self.quantization_algorithm
        if self.float_result_path:
            res["float_result_path"] = self.float_result_path
        if self.quant_result_path:
            res["quant_result_path"] = self.quant_result_path
        res["quantization_parameters"] = self.quantization_parameters.to_dict()
        res["loss"] = self.loss.to_dict()
        return res


class LayersForSelectedLoss(object):
    """Helper class for specifying loss for layer."""

    def __init__(self, name, tensor_type, loss):
        self.name = name
        self.tensor_type = tensor_type
        self.loss = loss

    def to_dict(self):
        res = collections.OrderedDict()
        res["name"] = self.name
        res["tensor_type"] = self.tensor_type
        res["loss"] = self.loss
        return res

    def __eq__(self, other):
        return self.loss == other.loss

    def __gt__(self, other):
        return self.loss > other.loss

    def __le__(self, other):
        return self.loss < other.loss


class HybridQuantizationInfo(object):
    """Quantization information for hybrid tensor."""

    def __init__(self, name="cos_similarity", threshold=0.99):
        self.loss_name = name
        self.loss_threshold = threshold
        self.sorted_layers: List[LayersForSelectedLoss] = []

    def to_dict(self):
        res = collections.OrderedDict()
        res["loss_name"] = self.loss_name
        res["loss_threshold"] = self.loss_threshold
        res["sorted_layers"] = []
        for sl in self.sorted_layers:
            res["sorted_layers"].append(sl.to_dict())
        return res

    def from_dict(self, data):
        self.loss_name = data["loss_name"]
        self.loss_threshold = data["loss_threshold"]
        for d in data["sorted_layers"]:
            curr_lfsl = LayersForSelectedLoss(**d)
            self.sorted_layers.append(curr_lfsl)

    def get_hybrid_layers(self):
        """Get layer name according to threshold."""
        layer_name = []
        for sl in self.sorted_layers:
            if sl.tensor_type == "const":
                continue

            curr_name = sl.name
            if ":out" in curr_name:
                curr_name = curr_name.split(":out")[0]
            if self.loss_name == "cos_similarity":
                if sl.loss < self.loss_threshold:
                    layer_name.append(curr_name)
            else:
                if sl.loss > self.loss_threshold:
                    layer_name.append(curr_name)
        return layer_name


class ModelQuantizationInfo(object):
    """Quantizaiton for the whole model."""

    def __init__(self):
        self.source = "hhb"
        self.trace_type = "acc"
        self.version = "2.0"
        self.calibration_dataset_num = 1
        self.layers_info: List[LayerQuantizatonInfo] = []
        self.hybrid_layers = None

    def update_layer_info(self, float_outs, qnn_outs, quant_info, config):
        """Get loss information according to dump data."""
        self.calibration_dataset_num = len(list(float_outs.values())[0])
        for layer, value in qnn_outs.items():
            tensor_type = "activate"
            if isinstance(layer, (Tuple, TupleGetItem)):
                continue
            elif isinstance(layer, Call):
                layer_name = str(layer.attrs.layer_name)
            elif isinstance(layer, Constant):
                layer_name = str(layer.span.source_name.name)
                tensor_type = "const"
            elif isinstance(layer, Var):
                layer_name = layer.name_hint
            else:
                pass

            outs_quant_info = quant_info[layer_name] if layer_name in quant_info else []

            out_num = len(value[0])
            for o_idx in range(out_num):
                curr_quant_info = outs_quant_info[o_idx] if outs_quant_info else []
                trans_quant_info = np.array(curr_quant_info, dtype=np.float32)
                trans_quant_info = np.transpose(trans_quant_info, (1, 0))

                if tensor_type == "activate":
                    curr_name = layer_name + ":out" + str(o_idx)
                else:
                    curr_name = layer_name
                layer_quant = LayerQuantizatonInfo(curr_name, tensor_type)
                layer_quant.quantization_schema = config["quantization_scheme"]
                layer_quant.quantization_algorithm = config["calibrate_mode"]

                layer_quant.quantization_parameters.min_values = trans_quant_info[0].tolist()
                layer_quant.quantization_parameters.max_values = trans_quant_info[1].tolist()
                layer_quant.quantization_parameters.scales = trans_quant_info[2].tolist()
                layer_quant.quantization_parameters.zero_points = trans_quant_info[3].tolist()

                layer_float_data = np.array(float_outs[layer], dtype=np.float32)
                layer_float_data = layer_float_data[:, o_idx]
                if config["channel_quantization"]:
                    layer_quant.quantization_parameters.origin_min_values = [
                        np.min(layer_float_data).tolist()
                    ]
                    layer_quant.quantization_parameters.origin_max_values = [
                        np.max(layer_float_data).tolist()
                    ]

                else:
                    layer_quant.quantization_parameters.origin_min_values = [
                        np.min(layer_float_data).tolist()
                    ]
                    layer_quant.quantization_parameters.origin_max_values = [
                        np.max(layer_float_data).tolist()
                    ]
                layer_quant.loss.initialize()

                for c_idx in range(self.calibration_dataset_num):
                    curr_float_value = float_outs[layer][c_idx][o_idx]
                    curr_qnn_value = value[c_idx][o_idx]

                    if config["debug_level"] == "INFO":
                        dump_dir = os.path.join(os.path.dirname(config["params_path"]), "dump")
                        if not os.path.exists(dump_dir):
                            os.makedirs(dump_dir)
                        float_path = os.path.join(
                            dump_dir,
                            str(c_idx)
                            + "_"
                            + re.sub(r"[/:\s\.]", "_", curr_name)
                            + "_float.tensor",
                        )
                        layer_quant.float_result_path = float_path
                        curr_float_value.tofile(
                            float_path,
                            "\n",
                        )

                        quant_path = os.path.join(
                            dump_dir,
                            str(c_idx) + "_" + re.sub(r"[/:\s\.]", "_", curr_name) + "_qnn.tensor",
                        )
                        layer_quant.quant_result_path = quant_path
                        curr_qnn_value.tofile(
                            quant_path,
                            "\n",
                        )

                    layer_quant.loss.cal_loss(curr_float_value, curr_qnn_value)
                    if tensor_type == "const":
                        break
                layer_quant.loss.update_results()
                self.layers_info.append(layer_quant)

    def update_hybrid_layers(self, name="cos_similarity", threshold=0.99, threshold_type="avg"):
        """Generate hybrid quantization layer."""
        self.hybrid_layers = HybridQuantizationInfo(name, threshold)
        for li in self.layers_info:
            curr_loss = li.loss.get_loss_by_name(name)
            # FIXME(@chenf): mean values of average value of loss for current loss.
            if threshold_type == "max":
                loss_value = np.mean(curr_loss.max_values).tolist()
            elif threshold_type == "min":
                loss_value = np.mean(curr_loss.min_values).tolist()
            else:
                loss_value = np.mean(curr_loss.avg_values).tolist()

            hl = LayersForSelectedLoss(li.name, li.tensor_type, loss_value)
            self.hybrid_layers.sorted_layers.append(hl)

            if name == "cos_similarity":
                self.hybrid_layers.sorted_layers.sort()
            else:
                self.hybrid_layers.sorted_layers.sort(reverse=True)

    def to_dict(self):
        """Convert to dict"""
        res = collections.OrderedDict()
        res["source"] = self.source
        res["trace_type"] = self.trace_type
        res["version"] = self.version
        res["calibration_dataset_num"] = self.calibration_dataset_num
        res["layers_info"] = []
        for li in self.layers_info:
            res["layers_info"].append(li.to_dict())

        res["hybrid_layers"] = {}
        if self.hybrid_layers is not None:
            res["hybrid_layers"] = self.hybrid_layers.to_dict()
        return res


def get_float_config():
    """Config for float32 inference"""
    config = {}
    config["nbit_input"] = 32
    config["nbit_weight"] = 32
    config["nbit_activation"] = 32
    config["dtype_input"] = "float32"
    config["dtype_weight"] = "float32"
    config["dtype_activation"] = "float32"
    return config


def get_quant_config(origin_config):
    """Conifg for quant inference"""
    config = {}
    if origin_config["dtype_weight"] == "float32":
        if origin_config["quantization_scheme"] == "int8_sym":
            config["nbit_input"] = 8
            config["nbit_weight"] = 8
            config["nbit_activation"] = 32
            config["dtype_input"] = "int8"
            config["dtype_weight"] = "int8"
            config["dtype_activation"] = "int32"
        elif origin_config["quantization_scheme"] == "int16_sym":
            config["nbit_input"] = 16
            config["nbit_weight"] = 16
            config["nbit_activation"] = 32
            config["dtype_input"] = "int16"
            config["dtype_weight"] = "int16"
            config["dtype_activation"] = "int32"
    return config


def load_lib(module_factory, output_dir):
    """load built lib"""
    contrib_dir = os.path.dirname(os.path.realpath(os.path.expanduser(__file__)))
    contrib_dir = os.path.realpath(os.path.join(contrib_dir, "..", "..", "..", ".."))
    source_dir = os.path.join(contrib_dir, "..")
    include_path0 = os.path.join(source_dir, "install_nn2", "x86", "include", "csinn")
    include_path00 = os.path.join(source_dir, "install_nn2", "x86", "include", "shl_public")
    include_path1 = os.path.join(contrib_dir, "hhb", "install_nn2", "x86", "include", "csinn")
    include_path11 = os.path.join(contrib_dir, "hhb", "install_nn2", "x86", "include", "shl_public")
    ref_x86_dir0 = os.path.join(source_dir, "install_nn2", "x86", "lib")  # for source
    ref_x86_dir1 = os.path.join(
        contrib_dir, "hhb", "install_nn2", "x86", "lib"
    )  # for package binary
    lib_path = os.path.join(output_dir, "quant.so")
    kwargs = {}
    kwargs["options"] = [
        "-O2",
        "-g",
        "-I" + include_path0,
        "-I" + include_path00,
        "-I" + include_path1,
        "-I" + include_path11,
        "-L" + ref_x86_dir0,
        "-L" + ref_x86_dir1,
        "-lshl",
    ]
    kwargs["cc"] = "gcc"
    lib = module_factory.get_lib()
    lib.export_library(lib_path, fcompile=False, workspace_dir=output_dir, **kwargs)
    lib = tvm.runtime.load_module(lib_path)
    return lib


def build_mod(mod, config=None):
    """Build module"""
    func = mod["main"]
    func = func.with_attr("global_symbol", tvm.runtime.container.String("csinn"))
    func = func.with_attr("Compiler", "csinn")
    mod["csinn_0"] = func

    output_dir = "."
    if config is not None:
        with tvm.transform.PassContext(opt_level=3, config={"relay.ext.csinn.options": config}):
            csinn_mod = csinn.partition_for_csinn(mod)
            factory = relay.build(csinn_mod, target="c")

            output_dir = os.path.dirname(config["params_path"])
    else:
        csinn_mod = csinn.partition_for_csinn(mod)
        factory = relay.build(csinn_mod, target="c")

        curr_options = transform.PassContext.current().config["relay.ext.csinn.options"]
        output_dir = os.path.dirname(str(getattr(curr_options, "params_path")))

    tvm_q = collect_quant_info()
    python_q = {}

    def _convert_tvm(tvm_obj):
        res = []
        for v in tvm_obj:
            if isinstance(v, tvm.tir.expr.FloatImm):
                # avoid accuracy loss
                res.append(float(str(np.float32(v))))
            else:
                res.append(_convert_tvm(v))
        return res

    for k, v in tvm_q.items():
        python_q[str(k)] = _convert_tvm(v)

    return factory, output_dir, python_q


def qnn_inference(factory, dataset, output_dir):
    """Inference qnn model."""
    ctx = tvm.cpu(0)
    lib = load_lib(factory, output_dir)
    m = graph_executor.create(factory.get_graph_json(), lib, ctx)
    m.load_params(tvm.runtime.save_param_dict(factory.get_params()))

    res = []
    for d in dataset:
        m.run(**d)
        tmp = []
        for i in range(m.get_num_outputs()):
            output = m.get_output(i).asnumpy()
            tmp.append(output)
        res.append(tmp)
    return res


def _set_tuple_getitem_out(t, outs_map):
    assert t.tuple_value in outs_map, "tuple getitem not find input."
    pre_call_outs = outs_map[t.tuple_value]

    outs_map[t] = []
    for data in pre_call_outs:
        outs_map[t].append([data[t.index]])


def inference_constant_node(dataset, layer_name, quant_config, tensor_type="activate"):
    """Inference constant node for qnn."""
    input_name = "input"

    call_data = []
    if tensor_type == "activate":
        new_data = [d[0] for d in dataset]
        quant_params = get_out_params(
            new_data,
            quant_config["calibrate_mode"],
            quant_config["activate_quantized_type"],
        )
        shape = dataset[0][0].shape
        dtype = dataset[0][0].dtype.name

        for b in dataset:
            call_data.append({input_name: b[0]})
    else:
        quant_params = get_weight_params(dataset, quant_config)
        shape = dataset.shape
        dtype = dataset.dtype.name

        call_data.append({input_name: dataset})
    node = relay.var(input_name, shape=shape, dtype=dtype)
    node = relay.qnn.op.csi_reshape(
        node,
        shape,
        out_dtype="float32",
        q_params=[quant_params, quant_params],
        layer_name=layer_name,
    )

    nmod = IRModule.from_expr(node)

    curr_config = copy.deepcopy(quant_config)
    curr_config.update(get_quant_config(curr_config))
    factory, output_dir, quant_info = build_mod(nmod, curr_config)
    res = qnn_inference(factory, call_data, output_dir)
    return res, quant_info


class DumpLayerOutput(relay.expr_functor.ExprVisitor):
    """Dump every layer in qnn module."""

    def __init__(self, dataset, config):
        super(DumpLayerOutput, self).__init__()

        self.config = config
        self.config["target"] = "x86_ref"
        self.config["hybrid_layer_name"] = []
        self.config["trace"] = []
        self.dataset = dataset
        self.input_group_num = len(self.dataset)

        # outs_map map call => list of multiple batch output of current call
        # outs_map = {call: [[], ...]}
        self.float_outs_map = collections.OrderedDict()
        self.qnn_outs_map = collections.OrderedDict()

        # {layer_name:
        #   [ # multiple outputs
        #       [ # multiple channel
        #           [min, max, scale, zero_point]
        #       ]
        #   ]
        # }
        self.quant_info = collections.OrderedDict()

    def visit_var(self, var):
        self.float_outs_map[var] = []
        self.qnn_outs_map[var] = []
        for in_data in self.dataset:
            data = in_data[var.name_hint]

            self.float_outs_map[var].append([data])

        self.qnn_outs_map[var], var_quant_info = inference_constant_node(
            self.float_outs_map[var], var.name_hint, self.config, "activate"
        )

        self.quant_info.update(var_quant_info)

    def visit_tuple_getitem(self, t):
        self.visit(t.tuple_value)
        _set_tuple_getitem_out(t, self.float_outs_map)
        _set_tuple_getitem_out(t, self.qnn_outs_map)

    def visit_tuple(self, tup):
        self.float_outs_map[tup] = []
        self.qnn_outs_map[tup] = []
        for _ in range(self.input_group_num):
            self.float_outs_map[tup].append([])
            self.qnn_outs_map[tup].append([])
        for f in tup.fields:
            self.visit(f)
            assert f in self.float_outs_map, "tuple not find input."
            assert f in self.qnn_outs_map, "tuple not find input."
            for i in range(self.input_group_num):
                self.float_outs_map[tup][i].extend(self.float_outs_map[f][i])
                self.qnn_outs_map[tup][i].extend(self.qnn_outs_map[f][i])

    def visit_constant(self, const):
        data = const.data.asnumpy()
        self.float_outs_map[const] = [[data]]
        self.qnn_outs_map[const], const_quant_info = inference_constant_node(
            data, const.span.source_name.name, self.config, "const"
        )

        self.quant_info.update(const_quant_info)

    def visit_call(self, call):
        _ = [self.visit(arg) for arg in call.args]

        nargs = []

        call_dataset = []
        for _ in range(self.input_group_num):
            call_dataset.append({})
        for i, arg in enumerate(call.args):
            arg_name = "var_" + str(i)
            if isinstance(arg, Constant):
                nargs.append(arg)
            elif isinstance(arg, (Call, TupleGetItem, Var)):
                i_data = self.float_outs_map[arg]
                # for i_d in i_data:
                for g in range(self.input_group_num):
                    call_dataset[g][arg_name] = i_data[g][0]
                nargs.append(
                    relay.var(arg_name, shape=i_data[0][0].shape, dtype=i_data[0][0].dtype.name)
                )
            elif isinstance(arg, Tuple):
                tmp_args = []
                for j in range(len(arg)):
                    if arg.fields[j] in self.float_outs_map:
                        t_i_data = self.float_outs_map[arg.fields[j]]
                        t_arg_name = arg_name + "_" + str(j)
                        for g in range(self.input_group_num):
                            call_dataset[g][t_arg_name] = t_i_data[g][0]
                        inter_node = relay.var(
                            t_arg_name, shape=t_i_data[0][0].shape, dtype=t_i_data[0][0].dtype.name
                        )
                        if call.op.name == "qnn.csi.concatenate":
                            reshape_q = call.attrs.q_params[j]
                            reshape_q = [q.value for q in reshape_q]
                            reshape_q = [reshape_q, reshape_q]
                            inter_node = relay.qnn.op.csi_reshape(
                                inter_node,
                                t_i_data[0][0].shape,
                                out_dtype="float32",
                                q_params=reshape_q,
                                layer_name=str(call.attrs.layer_name) + "_reshape" + str(j),
                            )
                        tmp_args.append(inter_node)
                    elif isinstance(arg.fields[j], Constant):
                        tmp_args.append(arg.fields[j])
                    else:
                        raise Exception("can not find input data.")
                nargs.append(Tuple(tmp_args))

        ncall = Call(call.op, nargs, call.attrs)
        if ncall.op.name == "qnn.csi.split":
            split_out = []
            split_out_shape = infer_shape(ncall)
            for idx in range(len(split_out_shape.fields)):
                split_q = call.attrs.q_params[idx + 1]
                split_q = [q.value for q in split_q]
                split_q = [split_q, split_q]
                tmp_item = TupleGetItem(ncall, idx)
                split_out.append(
                    relay.qnn.op.csi_reshape(
                        tmp_item,
                        split_out_shape.fields[idx].concrete_shape,
                        out_dtype="float32",
                        q_params=split_q,
                        layer_name=str(call.attrs.layer_name) + "_reshape" + str(idx),
                    )
                )
            ncall = Tuple(split_out)
        nmod = IRModule.from_expr(ncall)

        # dump layers with quantized dtype
        curr_config = copy.deepcopy(self.config)
        curr_config.update(get_quant_config(curr_config))
        factory, output_dir, call_quant_info = build_mod(nmod, curr_config)
        self.quant_info.update(call_quant_info)
        self.qnn_outs_map[call] = qnn_inference(factory, call_dataset, output_dir)

        # dump layers with float dtype
        curr_config = copy.deepcopy(self.config)
        curr_config.update(get_float_config())

        factory, output_dir, _ = build_mod(nmod, curr_config)
        self.float_outs_map[call] = qnn_inference(factory, call_dataset, output_dir)


def auto_hybrid_quantize(module, dataset, quant_config, output_dir="."):
    """
    main steps:
        1. dump layer output values with float32 and quantized type separately;
        2. calculate loss(float32 vs quantized type);
        3. select layers that will be quantized with hybrid method.
    """
    dlo = DumpLayerOutput(dataset, quant_config)
    dlo.visit(module["main"])

    mqi = ModelQuantizationInfo()
    mqi.update_layer_info(dlo.float_outs_map, dlo.qnn_outs_map, dlo.quant_info, quant_config)
    mqi.update_hybrid_layers(
        quant_config["quantization_loss_algorithm"], quant_config["quantization_loss_threshold"]
    )
    json_data = mqi.to_dict()
    to_json(json_data, os.path.join(output_dir, "model.quant.json"))

    return module
