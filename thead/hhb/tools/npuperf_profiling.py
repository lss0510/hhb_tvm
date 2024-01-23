# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License") you may not use this file except in compliance
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
"""
Profiling tools for ASP with NPU Performance Modeling.
"""
import os
import collections
import time
from typing import List
import argparse

from tvm.relay.quantize.qnn_transform import QNNConvertDict
from ..core.common import to_json, HHBException
from ..core.profiler_manage import ArchConfigBase
from hhb.analysis.trace import HHBTrace, HHBTraceEventComplete, HHBTraceEventCategory

try:
    import npuperf
except ImportError:
    raise HHBException("Current task depends on npuperf tool, please install it first.")

from npuperf.classes.workload.json_parser.json2workload import Json2WorkloadParser
from npuperf.classes.cost_model.cost_model import CostModelEvaluation
from npuperf.classes.stages import (
    HardwareGeneratorStage,
    HardwareModifierStage,
    MappingGeneratorStage,
    JsonWorkloadParserStage,
    WorkloadParserStage,
    SumAndSaveAllLayersStage,
    WorkloadStage,
    SpatialMappingConversionStage,
    MinimalLatencyStage,
    LomaStage,
    ZigZagCostModelStage,
    PickleSaveStage,
    PlotTemporalMappingsStage,
    MainStage,
    SimpleSaveStage,
    StationaryLomaStage,
)


class NpuperfArchConfig(ArchConfigBase):
    def __init__(self, file) -> None:
        super().__init__(file)
        # arch name
        self.name = None

        # temporal data flow
        self.flow = None
        self.hardware_level = "default"

        # arch config level 1:
        #   combine arch name and the following parameters
        self.gb_size = None
        self.gb_bw = None
        self.dram_bw = None

        # arch config level 2:
        self.detailed_config = None

    def initialize(self):
        if not self.data:
            return
        self.name = self.data.get("name", self.name)
        self.flow = self.data.get("flow", self.flow)
        self.hardware_level = self.data.get("hardware_level", self.hardware_level)

        hardware = self.data.get("hardware", None)
        if hardware:
            level1 = hardware.get("level1", None)
            if level1:
                self.gb_size = level1.get("gb_size", self.gb_size)
                self.gb_bw = level1.get("gb_bw", self.gb_bw)
                self.dram_bw = level1.get("dram_bw", self.dram_bw)

            self.detailed_config = hardware.get("level2", self.detailed_config)

    def check(self):
        all_hw = get_all_HW_name()
        if self.hardware_level in ("default", "level1") and self.name not in all_hw:
            raise HHBException(
                f"{self.name} is not built-in HW name, please select from {all_hw} or reset 'hardware_level': 'level2' and customize hardware."
            )


def convert_qnn_ir_to_npm_input(qnn_ir, target_layout="NCHW"):
    """convert qnn ir for npuperf"""

    def _modify_tensor_name(data: collections.OrderedDict):
        if not data or "layers" not in data.keys():
            return data
        data_cp = data

        tensor_idx = 0
        for layer in data_cp["layers"]:
            for o in layer["outputs"]:
                o["name"] = "tensor_" + str(tensor_idx)
                tensor_idx += 1

            # update other inputs
            for l in data_cp["layers"]:
                for i in l["inputs"]:
                    if i["hash_value"] == layer["hash_value"]:
                        o = layer["outputs"][i["index"]]
                        i["name"] = o["name"]

        const_idx = 0
        for layer in data_cp["layers"]:
            for i in layer["inputs"]:
                if "data" in i:
                    i["name"] = "const_" + str(const_idx)
                    const_idx += 1

        return data_cp

    def _filter_unused_values(data: collections.OrderedDict):
        if not data or "layers" not in data.keys():
            return data
        data_cp = data
        for layer in data_cp["layers"]:
            if "inputs" not in layer:
                continue
            for i in layer["inputs"]:
                if "q_param" in i:
                    i.pop("q_param")
                if "data" in i:
                    i.pop("data")
                    i["is_const"] = 1
                else:
                    i["is_const"] = 0

                if "hash_value" in i:
                    i.pop("hash_value")
                if "index" in i:
                    i.pop("index")

            if "hash_value" in layer:
                layer.pop("hash_value")

            # attrs
            if "out_dtype" in layer["attrs"]:
                layer["attrs"].pop("out_dtype")
            if not layer["attrs"]:
                layer.pop("attrs")
        return data_cp

    def _is_depthwise_conv(in_shape, kernel_shape, group, layout):
        res = False
        if layout == "NCHW" and kernel_shape[1] == 1 and group == in_shape[1]:
            res = True
        elif layout == "NHWC" and kernel_shape[0] == 1 and group == in_shape[3]:
            res = True
        return res

    def _get_tensor_act_layout(shape, layout="NCHW"):
        if len(shape) == 1 or len(shape) == 0:
            return "N"
        elif len(shape) == 2:
            return "NC"
        elif len(shape) == 3:
            if layout == "NCHW":
                return "NCW"
            if layout == "NHWC":
                return "NWC"
        elif len(shape) == 4:
            if layout == "NCHW":
                return "NCHW"
            if layout == "NHWC":
                return "NHWC"
        elif len(shape) == 5:
            if layout == "NCHW":
                return "NCDHW"
            if layout == "NHWC":
                return "NDHWC"
        elif len(shape) == 6:
            if layout == "NCHW":
                return "NLCDHW"
        else:
            raise ValueError(f"Unsupported shape size: {len(shape)}")

    def _get_tensor_weight_layout(shape, layout="NCHW"):
        if len(shape) == 0:
            return "NULL"
        elif len(shape) == 1:
            return "O"
        elif len(shape) == 2:
            return "OI"
        elif len(shape) == 3:
            if layout == "NCHW":
                return "OIW"
            if layout == "NHWC":
                return "OWI"
        elif len(shape) == 4:
            if layout == "NCHW":
                return "OIHW"
            if layout == "NHWC":
                return "OHWI"
        elif len(shape) == 5:
            if layout == "NCHW":
                return "OIDHW"
            if layout == "NHWC":
                return "ODHWI"
        else:
            raise ValueError(f"Unsupported shape size: {len(shape)}")

    def _modify_values(data: collections.OrderedDict):
        if not data or "layers" not in data.keys():
            return data
        data_cp = data
        for layer in data_cp["layers"]:
            is_depthwise_conv2d = False

            for i in layer["inputs"]:
                i["layout"] = _get_tensor_act_layout(i["dim"], target_layout)
            for o in layer["outputs"]:
                o["layout"] = _get_tensor_act_layout(o["dim"], target_layout)

            if layer["op_type"] in ("qnn.csi.conv2d", "qnn.csi.deconv2d", "qnn.csi.dense"):
                if layer["op_type"] not in ("qnn.csi.dense",):
                    is_depthwise_conv2d = _is_depthwise_conv(
                        layer["inputs"][0]["dim"],
                        layer["inputs"][1]["dim"],
                        layer["attrs"]["groups"],
                        target_layout,
                    )
                layer["inputs"][1]["layout"] = _get_tensor_weight_layout(
                    layer["inputs"][1]["dim"], target_layout
                )
                layer["inputs"][2]["layout"] = _get_tensor_weight_layout(
                    layer["inputs"][2]["dim"], target_layout
                )

                if is_depthwise_conv2d:
                    if target_layout == "NCHW":
                        layer["inputs"][1]["layout"] = "O1HW"
                    else:
                        layer["inputs"][1]["layout"] = "1HWO"

                # fix weight layout for deconv
                if (
                    layer["op_type"] == "qnn.csi.deconv2d"
                    and layer["inputs"][1]["layout"] == "OIHW"
                ):
                    layer["inputs"][1]["layout"] = "IOHW"

        return data_cp

    dtj = QNNConvertDict()
    dtj.visit(qnn_ir["main"])
    data = dtj.qnn_data
    data = _modify_tensor_name(data)
    data = _filter_unused_values(data)
    data = _modify_values(data)

    return data


def convert_workload(model_data: collections.OrderedDict, dla_name: str):
    """convert json data into workload for npuperf"""
    mapping_path = get_mapping_by_name(dla_name)
    json_parser = Json2WorkloadParser(
        model_data,
        mapping_path,
        export_path="",
        merge_activation_function=True,
    )

    workload = json_parser.run()

    return workload


def get_all_submodule_name(pkg_path):
    """get submodule by specified path."""
    import importlib
    import pkgutil

    pkg = importlib.import_module(pkg_path)

    module_names = set()
    for _, name, _ in pkgutil.iter_modules(pkg.__path__):
        module_names.add(name)
    return list(module_names)


def get_all_HW_name():
    """get all hardware name in npuperf."""
    pkg_path = "npuperf.inputs.HW"
    return get_all_submodule_name(pkg_path)


def get_mapping_by_name(hw_name: str):
    """get mapping path with specified name."""
    main_mapping_path = "npuperf.inputs.Mapping"
    all_mapping = get_all_submodule_name(main_mapping_path)

    res = None
    for m in all_mapping:
        if hw_name.startswith(m):
            res = m
            break
    if res is None:
        raise ValueError("No mapping file was found!\n")

    res = main_mapping_path + "." + res
    return res


class Npuperf2Trace(object):
    """convert npuperf log into chrome trace."""

    def __init__(self, all_cme: List[CostModelEvaluation], arch_name: str) -> None:
        self.curr_ts = round(time.time() * (10**6))  # us timestamps
        self.all_cme = all_cme

        self.trace = HHBTrace()
        self.init_trace(arch_name=arch_name)

    def init_trace(self, arch_name):
        # add otherData field
        self.trace.update_other_data(
            {
                "source": f"npuperf {npuperf.__version__}",
                "hardware": arch_name,
            }
        )

    def export(self, path):
        to_json(self.chrome_trace, path, with_format=True)

    def run(self):
        for cme in self.all_cme:
            op_start_ts = self.curr_ts
            cme_name = str(cme.layer)
            layer_name = cme.layer.name  # (fixme) use actaul layer name

            ideal_latency = round(cme.ideal_cycle / 1e6 * 1e3)
            spatial_latency = round(cme.spatial_stall_cycle / 1e6 * 1e3)
            temporal_latency = round(cme.SS_comb / 1e6 * 1e3)
            # mac_latency = round(cme.latency_total0 / 1e6 * 1e3)
            mac_latency = ideal_latency + spatial_latency + temporal_latency

            onload_latency = round(cme.data_loading_cycle / 1e6 * 1e3)
            offload_latency = round(cme.data_offloading_cycle / 1e6 * 1e3)
            # total_latency = round(cme.latency_total2 / 1e6 * 1e3)
            total_latency = mac_latency + onload_latency + offload_latency

            # total latency = MAC latency + onloading latecny + offloading latency
            args = collections.OrderedDict()
            args["name"] = layer_name
            args["total_energy(mJ)"] = cme.energy_total / 1e9
            args["MAC utilization 2"] = cme.MAC_utilization2
            args["memory"] = cme.mem_ins_demc
            self.trace.insert_event(
                HHBTraceEventComplete(
                    name=cme_name,
                    cat=HHBTraceEventCategory.KERNEL,
                    ts=op_start_ts,
                    dur=total_latency,
                    args=args,
                )
            )
            self.curr_ts += total_latency

            # onloading latency
            onload_start_ts = op_start_ts
            args = collections.OrderedDict()
            args["name"] = layer_name
            self.trace.insert_event(
                HHBTraceEventComplete(
                    name=cme_name + "::onloading",
                    cat=HHBTraceEventCategory.MEMORY,
                    ts=onload_start_ts,
                    dur=onload_latency,
                    args=args,
                )
            )

            # mac latency = ideal computation cycle + spatial stall + temporal stall
            mac_start_ts = onload_start_ts + onload_latency
            args = collections.OrderedDict()
            args["name"] = layer_name
            self.trace.insert_event(
                HHBTraceEventComplete(
                    name=cme_name + "::MAC",
                    cat=HHBTraceEventCategory.KERNEL,
                    ts=mac_start_ts,
                    dur=mac_latency,
                    args=args,
                )
            )

            # mac latency:: ideal latency
            ideal_start_ts = mac_start_ts
            args = collections.OrderedDict()
            args["name"] = layer_name
            self.trace.insert_event(
                HHBTraceEventComplete(
                    name=cme_name + "::MAC::ideal_computation",
                    cat=HHBTraceEventCategory.KERNEL,
                    ts=ideal_start_ts,
                    dur=ideal_latency,
                    args=args,
                )
            )

            # mac latency:: spatial stall
            spatial_start_ts = ideal_start_ts + ideal_latency
            args = collections.OrderedDict()
            args["name"] = layer_name
            self.trace.insert_event(
                HHBTraceEventComplete(
                    name=cme_name + "::MAC::spatial_stall",
                    cat=HHBTraceEventCategory.KERNEL,
                    ts=spatial_start_ts,
                    dur=spatial_latency,
                    args=args,
                )
            )

            # mac latency:: temporal stall
            temporal_start_ts = spatial_start_ts + spatial_latency
            args = collections.OrderedDict()
            args["name"] = layer_name
            self.trace.insert_event(
                HHBTraceEventComplete(
                    name=cme_name + "::MAC::temporal_stall",
                    cat=HHBTraceEventCategory.KERNEL,
                    ts=temporal_start_ts,
                    dur=temporal_latency,
                    args=args,
                )
            )

            # offloading latency
            offload_start_ts = mac_start_ts + mac_latency
            args = collections.OrderedDict()
            args["name"] = layer_name
            self.trace.insert_event(
                HHBTraceEventComplete(
                    name=cme_name + "::offloading",
                    cat=HHBTraceEventCategory.MEMORY,
                    ts=offload_start_ts,
                    dur=offload_latency,
                    args=args,
                )
            )


def create_args_for_npuperf(config: NpuperfArchConfig) -> argparse.Namespace:
    if config.hardware_level == "level2":
        hw = config.detailed_config
        hw["name"] = config.name
    else:
        hw = config.name
    data = {
        "hw": hw,
        "flow": config.flow,
        "gb_size": config.gb_size,
        "gb_bw": config.gb_bw,
        "dram_bw": config.dram_bw,
    }
    args = argparse.Namespace(**data)
    return args


def generate_trace(model_data, arch_config, save_temps=False, output_dir="."):
    """Generate trace data with npuperf tool."""
    all_hw = get_all_HW_name()
    if arch_config in all_hw:
        config = NpuperfArchConfig()
        config.name = arch_config
    else:
        config = NpuperfArchConfig(arch_config)
        config.name = (
            os.path.splitext(os.path.basename(arch_config))[0] if not config.name else config.name
        )
    config.initialize()
    config.check()

    stage_pipeline = [
        HardwareModifierStage,
        MappingGeneratorStage,
        JsonWorkloadParserStage,
        WorkloadParserStage,
        WorkloadStage,
        SpatialMappingConversionStage,
        MinimalLatencyStage,
        LomaStage,
        ZigZagCostModelStage,
    ]
    if config.flow:
        stage_pipeline[-2] = StationaryLomaStage
    if config.hardware_level == "level2":
        stage_pipeline[0] = HardwareGeneratorStage

    dump_filename_pattern = ""
    pickle_filename = ""
    plot_filename_pattern = ""
    if save_temps:
        stage_pipeline[4:4] = [
            SumAndSaveAllLayersStage,
            PickleSaveStage,
            # CompleteSaveStage,
            SimpleSaveStage,
            PlotTemporalMappingsStage,
        ]
        npuperf_output_root = os.path.join(output_dir, "npuperf_outputs")
        dump_filename_pattern = os.path.join(
            npuperf_output_root, "result_layerwise", config.name, "?.json"
        )
        pickle_filename = os.path.join(
            npuperf_output_root, "result_layerwise", config.name, f"{config.name}.pickle"
        )
        plot_filename_pattern = os.path.join(
            npuperf_output_root, "result_plot", config.name, "?.png"
        )

    args = create_args_for_npuperf(config)
    mainstage = MainStage(
        list_of_callables=stage_pipeline,
        args=args,
        loma_lpf_limit=6,  # required by LOMA stage, set to 6 - 8 will be nice.
        stationary=args.flow,
        json_NN=model_data,
        dump_filename_pattern=dump_filename_pattern,
        pickle_filename=pickle_filename,
        plot_filename_pattern=plot_filename_pattern,
    )

    results = mainstage.run()

    if save_temps:
        results = list([item for item in results[0][1]])
    else:
        results = list([item[0] for item in results])

    n2c = Npuperf2Trace(results, arch_name=config.name)
    n2c.run()
    trace_data = n2c.trace.to_dict()

    return trace_data
