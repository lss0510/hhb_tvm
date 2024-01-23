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
Chrome trace event format
"""
from enum import Enum
from typing import List
import copy
import os

import pandas

import hhb
from hhb.core.common import HHBException, from_json


HHB_TRACE_EVENT_MAP = {}
QNN_TRACE_PID = 0


def hhb_trace_envent_register(name):
    """
    Utility function to register a event class.

    Classes decorated with `hhb_trace_envent_register` will be put into HHB_TRACE_EVENT_MAP

    Example
    -------

        @hhb_trace_envent_register("a")
        class A():
            pass

    """

    def decorator(cls):
        HHB_TRACE_EVENT_MAP[name] = cls
        return cls

    return decorator


class HHBTraceEventCategory(Enum):
    RUNTIME = "runtime"
    CPU_OPERATOR = "cpu_operator"
    MEMORY = "memory"
    CPU_KERNEL = "cpu_kernel"
    NPU_KERNEL = "npu_kernel"
    KERNEL = "kernel"


class HHBTraceEventType(Enum):
    DURATION_B = "B"
    DURATION_E = "E"
    COMPLETE_X = "X"
    INSTANT_i = "i"
    COUNTER_C = "C"
    ASYNC_b = "b"
    ASYNC_n = "n"
    ASYNC_e = "e"
    FLOW_s = "s"
    FLOW_t = "t"
    FLOW_f = "f"
    METADATA_M = "M"


class HHBTraceEventFormatBase(object):
    def __init__(
        self,
        name: str = "",
        cat: HHBTraceEventCategory = HHBTraceEventCategory.KERNEL,
        ph: HHBTraceEventType = HHBTraceEventType.COMPLETE_X,
        ts: int = 0,
        pid: int = 0,
        tid: int = 0,
        args: dict = None,
    ) -> None:
        self.name = name
        self.cat = cat
        self.ph = ph
        self.ts = ts
        self.pid = pid
        self.tid = tid
        self.args = args

    def to_dict(self) -> dict:
        # get all non-function attributes
        res = {
            attr: getattr(self, attr)
            for attr in dir(self)
            if not callable(getattr(self, attr))
            and not attr.startswith("__")
            and not hasattr(HHBTraceEventFormatBase, attr)
        }
        if res["args"] is None:
            res.pop("args")
        res["cat"] = res["cat"].value
        res["ph"] = res["ph"].value

        return res

    def from_dict(self, data: dict):
        if not isinstance(data, dict):
            raise HHBException(f"Requried dict but get {type(data)}")
        for k, v in data.items():
            if hasattr(self, k):
                setattr(self, k, v)
        self.cat = HHBTraceEventCategory(self.cat)
        self.ph = HHBTraceEventType(self.ph)
        return self

    def __lt__(self, other):
        return self.ts < other.ts


@hhb_trace_envent_register("B")
class HHBTraceEventDurationBegin(HHBTraceEventFormatBase):
    def __init__(
        self,
        name: str = "",
        cat: HHBTraceEventCategory = HHBTraceEventCategory.KERNEL,
        ts: int = 0,
        pid: int = 0,
        tid: int = 0,
        args: dict = None,
    ) -> None:
        super().__init__(name, cat, HHBTraceEventType.DURATION_B, ts, pid, tid, args)


@hhb_trace_envent_register("E")
class HHBTraceEventDurationEnd(HHBTraceEventFormatBase):
    def __init__(
        self,
        name: str = "",
        cat: HHBTraceEventCategory = HHBTraceEventCategory.KERNEL,
        ts: int = 0,
        pid: int = 0,
        tid: int = 0,
        args: dict = None,
    ) -> None:
        super().__init__(name, cat, HHBTraceEventType.DURATION_E, ts, pid, tid, args)


@hhb_trace_envent_register("X")
class HHBTraceEventComplete(HHBTraceEventFormatBase):
    def __init__(
        self,
        name: str = "",
        cat: HHBTraceEventCategory = HHBTraceEventCategory.KERNEL,
        dur: int = 0,
        ts: int = 0,
        pid: int = 0,
        tid: int = 0,
        args: dict = None,
    ) -> None:
        super().__init__(name, cat, HHBTraceEventType.COMPLETE_X, ts, pid, tid, args)
        self.dur = dur


def dump_dataframe(data: pandas.DataFrame, topk=None, title=None, display=True, to_csv=None):
    if display:
        if title:
            print(f"{'-'*30} {title}: {'-'*30}")

        if topk is not None:
            print(data.head(topk).to_string(index=False))
        else:
            print(data.to_string(index=False))
        print()

    if to_csv:
        data.to_csv(to_csv, index=False)


class HHBTrace(object):
    def __init__(self) -> None:
        self.other_data: dict = {}
        self.trace_events: List[HHBTraceEventFormatBase] = []

    def update_other_data(self, *args, **kwargs):
        for arg in args:
            if not isinstance(arg, dict):
                raise HHBException("Only support for dict type for current positional arguments\n")
            self.other_data.update(arg)

        self.other_data.update(kwargs)

    def insert_event(self, event: HHBTraceEventFormatBase):
        if event:
            self.trace_events.append(event)

    def to_dict(self):
        events_l = []
        for event in self.trace_events:
            events_l.append(event.to_dict())

        res = {"otherData": self.other_data, "traceEvents": events_l}
        return res

    def from_dict(self, data: dict):
        if not isinstance(data, dict):
            raise HHBException(f"Requried dict but get {type(data)}\n")
        self.other_data = data.get("otherData", self.other_data)

        trace_events = data.get("traceEvents", [])
        for event in trace_events:
            if event["ph"] in HHB_TRACE_EVENT_MAP:
                self.insert_event(HHB_TRACE_EVENT_MAP[event["ph"]]().from_dict(event))
            else:
                raise HHBException(f"Don't know how to parse event type {event['ph']}\n")
        return self

    def convert_duration2complete(self):
        # convert duration_b + duration_e -> complete_x
        def _check_pair(event1: HHBTraceEventFormatBase, event2: HHBTraceEventFormatBase):
            res = False
            if (
                event1.name == event2.name
                and event1.cat == event2.cat
                and event1.pid == event2.pid
                and event1.tid == event2.tid
            ):
                res = True
            return res

        # sort event by ts
        self.trace_events.sort()

        new_events = []
        stack = []
        for event in self.trace_events:
            if isinstance(event, HHBTraceEventDurationBegin):
                stack.append(event)
            elif isinstance(event, HHBTraceEventDurationEnd):
                if len(stack) == 0:
                    raise HHBException(
                        "Invalid trace data, because the number of events with ph=E is more than "
                        "that of events with ph=B.\n"
                    )
                top_event = stack.pop()
                if _check_pair(top_event, event):
                    new_args = {}
                    if top_event.args:
                        new_args.update(top_event.args)
                    if event.args:
                        new_args.update(event.args)
                    if not new_args:
                        new_args = None
                    x_event = HHBTraceEventComplete(
                        name=top_event.name,
                        cat=top_event.cat,
                        dur=(event.ts - top_event.ts),
                        ts=top_event.ts,
                        pid=top_event.pid,
                        tid=top_event.tid,
                        args=new_args,
                    )
                    new_events.append(x_event)
                else:
                    raise HHBException("Invalid trace data, becase get mismatch events.\n")
            else:
                new_events.append(event)
        if len(stack) > 0:
            raise HHBException(
                "Invalid trace data, because the number of events with ph=B is more than "
                "that of events with ph=E.\n"
            )
        self.trace_events = new_events

    def _convert_complete2pandas(self) -> pandas.DataFrame:
        self.convert_duration2complete()

        data = {
            "Event Cat": [],
            "Event Type": [],
            "Duration(ms)": [],
            "Event Name": [],
        }

        for event in self.trace_events:
            if event.ph != HHBTraceEventType.COMPLETE_X:
                continue
            if (
                event.cat == HHBTraceEventCategory.KERNEL
                and event.args
                and "from" in event.args
                and event.args["from"] in ("relay", "qnn")
            ):
                # (FIXME@chenf: relay/qnn trace should be ignore)
                continue
            data["Event Cat"].append(event.cat.value)
            data["Event Type"].append(event.ph.value)
            data["Duration(ms)"].append(event.dur / 1000)  # ms
            data["Event Name"].append(event.name)
        df = pandas.DataFrame(data)

        return df

    def profile_complete_events_by_group(
        self, group_by="Event Cat", display=True, to_csv=None
    ) -> pandas.DataFrame:
        df = self._convert_complete2pandas()

        args = {
            "Total Duration(ms)": "sum",
            "Call Counts": "size",
            "Max Duration(ms)": "max",
            "Min Duration(ms)": "min",
            "Avg Duration(ms)": "mean",
        }
        grouped_df = df.groupby(group_by)["Duration(ms)"].agg(**args).reset_index()
        total_time = grouped_df["Total Duration(ms)"].sum()
        grouped_df["Percent(%)"] = (grouped_df["Total Duration(ms)"] / total_time) * 100
        grouped_df = grouped_df.sort_values(by=["Total Duration(ms)"], ascending=False)

        dump_dataframe(grouped_df, title="Events Summary", display=display, to_csv=to_csv)

        return group_by

    def profile_complete_events_all(self, topk: int = 10, display=True, to_csv=None):
        df = self._convert_complete2pandas()

        single_df = df.sort_values(by=["Duration(ms)"], ascending=False)
        total_time = single_df["Duration(ms)"].sum()
        single_df["Percent(%)"] = (single_df["Duration(ms)"] / total_time) * 100

        dump_dataframe(
            single_df,
            topk=topk,
            title=f"Cost top{topk} of all events",
            display=display,
            to_csv=to_csv,
        )

        return single_df

    def _convert_kernel2pandas(self) -> pandas.DataFrame:
        self.convert_duration2complete()

        data = {
            "Kernel Name": [],
            "Kernel Type": [],
            "Duration(ms)": [],
            "Layer Name": [],
        }

        for event in self.trace_events:
            if event.ph != HHBTraceEventType.COMPLETE_X:
                continue
            if event.cat not in (
                HHBTraceEventCategory.CPU_KERNEL,
                HHBTraceEventCategory.NPU_KERNEL,
                HHBTraceEventCategory.KERNEL,
            ):
                continue
            if event.cat not in (
                HHBTraceEventCategory.CPU_KERNEL,
                HHBTraceEventCategory.NPU_KERNEL,
            ):
                continue
            data["Kernel Name"].append(event.name)
            data["Kernel Type"].append(event.cat.value)
            data["Duration(ms)"].append(event.dur / 1000)
            data["Layer Name"].append(event.args["name"])

        df = pandas.DataFrame(data)

        return df

    def profile_kernel_by_group(
        self, group_by="Kernel Name", topk: int = 10, display=True, to_csv=None
    ) -> pandas.DataFrame:
        df = self._convert_kernel2pandas()

        args = {
            "Total Duration(ms)": "sum",
            "Call Counts": "size",
            "Max Duration(ms)": "max",
            "Min Duration(ms)": "min",
            "Avg Duration(ms)": "mean",
        }
        grouped_df = df.groupby(group_by)["Duration(ms)"].agg(**args).reset_index()
        total_time = grouped_df["Total Duration(ms)"].sum()
        grouped_df["Percent(%)"] = (grouped_df["Total Duration(ms)"] / total_time) * 100
        grouped_df = grouped_df.sort_values(by=["Total Duration(ms)"], ascending=False)

        dump_dataframe(
            grouped_df, topk=topk, title="Kernels Summary", display=display, to_csv=to_csv
        )

        return group_by

    def profile_kernel_all(self, topk: int = 10, display=True, to_csv=None):
        df = self._convert_kernel2pandas()

        single_df = df.sort_values(by=["Duration(ms)"], ascending=False)
        total_time = single_df["Duration(ms)"].sum()
        single_df["Percent(%)"] = (single_df["Duration(ms)"] / total_time) * 100

        dump_dataframe(
            single_df,
            topk=topk,
            title=f"Cost top{topk} of all kernels",
            display=display,
            to_csv=to_csv,
        )

        return single_df

    def get_layer_results(self):
        self.convert_duration2complete()
        # layer_name -> results
        res = {}
        for event in self.trace_events:
            if (
                event.ph != HHBTraceEventType.COMPLETE_X
                or event.cat != HHBTraceEventCategory.CPU_OPERATOR
                or event.name not in ("cpu_ops_execution", "subgraph_execution")
            ):
                continue
            if (
                not event.args
                or "output_files" not in event.args
                or "output_names" not in event.args
            ):
                continue
            layer_name = event.args["name"]
            res[layer_name] = event.args["output_files"]

            assert len(event.args["output_files"]) == len(
                event.args["output_names"]
            ), "Mismatch number of output_files and output_names"

            for idx, name in enumerate(event.args["output_names"]):
                res[name] = event.args["output_files"][idx]
        return res


class HHBIRTrace(object):
    def __init__(self, source: str = "hhb", trace_type: str = None, layers: list = None) -> None:
        self.source: str = source
        self.version: str = hhb.__version__
        self.trace_type: str = trace_type
        self.layers: list = layers

    def to_dict(self):
        return {
            "otherData": {
                "source": self.source,
                "version": self.version,
                "trace_type": self.trace_type,
            },
            "layers": self.layers,
        }

    def imported_from(self, layers: list, trace_type: str):
        self.trace_type = trace_type
        self.layers = layers
        return self

    def from_dict(self, data: dict):
        if not isinstance(data, dict):
            raise HHBException(f"Requried dict but get {type(data)}")
        other_data: dict = data.get("otherData", {})
        if other_data:
            self.source = other_data.get("source", self.source)
            self.trace_type = other_data.get("trace_type", self.trace_type)
        self.layers = data.get("layers", self.layers)
        return self

    def profile(self, profile_method=None, topk=10, display=True, output_dir=None):
        data = {
            "type": [],
            "exp": [],
            "comp": [],
            "mul": [],
            "sub": [],
            "add": [],
            "div": [],
            "fused_mul_add": [],
            "params": [],
            "output": [],
            "name": [],
        }

        for layer in self.layers:
            data["type"].append(layer["op"]["type"])
            data["exp"].append(layer["calculation_amount"]["exp"])
            data["comp"].append(layer["calculation_amount"]["comp"])
            data["mul"].append(layer["calculation_amount"]["mul"])
            data["sub"].append(layer["calculation_amount"]["sub"])
            data["add"].append(layer["calculation_amount"]["add"])
            data["div"].append(layer["calculation_amount"]["div"])
            data["fused_mul_add"].append(layer["calculation_amount"]["fused_mul_add"])
            data["params"].append(layer["memory"]["params"])
            data["output"].append(layer["memory"]["output"])
            data["name"].append(layer["op"]["name"])
        df = pandas.DataFrame(data)

        df["total_flops"] = df["exp"] + df["comp"] + df["mul"] + df["sub"] + df["add"] + df["div"]
        df.insert(8, "flops", df["total_flops"])
        df.drop("total_flops", axis=1, inplace=True)

        df["mem"] = df["params"] + df["output"]
        df.insert(11, "total_mem", df["mem"])
        df.drop("mem", axis=1, inplace=True)

        df.rename(columns={"fused_mul_add": "fused_mul_add(macc)"}, inplace=True)

        to_csv = None
        if output_dir is not None:
            to_csv = os.path.join(output_dir, f"{self.trace_type}_summary.csv")
        dump_dataframe(
            df,
            topk=topk,
            title=f"{self.trace_type.upper()} Summary",
            display=display,
            to_csv=to_csv,
        )

        if profile_method is None or "sorted_by_macc" in profile_method:
            sorted_df = df.sort_values(by=["fused_mul_add(macc)"], ascending=False)
            total = sorted_df["fused_mul_add(macc)"].sum()
            sorted_df["Percent for macc(%)"] = (sorted_df["fused_mul_add(macc)"] / total) * 100
            dump_dataframe(
                sorted_df,
                topk=topk,
                title=f"{self.trace_type.upper()} sorted by macc",
                display=display,
                to_csv=os.path.join(output_dir, f"{self.trace_type}_sorted_by_macc.csv")
                if to_csv
                else None,
            )

        if profile_method is None or "sorted_by_flops" in profile_method:
            sorted_df = df.sort_values(by=["flops"], ascending=False)
            total = sorted_df["flops"].sum()
            sorted_df["Percent for flops(%)"] = (sorted_df["flops"] / total) * 100
            dump_dataframe(
                sorted_df,
                topk=topk,
                title=f"{self.trace_type.upper()} sorted by flops",
                display=display,
                to_csv=os.path.join(output_dir, f"{self.trace_type}_sorted_by_flops.csv")
                if to_csv
                else None,
            )

        if profile_method is None or "sorted_by_total_mem" in profile_method:
            sorted_df = df.sort_values(by=["total_mem"], ascending=False)
            total = sorted_df["total_mem"].sum()
            sorted_df["Percent for total_mem(%)"] = (sorted_df["total_mem"] / total) * 100
            dump_dataframe(
                sorted_df,
                topk=topk,
                title=f"{self.trace_type.upper()} sorted by total_mem",
                display=display,
                to_csv=os.path.join(output_dir, f"{self.trace_type}_sorted_by_total_mem.csv")
                if to_csv
                else None,
            )

        return df


class HHBAccTrace(object):
    def __init__(self) -> None:
        self.source = "hhb"
        self.trace_type = "acc"
        self.layers = []

    def from_dict(self, data: dict):
        if not isinstance(data, dict):
            raise HHBException(f"Requried dict but get {type(data)}\n")
        if "trace_type" not in data or data["trace_type"] != "acc":
            raise HHBException("Unsupport for this file.")
        self.layers = data.get("layers_info", self.layers)

        return self

    def get_float_results(self):
        # layer_name -> float_path
        res = {}
        for layer in self.layers:
            if "float_result_path" not in layer:
                continue
            float_path = layer["float_result_path"]
            float_path = os.path.join("dump", os.path.basename(float_path))

            layer_name = layer["name"]
            res[layer_name] = float_path

        return res


def merge_qnn_csinn_trace(qnn_trace: HHBIRTrace, csinn_trace: HHBTrace):
    merge_trace = HHBTrace()
    merge_trace.other_data = {
        "source": "hhb",
        qnn_trace.source: {"version": qnn_trace.version, "trace_type": qnn_trace.trace_type},
        csinn_trace.other_data["source"]: csinn_trace.other_data,
    }
    merge_trace.trace_events = copy.deepcopy(csinn_trace.trace_events)

    name2event = {}
    for event in csinn_trace.trace_events:
        args = event.args
        if args and "name" in args:
            name2event[args["name"]] = event

    for layer in qnn_trace.layers:
        op_type = layer["op"]["type"]
        op_name = layer["op"]["name"]
        if op_name not in name2event:
            continue

        corr_event = name2event[op_name]

        qnn_event = HHBTraceEventComplete(
            name=op_type,
            cat=HHBTraceEventCategory.KERNEL,
            dur=corr_event.dur,
            ts=corr_event.ts,
            pid=QNN_TRACE_PID,
            tid=QNN_TRACE_PID,
            args={
                "name": op_name,
                "calculation_amount": layer["calculation_amount"],
                "memory": layer["memory"],
                "from": "qnn",
            },
        )
        merge_trace.insert_event(qnn_event)
    merge_trace.trace_events.sort()
    return merge_trace


def get_trace_obj(data):
    res = None
    if data["otherData"]["source"] == "hhb" and data["otherData"]["trace_type"] == "qnn":
        res = HHBIRTrace().from_dict(data)

    if "csinn" in data["otherData"]["source"]:
        res = HHBTrace().from_dict(data)

    return res


def merge_trace(files):
    res = None

    qnn_trace, csinn_trace = None, None

    trace1 = from_json(files[0])
    trace2 = from_json(files[1])

    trace1 = get_trace_obj(trace1)
    if trace1 and isinstance(trace1, HHBIRTrace):
        qnn_trace = trace1
    if trace1 and isinstance(trace1, HHBTrace):
        csinn_trace = trace1

    trace2 = get_trace_obj(trace2)
    if trace2 and isinstance(trace2, HHBIRTrace):
        qnn_trace = trace2
    if trace2 and isinstance(trace2, HHBTrace):
        csinn_trace = trace2

    csinn_trace.convert_duration2complete()

    if qnn_trace and csinn_trace:
        res = merge_qnn_csinn_trace(qnn_trace, csinn_trace)

    return res


if __name__ == "__main__":
    trace = HHBTrace()
    trace.update_other_data({"hardware": "c920", "version": "2.2.0"}, tool="hhb")
    trace.insert_event(
        HHBTraceEventComplete(
            name="c920_conv_kernel",
            cat=HHBTraceEventCategory.CPU_KERNEL,
            dur=50,
            args={"shape": [1, 3, 224, 224], "name": "layer0"},
        )
    )

    data = trace.to_dict()

    from pprint import pprint

    pprint(data)
