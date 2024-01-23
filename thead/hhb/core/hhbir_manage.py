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
"""Manage HHB IR"""
import os
import sys
import logging
from abc import ABC
from abc import abstractmethod
import yaml
import shutil
import glob

import tvm
from tvm import relay
from tvm import runtime
from tvm.relay import quantize as qtz
from tvm.relay.backend import executor_factory
from tvm.contrib import graph_runtime
from tvm.ir.tensor_type import TensorType
from tvm.ir.type import TupleType
from tvm.relay.op.contrib import csinn

from .common import (
    HHBException,
    get_target,
    hhb_ir_helper,
    HHB_IR,
    AttributeDict,
    ensure_compiler,
)


# pylint: disable=invalid-name
logger = logging.getLogger("HHB")


def match_mod_params(mod, params):
    """The params of module's main function match the params dict."""
    if not params:
        return mod, params
    var_name_list = []
    for arg in mod["main"].params:
        if arg.name_hint not in var_name_list:
            var_name_list.append(arg.name_hint)
    params_new = {}
    flag = False
    for k in params.keys():
        if k not in var_name_list and ("v" + k) in var_name_list:
            flag = True
            break

    if flag:
        logger.debug("mod does not match params, and try to update the params dict...")
        for k, v in params.items():
            if k not in var_name_list and ("v" + k) in var_name_list:
                params_new["v" + k] = v
            else:
                params_new[k] = v
        params = params_new

    return mod, params


def get_input_info_from_relay(mod, params):
    input_name_list = []
    input_shape_list = []
    input_dtype_list = []

    for arg in mod["main"].params:
        if (not params) or arg.name_hint not in params.keys():
            input_name_list.append(str(arg.name_hint))
            input_shape_list.append(list(map(int, arg.type_annotation.shape)))
            input_dtype_list.append(str(arg.type_annotation.dtype))

    return input_name_list, input_shape_list, input_dtype_list


def get_output_info_from_relay(mod, params):
    output_shape_list = []
    output_dtype_list = []

    # convert dynamic node into static node.
    mod_main_old = mod["main"]
    from tvm.relay.quantize.quantize_hhb import _bind_params

    if params:
        mod["main"] = _bind_params(mod["main"], params)
    with tvm.transform.PassContext(opt_level=2):
        mod = relay.transform.DynamicToStatic()(mod)

    if isinstance(mod["main"].ret_type, TupleType):
        for item in mod["main"].ret_type.fields:
            output_shape_list.append(list(map(int, item.shape)))
            output_dtype_list.append(str(item.dtype))
    elif isinstance(mod["main"].ret_type, TensorType):
        output_shape_list.append(list(map(int, mod["main"].ret_type.shape)))
        output_dtype_list.append(str(mod["main"].ret_type.dtype))
    else:
        raise HHBException("unsupport for {}".format(type(mod["main"].ret_type)))

    mod["main"] = mod_main_old
    return output_shape_list, output_dtype_list


def check_included_filename(model_path, hhbir):
    """Check whether included all files exists."""
    for k in hhbir.all_included_filenames:
        if not os.path.exists(os.path.join(model_path, k)):
            raise HHBException("There is no {} in {}".format(k, model_path))


def reorder_pixel_format(mod, params):
    """If original model's input data pixel format is rgb, then covert it to bgr,
    otherwise, then convert it to rgb.

    Parameters
    ----------
    mod : tvm.IRModule
        The relay module for compilation
    params : dict of str to tvm.nd.NDArray
        The parameter dict to be used by relay

    Returns
    -------
    mod : tvm.IRModule
        The modified relay module
    params : dict of str to tvm.nd.NDArray
        The modified parameter dict to be used by relay
    """

    def _impl_v1(mod, params):
        """Weights/biases are not embedded into relay."""

        class InnerVisitor(relay.ExprVisitor):
            """Counting the number of call"""

            def __init__(self):
                super(InnerVisitor, self).__init__()
                self.weight_name = []

            def visit_call(self, call):
                _ = [self.visit(arg) for arg in call.args]

                pre_call = call.args[0]
                if call.op.name == "nn.conv2d" and isinstance(pre_call, relay.expr.Var):

                    weight = call.args[1]
                    self.weight_name.append(weight.name_hint)

        iv = InnerVisitor()
        iv.visit(mod["main"])
        to_change = iv.weight_name

        for name in to_change:
            data = params[name].asnumpy()
            if data.shape[1] != 3:
                continue
            data = data[:, ::-1, :, :]
            params[name].copyfrom(data)
        return mod, params

    def _impl_v2(mod):
        """Weights/biases are embedded into relay."""

        class InnerMutator(relay.ExprMutator):
            """Counting the number of call"""

            def __init__(self):
                super(InnerMutator, self).__init__()

            def visit_call(self, call):
                op_args = [self.visit(arg) for arg in call.args]

                pre_call = op_args[0]
                if call.op.name == "nn.conv2d" and isinstance(pre_call, relay.expr.Var):
                    weight = op_args[1]
                    weight_val = weight.data.asnumpy()
                    if weight_val.shape[1] == 3:
                        weight_val = weight_val[:, ::-1, :, :]
                        weight.data.copyfrom(weight_val)
                    op_args[1] = weight
                new_call = relay.expr.Call(call.op, op_args, call.attrs, call.type_args, call.span)
                return new_call

        mod["main"] = InnerMutator().visit(mod["main"])
        return mod

    if params:
        return _impl_v1(mod, params)
    else:
        return _impl_v2(mod), params


class HHBIRType(object):
    """Denote the HHB IR type"""

    UNKNOWN = -1
    RELAY = 0
    QNN = 1
    FLOAT_CODEGEN = 2
    X86_QNN_CODEGEN = 3
    BOARD_QNN_CODEGEN = 4

    TYPE2NAME = {
        UNKNOWN: "unknown",
        RELAY: "relay",
        QNN: "qnn",
        FLOAT_CODEGEN: "float_codegen",
        X86_QNN_CODEGEN: "x86_qnn_codegen",
        BOARD_QNN_CODEGEN: "board_qnn_codegen",
    }
    NAME2TYPE = {v: k for k, v in TYPE2NAME.items()}


class HHBIRBase(ABC):
    """Abstract class for HHB command line interface.

    Provide a unified way to convert and save HHB model.

    """

    def __init__(self):
        self.all_included_filenames = []

    @staticmethod
    @abstractmethod
    def name():
        """IR name"""

    @staticmethod
    @abstractmethod
    def model_type():
        """IR type"""

    @abstractmethod
    def load_model(self, model_path):
        """Load a model from a given path.

        Parameters
        ----------
        model_path : str
            Path to a tar file
        """

    @abstractmethod
    def convert(self):
        """Convert input module into current stage."""

    @abstractmethod
    def save_model(self, model_path):
        """Save current module into files."""

    def check_ir_type(self, model_path):
        files = os.listdir(model_path)
        res = True
        for f in self.all_included_filenames:
            if f not in files:
                res = False
                break
        return res


@hhb_ir_helper
class HHBRelayIR(HHBIRBase):
    def __init__(self):
        super().__init__()
        self.mod_name = "relay.txt"
        self.params_name = "relay.params"

        self.all_included_filenames.extend([self.mod_name, self.params_name])

        self._mod = None
        self._params = None

    @staticmethod
    def name():
        return HHBIRType.TYPE2NAME[HHBIRType.RELAY]

    @staticmethod
    def model_type():
        return HHBIRType.RELAY

    def set_model(self, mod, params):
        self._mod = mod
        self._params = params

    def get_model(self):
        return self._mod, self._params

    def load_model(self, model_path):
        mod_path = os.path.join(model_path, self.mod_name)
        params_path = os.path.join(model_path, self.params_name)
        check_included_filename(model_path, self)
        with open(mod_path, "r") as f:
            mod = tvm.parser.fromtext(f.read())
        with open(params_path, "rb") as f:
            params = tvm.relay.load_param_dict(f.read())
        self._mod, self._params = match_mod_params(mod, params)

    def convert(self):
        pass

    def save_model(self, model_path):
        """Save current module into files."""
        assert self._mod, "Error: empty module."
        if not os.path.exists(model_path):
            raise HHBException("Directory {} is not exists".format(model_path))
        mod_path = os.path.join(model_path, self.mod_name)
        params_path = os.path.join(model_path, self.params_name)

        with open(mod_path, "w") as f:
            logger.info("exporting relay ir to {}".format(f.name))
            f.write(self._mod.astext())

        with open(params_path, "wb") as f:
            logger.info("exporting relay params to {}".format(f.name))
            f.write(tvm.relay.save_param_dict(self._params))


@hhb_ir_helper
class HHBQNNIR(HHBIRBase):
    def __init__(self):
        super().__init__()
        self.mod_name = "qnn.txt"
        self.params_name = "qnn.params"

        self.info_file = "qnn_info.yaml"
        self.info_dict = {}

        self.all_included_filenames.extend([self.mod_name])

        self._curr_mod = None
        self._curr_params = None

    @staticmethod
    def name():
        return HHBIRType.TYPE2NAME[HHBIRType.QNN]

    @staticmethod
    def model_type():
        return HHBIRType.QNN

    def get_model(self):
        return self._curr_mod, self._curr_params

    def set_model(self, mod, params=None):
        self._curr_mod = mod
        self._curr_params = params

    def load_model(self, model_path):
        mod_path = os.path.join(model_path, self.mod_name)
        params_path = os.path.join(model_path, self.params_name)
        # ensure all needed files existing
        check_included_filename(model_path, self)
        with open(mod_path, "r") as f:
            self._curr_mod = tvm.parser.fromtext(f.read())
            logger.info("restore qnn ir from %s" % mod_path)
        if os.path.exists(params_path):
            with open(params_path, "rb") as f:
                self._curr_params = tvm.relay.load_param_dict(f.read())
                logger.info("restore relay params from %s" % params_path)
        self._curr_mod, self._curr_params = match_mod_params(self._curr_mod, self._curr_params)

        if os.path.exists(os.path.join(model_path, self.info_file)):
            with open(os.path.join(model_path, self.info_file), "r") as f:
                info_dict = yaml.safe_load(f.read())
                self.info_dict["preprocess"] = AttributeDict(**info_dict["preprocess"])
                self.info_dict["qnn_config"] = AttributeDict(**info_dict["qnn_config"])

    def convert(self, input_module, qconfig, dataset=None, target="x86_ref"):
        if (
            not isinstance(input_module, (list, tuple))
            and not isinstance(input_module[0], tvm.ir.module.IRModule)
            and not isinstance(input_module[1], dict)
        ):
            raise HHBException("input_module should be [IRModule, dict]")

        from .quantization_manage import quantize_model

        qconfig["target"] = target
        qconfig["params_path"] = os.path.join(qconfig["params_path"], self.params_name)
        self._curr_mod = quantize_model(input_module[0], input_module[1], qconfig, dataset, target)

    def save_model(self, model_path, preprocess_params=None, config_dict=None):
        """Save current module into files."""
        assert self._curr_mod, "Error: empty module."
        if not os.path.exists(model_path):
            raise HHBException("Directory {} is not exists".format(model_path))
        mod_path = os.path.join(model_path, self.mod_name)
        params_path = os.path.join(model_path, self.params_name)

        with open(mod_path, "w") as f:
            logger.info("exporting qnn ir to {}".format(f.name))
            f.write(self._curr_mod.astext())

        if self._curr_params:
            with open(params_path, "wb") as f:
                logger.info("exporting qnn params to {}".format(f.name))
                f.write(tvm.relay.save_param_dict(self._curr_params))
        # if need, save extra info, which will be used in codegen stage
        info_path = os.path.join(model_path, self.info_file)
        if preprocess_params:
            logger.info("write preprocess info into %s" % info_path)
            with open(info_path, "w") as f:
                yaml.safe_dump({"preprocess": dict(preprocess_params)}, f, default_flow_style=False)

        if config_dict:
            logger.info("write quantization info into %s" % info_path)
            with open(info_path, "a") as f:
                yaml.safe_dump({"qnn_config": config_dict}, f, default_flow_style=False)

        # save qnn into onnx
        if config_dict is not None:
            with tvm.transform.PassContext(
                opt_level=3, config={"relay.ext.csinn.options": config_dict}
            ):
                from tvm.relay.quantize.qnn2onnx import (
                    InsertQDQToQNN,
                    qnn_to_onnx,
                    ConvertQnnToFloat16,
                )

                if config_dict["target"] in ("th1520", "hth1520") and config_dict[
                    "quantization_scheme"
                ] in ("int8_sym", "int16_sym"):
                    logger.warn(
                        "Can not save onnx for th1520 with quantization scheme int8_sym/int16_sym."
                    )
                else:
                    if config_dict["quantization_scheme"] == "float16":
                        qdq_qfunc = ConvertQnnToFloat16()(self._curr_mod)
                    elif config_dict["quantization_scheme"] == "float32":
                        qdq_qfunc = self._curr_mod
                    else:
                        qdq_qfunc = InsertQDQToQNN()(self._curr_mod)
                    qnn_onnx_path = os.path.join(model_path, "qnn.onnx")
                    qnn_to_onnx(qdq_qfunc, {}, "qnn_csi", path=qnn_onnx_path)


@hhb_ir_helper
class HHBFloatCodegenIR(HHBIRBase):
    def __init__(self):
        super().__init__()
        self.graph_name = "codegen_x86_float.json"
        self.lib_name = "codegen_x86_float.so"
        self.params_name = "codegen_x86_float.params"
        self.info_file = "codegen_x86_float.yaml"

        self._curr_module = None
        self.info_dict = {}

        self.all_included_filenames.extend([self.graph_name, self.lib_name, self.params_name])

    @staticmethod
    def name():
        return HHBIRType.TYPE2NAME[HHBIRType.FLOAT_CODEGEN]

    @staticmethod
    def model_type():
        return HHBIRType.FLOAT_CODEGEN

    def get_model(self):
        return self._curr_module

    def load_model(self, model_path):
        graph_path = os.path.join(model_path, self.graph_name)
        lib_path = os.path.join(model_path, self.lib_name)
        params_path = os.path.join(model_path, self.params_name)
        # ensure all needed files existing
        check_included_filename(model_path, self)

        graph = open(graph_path).read()
        params = bytearray(open(params_path, "rb").read())
        lib = runtime.load_module(lib_path)

        ctx = tvm.cpu(0)
        module = graph_runtime.create(graph, lib, ctx)
        module.load_params(params)

        self._curr_module = module

        if os.path.exists(os.path.join(model_path, self.info_file)):
            with open(os.path.join(model_path, self.info_file), "r") as f:
                self.info_dict = yaml.safe_load(f.read())

    def convert(self, input_module, board, opt_level):
        if (
            not isinstance(input_module, (list, tuple))
            and not isinstance(input_module[0], tvm.ir.module.IRModule)
            and not isinstance(input_module[1], dict)
        ):
            raise HHBException("input_module should be [IRModule, dict]")

        target = "llvm"
        model = relay.transform.DynamicToStatic()(input_module[0])
        with tvm.transform.PassContext(opt_level=opt_level):
            logger.info("building relay graph without quantization")
            self._curr_module = relay.build(model, target=target, params=input_module[1])

        (
            input_name_list,
            input_shape_list,
            input_dtype_list,
        ) = get_input_info_from_relay(input_module[0], input_module[1])
        output_shape_list, output_dtype_list = get_output_info_from_relay(
            input_module[0], input_module[1]
        )
        self.info_dict = {
            "input_name_list": input_name_list,
            "input_shape_list": input_shape_list,
            "input_dtype_list": input_dtype_list,
            "output_shape_list": output_shape_list,
            "output_dtype_list": output_dtype_list,
        }

    def save_model(self, model_path):
        """Save current module into files."""
        if not isinstance(self._curr_module, executor_factory.GraphExecutorFactoryModule):
            raise HHBException("need GraphExecutorFactoryModule")
        if not os.path.exists(model_path):
            raise HHBException("Directory {} is not exists".format(model_path))

        graph_path = os.path.join(model_path, self.graph_name)
        lib_path = os.path.join(model_path, self.lib_name)
        params_path = os.path.join(model_path, self.params_name)

        logger.info("write lib to %s", lib_path)
        self._curr_module.get_lib().export_library(lib_path)

        with open(graph_path, "w") as f:
            logger.info("write graph to %s", f.name)
            f.write(self._curr_module.get_graph_json())
        with open(params_path, "wb") as f:
            logger.info("write params to %s", f.name)
            f.write(relay.save_param_dict(self._curr_module.get_params()))

        if self.info_dict:
            info_path = os.path.join(model_path, self.info_file)
            logger.info("write extra info in %s" % info_path)
            with open(info_path, "w") as f:
                yaml.safe_dump(self.info_dict, f, default_flow_style=False)


@hhb_ir_helper
class HHBX86QnnCodegenIR(HHBIRBase):
    def __init__(self):
        super().__init__()
        self.lib_name = "codegen_x86_quant.so"
        self.lib_source_name = "lib0.c"
        self.params_name = "codegen_x86_quant.params"
        self.info_file = "codegen_x86_quant.yaml"

        self._curr_factory = None
        self._curr_module = None
        self.info_dict = {}

        self.all_included_filenames.extend([self.lib_name, self.lib_source_name, self.params_name])

    @staticmethod
    def name():
        return HHBIRType.TYPE2NAME[HHBIRType.X86_QNN_CODEGEN]

    @staticmethod
    def model_type():
        return HHBIRType.X86_QNN_CODEGEN

    def get_model(self):
        return self._curr_module

    def get_factory(self):
        return self._curr_factory

    def get_lib(self, output_dir):
        _, shl_dir, _, _, _ = find_base_path()
        include_path = os.path.join(shl_dir, "x86/include")
        ref_x86_dir = os.path.join(shl_dir, "x86/lib")

        lib_path = os.path.join(output_dir, "quant.so")
        kwargs = {}
        kwargs["options"] = [
            "-O2",
            "-g",
            "-I" + include_path,
            "-L" + ref_x86_dir,
            "-lshl",
        ]
        kwargs["cc"] = "gcc"
        lib = self.get_factory().get_lib()
        lib.export_library(lib_path, fcompile=False, workspace_dir=output_dir, **kwargs)
        lib = tvm.runtime.load_module(lib_path)
        return lib

    def load_model(self, model_path, module=None):
        raise HHBException("Unsupport x86 QNN model load")

    def set_quant_env(self, info_file):
        if not info_file:
            return
        with open(info_file, "r") as f:
            info_dict = yaml.safe_load(f.read())
            qconfig = qtz.qconfig(**info_dict["qnn_config"])
            with qconfig:
                logger.debug("restore quantization config from %s" % info_file)

    def get_quant_env(self, info_file):
        if not info_file:
            return
        with open(info_file, "r") as f:
            info_dict = yaml.safe_load(f.read())
        return info_dict["qnn_config"]

    def convert(self, input_module, board, opt_level, output_path, device_config):
        if (
            not isinstance(input_module, (list, tuple))
            and not isinstance(input_module[0], tvm.ir.module.IRModule)
            and not isinstance(input_module[1], dict)
            and input_module[1] is not None
        ):
            raise HHBException("input_module should be [IRModule, dict or None]")

        target = get_target(board)

        params_file = os.path.join(output_path, self.params_name)
        device_config["params_path"] = params_file

        mod = input_module[0]
        params = input_module[1]

        func = mod["main"]
        func = func.with_attr("global_symbol", tvm.runtime.container.String("csinn"))
        func = func.with_attr("Compiler", "csinn")
        mod["csinn_0"] = func

        logger.info("write params to %s", params_file)

        with tvm.transform.PassContext(
            opt_level=opt_level, config={"relay.ext.csinn.options": device_config}
        ):
            mod = csinn.partition_for_csinn(mod, params)
            factory = relay.build(mod, target=target, params=params)

        lib = factory.get_lib()
        self._curr_factory = factory
        self._curr_module = lib

        (
            input_name_list,
            input_shape_list,
            input_dtype_list,
        ) = get_input_info_from_relay(input_module[0], input_module[1])
        output_shape_list, output_dtype_list = get_output_info_from_relay(
            input_module[0], input_module[1]
        )
        self.info_dict = {
            "input_name_list": input_name_list,
            "input_shape_list": input_shape_list,
            "input_dtype_list": input_dtype_list,
            "output_shape_list": output_shape_list,
            "output_dtype_list": output_dtype_list,
        }

    def save_model(self, model_path):
        """Save current module into files."""
        lib_path = os.path.join(model_path, self.lib_name)
        kwargs = {}
        kwargs["options"] = ["-O0", "-g3"]
        logger.info("write lib to %s" % lib_path)
        self._curr_module.export_hhb_library(
            lib_path, fcompile=False, output_dir=model_path, **kwargs
        )

        info_path = os.path.join(model_path, self.info_file)
        logger.info("write extra info to %s" % info_path)
        with open(info_path, "w") as f:
            yaml.safe_dump(self.info_dict, f, default_flow_style=False)


def fcompile(filename, files, options=None):
    shutil.copy(files[0], filename)


@hhb_ir_helper
class HHBBoardQnnCodegenIR(HHBIRBase):
    def __init__(self, without_preprocess=False):
        super().__init__()
        self.mod = None
        self.params_name = "model.params"
        self.graph_info_name = "graph_info.bin"
        self.lib_source_name = "model.c"
        self.intrinsic_source_name = "intrinsic.c"
        self.main_source_name = "main.c"
        self.preprocess_source_name = "process.c"
        self.preprocess_header_name = "process.h"
        self.preio_source_name = "io.c"
        self.preio_header_name = "io.h"

        self.all_included_filenames.extend(
            [
                self.params_name,
                self.graph_info_name,
                self.lib_source_name,
                self.main_source_name,
                self.preprocess_source_name,
                self.preprocess_header_name,
            ]
        )

        if not without_preprocess:
            self.all_included_filenames.extend([self.preio_source_name, self.preio_header_name])

    @staticmethod
    def name():
        return HHBIRType.TYPE2NAME[HHBIRType.BOARD_QNN_CODEGEN]

    @staticmethod
    def model_type():
        return HHBIRType.BOARD_QNN_CODEGEN

    def get_model(self):
        return self._curr_module

    def load_model(self, model_path):
        pass

    def set_quant_env(self, info_file):
        if not info_file:
            return
        with open(info_file, "r") as f:
            info_dict = yaml.safe_load(f.read())
            qconfig = qtz.qconfig(**info_dict["qnn_config"])
            with qconfig:
                logger.debug("restore quantization config from %s" % info_file)

    def get_quant_env(self, info_file):
        if not info_file:
            return
        with open(info_file, "r") as f:
            info_dict = yaml.safe_load(f.read())
        return info_dict["qnn_config"]

    def convert(
        self,
        input_module,
        board,
        opt_level,
        output_path,
        device_config,
    ):
        mod, params = input_module
        self.mod = mod
        target = get_target(board)

        params_file = os.path.join(output_path, self.params_name)
        graph_info_file = os.path.join(output_path, self.graph_info_name)
        device_config["params_path"] = params_file
        device_config["graph_info_path"] = graph_info_file

        func = mod["main"]
        func = func.with_attr("global_symbol", tvm.runtime.container.String("csinn"))
        func = func.with_attr("Compiler", "csinn")
        mod["csinn_0"] = func

        logger.info("write params to %s", params_file)
        logger.debug("Generate trace data by {} strategy".format(device_config["trace_strategy"]))
        logger.debug(
            "Set the memory type of input tenosr as: {}".format(device_config["input_memory_type"])
        )
        logger.debug(
            "Set the memory type of output tenosr as: {}".format(
                device_config["output_memory_type"]
            )
        )

        if device_config["ahead_of_time"] == "intrinsic":
            with tvm.transform.PassContext(
                opt_level=3, config={"relay.ext.csinn.options": device_config}
            ):
                csinn_mod = csinn.partition_for_csinn(mod, params)
                factory = relay.build(csinn_mod, target=target, params=params)

            lib = factory.get_lib()
            self._curr_module = lib
            lib_path = os.path.join(output_path, self.intrinsic_source_name)

            logger.info("write lib source code to %s", lib_path)
            factory.export_library(lib_path, fcompile=fcompile)
            device_config["ahead_of_time"] = "unset"

        with tvm.transform.PassContext(
            opt_level=opt_level, config={"relay.ext.csinn.options": device_config}
        ):
            csinn_mod = csinn.partition_for_csinn(mod, params)
            factory = relay.build(csinn_mod, target=target, params=params)

        lib = factory.get_lib()
        self._curr_module = lib
        lib_path = os.path.join(output_path, self.lib_source_name)
        logger.info("write lib source code to %s", lib_path)
        factory.export_library(lib_path, fcompile=fcompile)

    def save_model(
        self,
        input_shape,
        output_shape,
        board,
        output_path,
        postprocess="top5",
        model_save="run_only",
        without_preprocess=False,
        preprocess_params=None,
        input_memory_type=None,
        q_scheme=None,
        codegen_config=None,
        hhb_gen=False,
    ):
        from .codegen_manage import (
            main_c_codegen,
            jit_c_codegen,
        )

        if board in ("th1520", "hth1520", "c920"):
            jit_c_codegen(
                self, input_shape, output_shape, board, output_path, preprocess_params, q_scheme
            )

        main_c_codegen(
            self,
            input_shape,
            output_shape,
            board,
            output_path,
            postprocess,
            model_save,
            without_preprocess,
            preprocess_params,
            input_memory_type,
            q_scheme,
            codegen_config.dynamic_shape,
            hhb_gen,
        )

        if codegen_config["with_py_wrapper"]:
            from .codegen_manage import get_execute_path

            exec_dir = get_execute_path()
            wrapper_dir = os.path.join(exec_dir, "config", "wrapper")

            shutil.copy(os.path.join(wrapper_dir, "model_client.py"), output_path)
            shutil.copy(os.path.join(wrapper_dir, "model_server.py"), output_path)
            shutil.copy(os.path.join(wrapper_dir, "model_wrapper.py"), output_path)

            target_path = os.path.join(wrapper_dir, "model_wrapper.tp")
            with open(target_path, "r") as f:
                target_data = f.read()
            if board in ("c906", "rvm", "c908", "c920", "c920v2"):
                tmp_data = "if (!output->is_const) {\n"
                tmp_data += " " * 4 + "shl_mem_free(output->data);\n"
                tmp_data += " " * 2 + "}"
                target_data = target_data.replace("#_hhb_wrapper_free_output_buf_#", tmp_data)
            else:
                target_data = target_data.replace("#_hhb_wrapper_free_output_buf_#", "")

            with open(os.path.join(output_path, "model_wrapper.c"), "w") as f:
                f.write(target_data)

        from .codegen_manage import package_sections

        package_sections(board, output_path, model_save)


def base_dir_exists(dir):
    found_dir = os.path.abspath(dir)
    found_dir = os.path.join(found_dir, "install_nn2/x86/include")
    if os.path.exists(found_dir) and os.path.isdir(found_dir):
        return found_dir
    else:
        return None


def find_base_path():
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        # for pyinstaller
        exe_dir = os.path.dirname(os.path.realpath(sys.executable))
    else:
        # for source file
        exe_dir = os.path.dirname(os.path.realpath(__file__))

    source_dir = os.path.join(exe_dir, "..", "..", "..")
    wheel_dir = os.path.join(exe_dir, "..")

    if base_dir_exists(exe_dir):
        # pyintaller
        base_found = exe_dir
        tvm_inc_dir = exe_dir + "/tvm/include/"
        dlpack_inc_dir = exe_dir + "/tvm/include/dlpack/"
        shl_dir = exe_dir + "/install_nn2/"
        prebuilt_dir = exe_dir + "/prebuilt/"
    elif base_dir_exists(source_dir):
        # source code
        base_found = source_dir
        tvm_inc_dir = source_dir + "/include/"
        dlpack_inc_dir = source_dir + "/3rdparty/dlpack/include/"
        shl_dir = source_dir + "/install_nn2/"
        prebuilt_dir = source_dir + "/thead/prebuilt/"
    elif base_dir_exists(wheel_dir):
        # wheel/pip
        base_found = wheel_dir
        tvm_inc_dir = wheel_dir + "/../tvm/include/"
        dlpack_inc_dir = wheel_dir + "/../tvm/dlpack/include/"
        shl_dir = wheel_dir + "/install_nn2/"
        prebuilt_dir = wheel_dir + "/prebuilt/"
    else:
        raise HHBException("Cannot find the executable base dir.\n")

    base_found = os.path.join(os.path.abspath(base_found), "")
    shl_dir = os.path.join(os.path.abspath(shl_dir), "")
    prebuilt_dir = os.path.join(os.path.abspath(prebuilt_dir), "")
    tvm_inc_dir = os.path.join(os.path.abspath(tvm_inc_dir), "")
    dlpack_inc_dir = os.path.join(os.path.abspath(dlpack_inc_dir), "")
    return base_found, shl_dir, prebuilt_dir, tvm_inc_dir, dlpack_inc_dir


@hhb_ir_helper
class HHBBoardBuildRuntime:
    def __init__(self, board, work_dir, intrinsic=False, link_lib="unset", android=False):
        self.board = board
        self.work_dir = work_dir
        self.intrinsic = intrinsic
        self.jit = False

        self.source_name = []
        self.runtime_name = ""
        self.jit_name = ""

        if board in ("th1520", "hth1520", "th1520_x86", "c920"):
            self.jit = True

        self.cflag = " -O2 -g "

        if board in ("c906", "c908", "c920", "c920v2", "rvm", "th1520", "hth1520"):
            self.compiler = "riscv64-unknown-linux-gnu-gcc"
            # check compiler version
            compiler_version = ensure_compiler(self.compiler)
            if compiler_version < 2.6:
                raise HHBException("Please upgrade compiler version.\n")

            self.cflag += "-mabi=lp64d "
        elif board in ("x86_ref", "th1520_x86"):
            self.compiler = "gcc"
        else:
            raise HHBException("Unsupport platform build: {}.\n".format(board))

        hhb_base_dir, shl_dir, prebuilt_dir, tvm_inc_dir, dlpack_dir = find_base_path()

        logger.info("HHB base dir: %s", hhb_base_dir)

        # all target have same include with x86
        self.include_dir = " -I" + shl_dir + "x86/include/ " + " -I" + dlpack_dir
        self.include_dir += " -I" + shl_dir + "x86/include/shl_public/ "
        self.include_dir += " -I" + shl_dir + "x86/include/csinn/ "

        self.include_dir += " -I" + tvm_inc_dir + " "

        self.link_flag = " -Wl,--gc-sections "
        # reuse cflag in link, but no -march
        self.link_flag += self.cflag

        if board in ("th1520", "hth1520"):
            self.link_flag += " -Wl,-unresolved-symbols=ignore-in-shared-libs "
            self.link_flag += " -L " + shl_dir + "th1520/lib/ -lshl "
        elif board == "th1520_x86":
            self.link_flag += " -Wl,-unresolved-symbols=ignore-in-shared-libs "
            self.link_flag += " -L " + shl_dir + "pnna_x86/lib/ -lshl_pnna_x86 "
        elif board == "rvm":
            self.link_flag += " -L " + shl_dir + "rvm/lib/ -lshl_rvm -static "
        elif board == "c906":
            self.link_flag += " -L " + shl_dir + "c906/lib/ -lshl -static "
            self.cflag += " -march=rv64gcv0p7_zfh_xtheadc "
        elif board == "c908":
            self.link_flag += " -L " + shl_dir + "c908/lib/ -lshl -static "
        elif board == "c920":
            if link_lib in ("unset", "shl_c920"):
                self.link_flag += " -L " + shl_dir + "c920/lib/ -lshl -fopenmp -static "
            elif link_lib == "shl_th1520":
                self.link_flag += " -Wl,-unresolved-symbols=ignore-in-shared-libs "
                self.link_flag += " -L " + shl_dir + "th1520/lib/ -lshl "
            else:
                raise HHBException("Unsupport link {} for c920.\n".format(link_lib))
            self.cflag += " -march=rv64gcv0p7_zfh_xtheadc "
        elif board == "c920v2":
            self.link_flag += " -L " + shl_dir + "c920v2/lib/ -lshl -static "
        elif board == "x86_ref":
            self.link_flag += " -L " + shl_dir + "x86/lib/ -lshl "
        else:
            raise HHBException("Unsupport platform build: {}.\n".format(board))

        self.include_dir += " -I " + prebuilt_dir + "runtime/cmd_parse"
        if android:
            runtime_dir = " -L " + prebuilt_dir + "runtime/riscv_android"
        else:
            if board in ("c906", "c908", "c920", "c920v2", "rvm", "th1520", "hth1520"):
                decode_dir = " -L " + prebuilt_dir + "decode/install/lib/rv"
                runtime_dir = " -L " + prebuilt_dir + "runtime/riscv_linux"
            else:
                decode_dir = " -L " + prebuilt_dir + "decode/install/lib/x86"
                runtime_dir = " -L " + prebuilt_dir + "runtime/x86_linux"

        if board in ("c906", "c908", "c920", "c920v2", "rvm", "th1520", "hth1520", "th1520_x86"):
            self.link_flag += (
                decode_dir + runtime_dir + " -lprebuilt_runtime -ljpeg -lpng -lz -lstdc++ -lm "
            )
        else:
            self.link_flag += decode_dir + runtime_dir + " -lprebuilt_runtime -fopenmp -lm -lstdc++"

    def prefix_path(self, filename):
        return " " + self.work_dir + "/" + filename + " "

    def csource_command_line(self, source_name):
        cmd_line = (
            self.compiler
            + self.cflag
            + self.include_dir
            + " -I"
            + self.work_dir
            + self.prefix_path(source_name + ".c")
            + " -c -o "
            + self.prefix_path(source_name + ".o")
        )
        self.source_name.append(source_name)
        return cmd_line

    def build_c(self):
        cmd = self.csource_command_line("main")
        logger.info(cmd)
        os.system(cmd)
        cmd = self.csource_command_line("model")
        logger.info(cmd)
        os.system(cmd)

        if self.jit:
            cmd = self.csource_command_line("jit")
            logger.info(cmd)
            os.system(cmd)

        if self.intrinsic:
            cmd = self.csource_command_line("intrinsic")
            logger.info(cmd)
            os.system(cmd)

    def link_elf(self, runtime_name="hhb_runtime", jit_name="hhb_jit"):
        self.runtime_name = runtime_name
        self.jit_name = jit_name
        cmd_line = (
            self.compiler
            + self.prefix_path("model.o")
            + self.prefix_path("main.o")
            + " -o "
            + self.prefix_path(runtime_name)
        )
        if self.intrinsic:
            cmd_line += self.prefix_path("intrinsic.o")

        cmd_line += self.link_flag
        logger.info(cmd_line)
        os.system(cmd_line)

        if self.jit:
            cmd_line = (
                self.compiler
                + self.prefix_path("model.o")
                + self.prefix_path("jit.o")
                + " -o "
                + self.prefix_path(jit_name)
            )
            cmd_line += self.link_flag
            logger.info(cmd_line)
            os.system(cmd_line)

    def generate_makefile(self):
        """Generate corresponding makefile."""
        from .codegen_manage import get_execute_path

        exec_dir = get_execute_path()
        tp_path = os.path.join(exec_dir, "config", "template", "makefile.tp")

        #######################################################################
        #
        # Generate compiler
        #
        with open(tp_path, "r") as f:
            makefile_data = f.read()
        makefile_data = makefile_data.replace("#_hhb_makefile_compiler_#", self.compiler)

        #######################################################################
        #
        # Generate include
        #
        self.include_dir += " -I."
        include_data = ""
        if self.include_dir:
            include_data = self.include_dir.strip().split(" ")
            include_data = [i for i in include_data if i]
            include_data = " \\\n\t".join(include_data)
        makefile_data = makefile_data.replace("#_hhb_makefile_include_#", include_data)

        #######################################################################
        #
        # Generate cflags
        #
        makefile_data = makefile_data.replace("#_hhb_makefile_cflag_#", self.cflag + "${INCLUDE}")

        #######################################################################
        #
        # Generate ldflags
        #
        ld_data = self.link_flag.strip().split(" ")
        ld_data = list(filter(lambda x: x, ld_data))
        ld_data_p1 = []
        ld_data_p2 = []
        idx = 0
        while idx < len(ld_data):
            if ld_data[idx] == "-L":
                ld_data_p1.extend([ld_data[idx], ld_data[idx + 1]])
                idx += 2
            else:
                ld_data_p2.append(ld_data[idx])
                idx += 1
        p1_str = ""
        for i in range(len(ld_data_p1) // 2):
            p1_str += ld_data_p1[2 * i] + ld_data_p1[2 * i + 1] + " \\\n\t"
        makefile_data = makefile_data.replace("#_hhb_makefile_ldflag1_#", p1_str)
        p2_str = " ".join(ld_data_p2)
        makefile_data = makefile_data.replace("#_hhb_makefile_ldflag2_#", p2_str)

        #######################################################################
        #
        # Generate objects
        #
        obj_data = ""
        if self.source_name:
            for sf in self.source_name:
                curr_obj = f"{sf}.o: {sf}.c\n"
                curr_obj += "\t$(CC) $(CFLAGS) -c -o $@  $^\n"

                obj_data += curr_obj + "\n"
        obj_data = obj_data.rstrip()
        makefile_data = makefile_data.replace("#_hhb_makefile_compile_obj_#", obj_data)

        #######################################################################
        #
        # Generate elf
        #
        elf_data = f"{self.runtime_name}: main.o model.o"
        if self.intrinsic:
            elf_data += " intrinsic.o"
        elf_data += "\n"
        elf_data += "\t$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)\n"

        if self.jit:
            elf_data += "\n"
            elf_data += f"{self.jit_name}: jit.o model.o\n"
            elf_data += "\t$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)\n"
        elf_data = elf_data.rstrip()
        makefile_data = makefile_data.replace("#_hhb_makefile_elf_#", elf_data)

        #######################################################################
        #
        # Generate clean
        #
        clean_data = f"-rm *.o {self.runtime_name}"
        if self.jit:
            clean_data += f" {self.jit_name}"
        makefile_data = makefile_data.replace("#_hhb_makefile_clean_#", clean_data)

        #######################################################################
        #
        # Generate python wrapper
        #
        wrapper_data = ""
        if (
            os.path.exists(os.path.join(self.work_dir, "model_wrapper.c"))
            and self.board != "th1520_x86"
        ):
            wrapper_data = "libmodel_wrapper.so: model_wrapper.c model.c io.c\n"
            wrapper_data += "\t$(CC) $(CFLAGS) -fPIC -shared -o $@ $^ $(LDFLAGS)\n"
        makefile_data = makefile_data.replace("#_hhb_makefile_py_wrapper_#", wrapper_data)

        #######################################################################
        #
        # Generate target
        #
        target_data = self.runtime_name
        if self.jit:
            target_data += " " + self.jit_name
        makefile_data = makefile_data.replace("#_hhb_makefile_default_target_#", target_data)

        #######################################################################
        #
        # Generate simulation exectution
        #
        sim_exec_data = ""
        if self.board in ("x86_ref", "c906", "c908", "c920", "c920v2", "th1520_x86"):
            if self.board == "th1520_x86":
                _, shl_dir, _, _, _ = find_base_path()
                sim_exec_data += "NNA_DDK_DIR ?=\n"
                sim_exec_data += f"export LD_LIBRARY_PATH=${{NNA_DDK_DIR}}/x86:{os.path.join(shl_dir, 'pnna_x86', 'lib')}\n"
                sim_exec_data += "export SIM_VISION_PATH=${NNA_DDK_DIR}/x86/sim_nna.so\n"
            sim_exec_data += f"run_sim: {self.runtime_name}\n"
            input_data = glob.glob(os.path.join(self.work_dir, "*.bin"))
            input_data = list(map(lambda x: os.path.basename(x), input_data))
            if "graph_info.bin" in input_data:
                input_data.remove("graph_info.bin")
            if not input_data:
                input_data = ["data.0.bin"]
            input_data = sorted(input_data)

            exec_prefix = ""
            if self.board == "c906":
                exec_prefix += "qemu-riscv64 -cpu c906fdv "
            elif self.board == "c908":
                exec_prefix += "qemu-riscv64 -cpu c908v "
            elif self.board == "c920":
                exec_prefix += "qemu-riscv64 -cpu c920 "
            elif self.board == "c920v2":
                exec_prefix += "qemu-riscv64 -cpu c920v2 "
            sim_exec_data += f"\t{exec_prefix}./{self.runtime_name} hhb.bm {' '.join(input_data)}\n"

            if self.board == "th1520_x86" and self.jit:
                sim_exec_data += f"\nrun_jit: {self.jit_name}\n"
                sim_exec_data += f"\t./{self.jit_name} hhb.bm\n"
        makefile_data = makefile_data.replace("#_hhb_makefile_run_sim_#", sim_exec_data)

        target_path = os.path.join(self.work_dir, f"makefile.{self.board}")
        with open(target_path, "w") as f:
            f.write(makefile_data)


def guess_ir_type(file_path):
    """
    Get IR type according to the files in file_path

    Parameters
    ----------
    file_path : str
        The directroy that hold module files

    Returns
    -------
    res : HHBIRType
        Inferred IR type
    """
    hhb_ir_type = []
    for h in HHB_IR:
        if h == HHBBoardBuildRuntime:
            continue
        hhb_ir_obj = h()
        hhb_ir_type.append(int(hhb_ir_obj.check_ir_type(file_path)))

    if sum(hhb_ir_type) == 0:
        return HHBIRType.UNKNOWN
    elif sum(hhb_ir_type) == 1:
        idx = hhb_ir_type.index(1)
        return HHB_IR[idx].model_type()
    else:
        raise HHBException("There are more than one hhb ir detected...")
