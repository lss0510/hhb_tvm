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

import pytest
import tempfile
import os
import json

import numpy as np

import hhb
import tvm
from tvm import relay


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HHB_ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../"))
SIMULATE_DATA = os.path.join(HHB_ROOT_DIR, "tests/thead/images/n01440764_188.JPEG")
AiBench_DIR = os.path.join(HHB_ROOT_DIR, "../hhb_test/scripts/ai-bench")


def test_Config_attribute():
    config = hhb.Config()
    assert config.optimize.board.value == "unset"
    assert config.quantize.quantization_scheme.value == "uint8_asym"

    config = hhb.Config("x86_ref")
    assert config.optimize.board.value == "x86_ref"
    assert config.quantize.quantization_scheme.value == "int8_asym"

    with pytest.raises(Exception) as e:
        hhb.Config("arm")


def test_Config_update_config_from_module():
    x = relay.var("x", relay.TensorType((1, 3, 224, 224), "float32"))
    z = relay.nn.relu(x)
    func = relay.Function([x], z)
    mod = tvm.ir.IRModule.from_expr(func)

    hhb_ir = hhb.Compiler().create_relay_ir()
    hhb_ir.set_model(mod, None)

    config = hhb.Config()
    assert config.import_config.input_name.value is None
    assert config.import_config.input_shape.value is None
    assert config.import_config.output_name.value is None

    config.update_config_from_module(hhb_ir)
    assert tuple(config.import_config.input_name.value) == ("x",)
    assert tuple(config.import_config.input_shape.value[0]) == (1, 3, 224, 224)
    assert tuple(config.import_config.output_name.value) == ("output0",)


def test_Config_update_config_from_file():
    tmp_dir = tempfile.gettempdir()
    right_config = {
        "preprocess": {
            "data_scale": 2.0,
            "pixel_format": "BGR",
        }
    }
    j_path = os.path.join(tmp_dir, "right_config.json")
    with open(j_path, "w") as f:
        json.dump(right_config, f, indent=2)

    config = hhb.Config()
    assert config.preprocess.data_scale.value == 1
    assert config.preprocess.pixel_format.value == "RGB"
    config.update_config_from_file(j_path)
    assert config.preprocess.data_scale.value == 2
    assert config.preprocess.pixel_format.value == "BGR"


def test_Config_generate_cmd_config():
    config = hhb.Config()
    with pytest.raises(Exception) as e:
        config.generate_cmd_config()
    config.import_config.input_name.value = ["input"]
    config.import_config.input_shape.value = [[1, 3]]
    config.import_config.output_name.value = ["output"]
    config.generate_cmd_config()
    assert config._cmd_config is not None
    assert config._cmd_config.preprocess_config.data_scale == 1
    assert config._cmd_config.quantize_config.quantization_scheme == "uint8_asym"


def test_Compiler_simulate():
    compiler = hhb.Compiler("x86_ref")
    model_file = os.path.join(AiBench_DIR, "net/onnx/mobilenet/mobilenetv1.onnx")
    input_name = ["data_input"]
    input_shape = [[1, 3, 224, 224]]
    output_name = ["prob_Y"]

    compiler.import_model(
        model_file,
        input_name=input_name,
        input_shape=input_shape,
        output_name=output_name,
    )

    compiler.config.preprocess.data_mean.value = [103.94, 116.78, 123.68]
    compiler.config.preprocess.data_scale.value = 0.0170068027211
    compiler.config.preprocess.pixel_format.value = "BGR"
    tmp_dir = tempfile.gettempdir()
    compiler.config.common.output.value = os.path.join(tmp_dir, "mv1_out")

    dataset_list = compiler.preprocess(SIMULATE_DATA)

    compiler.quantize(dataset_list)
    compiler.codegen()
    compiler.create_executor()

    for data in dataset_list:
        output = compiler.inference(data)
        assert np.argmax(output[0]) == 0


def test_Compiler_codegen():
    compiler = hhb.Compiler("th1520")
    model_file = os.path.join(AiBench_DIR, "net/onnx/mobilenet/mobilenetv1.onnx")
    input_name = ["data_input"]
    input_shape = [[1, 3, 224, 224]]
    output_name = ["prob_Y"]

    compiler.import_model(
        model_file,
        input_name=input_name,
        input_shape=input_shape,
        output_name=output_name,
    )
    compiler.config.preprocess.data_mean.value = [103.94, 116.78, 123.68]
    compiler.config.preprocess.data_scale.value = 0.0170068027211
    compiler.config.preprocess.pixel_format.value = "BGR"
    tmp_dir = tempfile.gettempdir()
    compiler.config.common.output.value = os.path.join(tmp_dir, "mv1_codegen_out")

    dataset_list = compiler.preprocess(SIMULATE_DATA)

    compiler.quantize(dataset_list)
    compiler.codegen()
    result = os.listdir(compiler.config.common.output.value)
    assert "model.c" in result
    assert "hhb.bm" in result


if __name__ == "__main__":
    test_Config_attribute()
    test_Config_update_config_from_module()
    test_Config_update_config_from_file()
    test_Config_generate_cmd_config()
    test_Compiler_simulate()
    test_Compiler_codegen()
