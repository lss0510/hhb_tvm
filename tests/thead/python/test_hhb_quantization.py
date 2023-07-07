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

"""HHB quantization serialization test cases"""
import numpy as np

import tvm


def calculate_quant_by_hhb(init_params, quant_scheme):
    from tvm.relay.backend.contrib.csinn_backend import QnnConfig, QuantCalculator
    import hhb
    from hhb.core.quantization_manage import get_config_dict

    comp = hhb.Compiler("x86_ref")
    comp.config.quantize.quantization_scheme.value = quant_scheme
    comp.config.import_config.input_shape.value = "1 2 3"
    comp.config.import_config.output_name.value = "out"
    comp.config.generate_cmd_config()
    cmd_config = get_config_dict(comp.config._cmd_config)
    cmd_config["target"] = comp.config._cmd_config.board

    scale_zp = []
    with tvm.transform.PassContext(opt_level=3, config={"relay.ext.csinn.options": cmd_config}):
        q = QnnConfig()
        qc = QuantCalculator(q)

        quant_res = qc.get_quant_params(init_params, "input")

        assert len(quant_res.qinfo) == quant_res.q_size
        for qinfo in quant_res.qinfo:
            scale = float(qinfo.scale)
            zp = float(qinfo.zero_point)
            scale_zp.append([scale, zp])
    return np.array(scale_zp, dtype=np.float32)


def calculate_quant_by_numpy(init_params, quant_scheme):
    def _cal_scale_zp(min_value, max_value, quant_scheme):
        if quant_scheme == "uint8_asym":
            range = np.array(255, dtype=np.float32)
            min_value = min(min_value, 0.0)
            min_value = np.array(min_value, dtype=np.float32)
            max_value = max(max_value, 0.0)
            max_value = np.array(max_value, dtype=np.float32)

            scale = (max_value - min_value) / range

            if np.isclose(scale, 0.0):
                scale = np.abs(max_value)
            zp = np.round(-min_value / scale)
            zp = zp.tolist()
            zp = max(zp, 0.0)
            zp = min(zp, range)

        return [scale, float(zp)]

    scale_zp = []

    min_max = init_params[3:]
    assert len(min_max) % 2 == 0
    for i in range(len(min_max) // 2):
        scale_zp.append(_cal_scale_zp(min_max[2 * i], min_max[2 * i + 1], quant_scheme))
    return np.array(scale_zp, dtype=np.float32)


def verify_results(init_params, quant_scheme, rtol=1e-7, atol=0):
    hhb_results = calculate_quant_by_hhb(init_params, quant_scheme)
    numpy_results = calculate_quant_by_numpy(init_params, quant_scheme)

    for relay_res, onnx_res in zip(numpy_results, hhb_results):
        np.testing.assert_allclose(relay_res, onnx_res, rtol=rtol, atol=atol)


def test_pertensor():
    def _test(min_max, quant_scheme):
        init_params = [1, 0, 0] + min_max
        verify_results(init_params, quant_scheme, rtol=1e-5, atol=1e-5)

    _test([-10.0, 10.0], "uint8_asym")
    _test([-10.0, -1.0], "uint8_asym")
    _test([1.0, 10.0], "uint8_asym")


if __name__ == "__main__":
    test_pertensor()
