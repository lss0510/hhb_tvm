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
"""FFI APIs for CSINN api."""
import numpy as np

import tvm
from tvm.runtime import Object
from . import _csinn


def collect_quant_info():
    quant = _csinn.collect_quant_info()
    return quant


def emit_binary_model(path, section_map):
    return _csinn.emit_binary_model(path, section_map)


class QnnConfig(Object):
    """Get quantization config which is defined in CodegenCSINN::cfg at
    relay/contrib/csinn/csinn.h.
    It is a QConfig_ object that holds some quant info.

    Note: we must to execute QnnConfig() under the passcontext, due to depend on
    relay.ext.csinn.options:
    Examples
    --------
    .. code-block:: python

        with tvm.transform.PassContext(opt_level=3, config={"relay.ext.csinn.options": cmd_config}):
            q = QnnConfig()
            qc = QuantCalculator(q)
            ...
    """

    def __init__(self) -> None:
        self.__init_handle_by_constructor__(_csinn.QnnConfig)


class QuantCalculator(object):
    """Class to calculate quantization information. It is the wrapper for QuantCalculator in
    C++(relay/backend/contrib/csinn/quant_cal.h)"""

    def __init__(self, qnnconfig) -> None:
        """Set qnnconfig.

        Parameters
        ----------
        qnnconfig : QnnConfig
            Quantization config that can be created by QnnConfig()
        """
        self.qnnconfig = qnnconfig

    def get_quant_params(self, q_params, const_kind):
        """Calulate quantizaiton information according to qnnconfig.

        Parameters
        ----------
        q_params : list[]
            Initial quantization information: [tensor_type, USE_MINMAX, q_type, min, max, ...] or
            [tensor_type, USE_SCALE, q_type, scale, zp, ...]

            tensor_type: int, ACTIVATE(1) or WEIGHT(0),
            q_type: int, PER_TENSOR(0) or PER_CHANNEL(1)

        const_kind : str
            Choose from ["conv_kernel", "depthwise_kernel", "conv_bias", "depthwise_bias"].
            It is mainly used in per-channel quantization.

        Returns
        -------
            quant_params : QuantParams
                Qauntization info after calculation, which is defined in QuantParams at
                relay/backend/contrib/csinn/format.h
        """
        return _csinn.GetQuantParams([q_params], self.qnnconfig, const_kind)

    def _convert_and_check_data(self, data):
        """Get data with tvm.runtime.ndarray.NDArray type"""
        new_data = data
        if not isinstance(data, (np.ndarray, tvm.runtime.ndarray.NDArray)):
            try:
                new_data = np.array(data, dtype=np.float32)
            except:
                raise ValueError("data cannot be converted to np.float32!")
        if isinstance(new_data, np.ndarray):
            tvm_data = tvm.runtime.ndarray.empty(new_data.shape, new_data.dtype)
            tvm_data.copyfrom(new_data)
            new_data = tvm_data
        assert isinstance(
            new_data, tvm.runtime.ndarray.NDArray
        ), f"need NDArray but get {type(new_data)}"
        assert new_data.dtype == "float32", f"only support for float32, but get {new_data.dtype}"

        return new_data

    def quantize_weight(self, data, quant_params, depthwise_kernel=False):
        """Quantize weight data into specified type.

        Parameters
        ----------
        data : list or np.ndarray or tvm.runtime.ndarray.NDArray
            Original fp32 data.

        quant_params : QuantParams
            The quantizaiton parameters for data

        depthwise_kernel : Optional[bool]
            The weight data is whether depthwise kernel

        Returns
        -------
        quantized_data : tvm.runtime.ndarray.NDArray
            The quantized data
        """
        data = self._convert_and_check_data(data)
        return _csinn.QuantizeWeight(data, quant_params.dtype, quant_params, depthwise_kernel)

    def quantize_bias(self, data, target_dtype, input_quant_params, weight_quant_params):
        """Quantize bias data into specified type.

        Parameters
        ----------
        data : list or np.ndarray or tvm.runtime.ndarray.NDArray
            Original fp32 data.

        target_dtype : str
            The dtype that data will be quantized.

        input_quant_params : QuantParams
            The quantizaiton parameters for the input of conv

        weight_quant_params : QuantParams
            The quantizaiton parameters for the weight of conv

        Returns
        -------
        quantized_data : tvm.runtime.ndarray.NDArray
            The quantized data
        """
        data = self._convert_and_check_data(data)
        return _csinn.QuantizeBias(data, target_dtype, input_quant_params, weight_quant_params)


def llm_quantize_block_32(data, dim_count, dim, mtype):
    """_summary_

    Args:
        data (_type_): _description_
        dim_count (_type_): _description_
        dim (_type_): _description_
        mtype (_type_): _description_

    Returns:
        _type_: _description_
    """
    return _csinn.llm_quantize_block_32(data, dim_count, dim, mtype)
