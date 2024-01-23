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
# pylint: disable=no-else-return, inconsistent-return-statements, no-else-raise
""" llm import """
import os
import sys
import json
import ctypes
import numpy as np
from enum import Enum
from transformers import AutoModel, AutoModelForCausalLM
import logging
from tvm.relay.backend.contrib.csinn_backend import llm_quantize_block_32

logger = logging.getLogger("HHB")
LOG = 25


class csinn_dtype_enum(Enum):
    CSINN_DTYPE_BOOL = 0  # < Boolean
    CSINN_DTYPE_INT4 = 1  # < Signed 4 bit fixed-point
    CSINN_DTYPE_UINT8 = 2  # < Unsigned 8 bit fixed-point
    CSINN_DTYPE_INT8 = 3  # < Signed 8 bit fixed-point
    CSINN_DTYPE_UINT16 = 4  # < Unsigned 16 bit fixed-point
    CSINN_DTYPE_INT16 = 5  # < Signed 16 bit fixed-point
    CSINN_DTYPE_UINT32 = 6  # < Unsigned 32 bit fixed-point
    CSINN_DTYPE_INT32 = 7  # < Signed 32 bit fixed-point
    CSINN_DTYPE_FLOAT16 = 8  # < Half-precision floating-point
    CSINN_DTYPE_BFLOAT16 = 9  # < Brain floating-point
    CSINN_DTYPE_FLOAT32 = 10  # < Single-precision floating-point
    CSINN_DTYPE_FLOAT64 = 11  # < Double-precision floating-point
    CSINN_DTYPE_INT64 = 12  # < Signed 64 bit fixed-point
    CSINN_DTYPE_SIZE = 13


#  CSI-NN data memory type
class csinn_mem_type_enum(Enum):
    CSINN_MEM_TYPE_CPU_NOT_ALIGNED = 0  # < Default storage
    CSINN_MEM_TYPE_CPU_ALIGNED = 1  # < Aligned storage
    CSINN_MEM_TYPE_DMABUF = 2  # < DMA buf
    CSINN_MEM_TYPE_ASP42 = 3  # < Structed sparsity 4:2
    CSINN_MEM_TYPE_ASP41 = 4  # < Structed sparsity 4:1
    # < Accelerator driver or others alloced CPU memory
    CSINN_MEM_TYPE_CPU_ACC = 5
    CSINN_MEM_TYPE_BLOCK_Q2_K = 6  # < Block quantization from llama.cpp
    CSINN_MEM_TYPE_BLOCK_Q4_0 = 7  # < Block quantization from llama.cpp
    CSINN_MEM_TYPE_BLOCK_Q8_0 = 8  # < Block quantization from llama.cpp


StrToSHLDtype = {
    "int4": csinn_dtype_enum.CSINN_DTYPE_INT4.value,
    "int8": csinn_dtype_enum.CSINN_DTYPE_INT8.value,
    "uint16": csinn_dtype_enum.CSINN_DTYPE_UINT16.value,
    "int16": csinn_dtype_enum.CSINN_DTYPE_INT16.value,
    "uint32": csinn_dtype_enum.CSINN_DTYPE_UINT32.value,
    "int32": csinn_dtype_enum.CSINN_DTYPE_INT32.value,
    "float16": csinn_dtype_enum.CSINN_DTYPE_FLOAT16.value,
    "bfloat16": csinn_dtype_enum.CSINN_DTYPE_BFLOAT16.value,
    "float32": csinn_dtype_enum.CSINN_DTYPE_FLOAT32.value,
    "int64": csinn_dtype_enum.CSINN_DTYPE_INT64.value,
}


def convert_model_to_json(model, config=None, save_dir="hhb_out"):
    """_summary_

    Args:
        model (Pytorch model): origin model.
        config (json): config json file of LLM.
        save_dir (str, optional): _description_. Defaults to "hhb_out".

    Returns:
        _type_: _description_
    """

    class ModelConverter:
        """_summary_"""

        def __init__(self, model_dict, config, save_path) -> None:
            """ """
            self.model_dict = model_dict
            self.config = config
            self.save_path = save_path

        def _get_layer_name(self, layer_name):

            """
            Supported model: chatglm, llama2, Qwen

            """
            name_dict = {
                "embd_weight": ["embedding", "embed_tokens", "wte"],
                "output_norm": ["final_layernorm", "model.norm", "ln_f"],
                "output_layer": ["output_layer", "lm_head", "lm_head"],
            }
            for key, val in name_dict.items():
                for name in val:
                    if name in layer_name:
                        return key
            layer_numer = layer_name.split(".")
            for d in layer_numer:
                if d.isdigit():
                    return d
            return None

        def convert_to_json(self):
            bin_file_path = "/".join([self.save_path, "shl_llm_weight.bin"])
            json_file_path = "/".join([self.save_path, "shl_llm_weight.json"])
            data_offset = 0
            content = {}
            model = {}
            tensor_none = {}
            layers = []
            if self.config.get("num_layers") is None:
                if self.config.get("num_hidden_layers") is not None:
                    self.config["num_layers"] = self.config["num_hidden_layers"]
                else:
                    logger.error("num_layers is required")
                    sys.exit(0)
            for i in range(self.config["num_layers"]):
                layers.append(tensor_none.copy())

            with open(bin_file_path, "wb") as weight_file:
                for key, value in self.model_dict.items():
                    logger.log(LOG, "Convert layer: {}".format(key))
                    name = self._get_layer_name(key)
                    if name == None:
                        logger.warning("Find layer: {} connet be converted!".format(key))
                        continue
                    data = value.to("cpu").numpy()
                    tensor = {}
                    tensor["data_offset"] = data_offset
                    dims = {}
                    for i in range(value.ndim):
                        dims[str(i)] = value.shape[i]
                    tensor["dim"] = dims
                    tensor["dim_count"] = value.ndim
                    tensor["dtype"] = StrToSHLDtype[str(data.dtype)]
                    tensor["mtype"] = csinn_mem_type_enum.CSINN_MEM_TYPE_CPU_NOT_ALIGNED.value
                    tensor["name"] = key
                    bytes_to_save = data.tobytes()
                    weight_file.write(bytes_to_save)
                    data_offset += data.nbytes
                    if name.isdigit():
                        layers[int(name)][key] = tensor
                    elif name:
                        model[name] = tensor

            logger.log(LOG, "data_offset:{}".format(data_offset))
            content["config"] = {}
            del self.config["_name_or_path"]
            content["config"]["model_params"] = self.config
            content["config"]["shl_params"] = {}
            content["config"]["shl_params"]["shl_model_type"] = "weight_only"
            model["layer"] = layers
            model["layers_num"] = self.config["num_layers"]
            content["model"] = model
            out_file = open(json_file_path, "w")
            json.dump(content, out_file, indent=4)

    logger.log(LOG, "Convert model to json file")
    model_converter = ModelConverter(model.state_dict(), config.to_dict(), save_dir)
    model_converter.convert_to_json()
    logger.log(LOG, "Convert end...")


def llm_import(name_or_dir, save_dir):
    """import LLM from format {float32, float16, int8, int4}

    Args:
        name_or_dir (list, str): The path of origin model.
        save_dir (str): The save path of converted model.
    """

    if isinstance(name_or_dir, list):
        name_or_dir = name_or_dir[0]

    if not isinstance(save_dir, str):
        logger.warning("save_path only support type string, not {}".format(type(save_dir)))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    try:
        model = AutoModelForCausalLM.from_pretrained(name_or_dir, trust_remote_code=True)
    except ValueError:
        model = AutoModel.from_pretrained(name_or_dir, trust_remote_code=True)
    convert_model_to_json(model, model.config, save_dir)


def base_quantize(model, config, quantization_scheme, save_dir="hhb_out"):
    """parse llm, convert model to json

    Args:
        model (_type_): _description_
        config (_type_, optional): _description_. Defaults to None.
        quantization_scheme (str, optional): _description_. Defaults to "float32".
        save_path (str, optional): _description_. Defaults to "hhb_out".

    Returns:
        _type_: _description_
    """

    class Quantizer:
        """_summary_"""

        def __init__(self, model_dict, config, qtype, save_path) -> None:
            """ """
            self.model_dict = model_dict
            self.config = config
            self.qtype = qtype
            self.save_path = save_path

        def _get_layer_name(self, layer_name):

            """
            Supported model: chatglm, llama2, Qwen

            """
            name_dict = {
                "embd_weight": ["embedding", "embed_tokens", "wte"],
                "output_norm": ["final_layernorm", "model.norm", "ln_f"],
                "output_layer": ["output_layer", "lm_head", "lm_head"],
            }
            for key, val in name_dict.items():
                for name in val:
                    if name in layer_name:
                        return key
            layer_numer = layer_name.split(".")
            for d in layer_numer:
                if d.isdigit():
                    return d
            return None

        def quantize_block_32(self, weight_file, data, mtype):
            """_summary_

            Args:
                weight_file (_io.BufferedWriter): bin_file of quantized weight, scale, zero_point
                data (torch.Tensor, numpy.ndarray): float weight
                mtype (csinn_mem_type_enum): _description_

            Returns:
                int : length of quantized weight, scale, zero_point
            """
            if isinstance(type(data), np.ndarray):
                data = data.numpy()
            dim_count = data.ndim
            data_ptr = data.ctypes.data_as(ctypes.c_void_p)
            dim = np.array([data.shape[i] for i in range(dim_count)], dtype=np.int32)
            dim_ptr = dim.ctypes.data_as(ctypes.c_void_p)
            result = llm_quantize_block_32(data_ptr, dim_count, dim_ptr, mtype)
            weight_file.write(result)
            return len(result)

        def from_pretrained(self):
            bin_file_path = "/".join([self.save_path, "shl_llm_weight_quantize.bin"])
            json_file_path = "/".join([self.save_path, "shl_llm_weight_quantize.json"])

            data_offset = 0
            content = {}
            model = {}
            tensor_none = {}
            layers = []
            if self.config.get("num_layers") is None:
                if self.config.get("num_hidden_layers") is not None:
                    self.config["num_layers"] = self.config["num_hidden_layers"]
                else:
                    logger.error("num_layers is must")
                    sys.exit(0)
            for i in range(self.config["num_layers"]):
                layers.append(tensor_none.copy())

            with open(bin_file_path, "wb") as weight_file:
                for key, value in self.model_dict.items():
                    logger.log(LOG, "Quantize layer: {}".format(key))
                    name = self._get_layer_name(key)
                    if name == None:
                        logger.warning("Find layer: {} connet be quantized!".format(key))
                        continue
                    tensor = {}
                    tensor["data_offset"] = data_offset
                    dims = {}
                    for i in range(value.ndim):
                        dims[str(i)] = value.shape[i]
                    tensor["dim"] = dims
                    tensor["dim_count"] = value.ndim
                    data = value.to("cpu").numpy()
                    if name in ["embd_weight", "output_layer"]:
                        if self.qtype == "q8_0":
                            offset = self.quantize_block_32(
                                weight_file,
                                data,
                                csinn_mem_type_enum.CSINN_MEM_TYPE_BLOCK_Q8_0.value,
                            )
                            tensor["dtype"] = csinn_dtype_enum.CSINN_DTYPE_INT8.value
                            tensor["mtype"] = csinn_mem_type_enum.CSINN_MEM_TYPE_BLOCK_Q8_0.value
                        elif self.qtype == "q4_0":
                            offset = self.quantize_block_32(
                                weight_file,
                                data,
                                csinn_mem_type_enum.CSINN_MEM_TYPE_BLOCK_Q4_0.value,
                            )
                            tensor["dtype"] = csinn_dtype_enum.CSINN_DTYPE_INT4.value
                            tensor["mtype"] = csinn_mem_type_enum.CSINN_MEM_TYPE_BLOCK_Q4_0.value
                        else:
                            logger.error("Unsupported quantization scheme")
                            sys.exit()
                        tensor["name"] = key
                        model[name] = tensor
                    elif name == "output_norm":
                        tensor["dtype"] = StrToSHLDtype[str(data.dtype)]
                        tensor["mtype"] = csinn_mem_type_enum.CSINN_MEM_TYPE_CPU_NOT_ALIGNED.value
                        tensor["name"] = key
                        bytes_to_save = data.tobytes()
                        weight_file.write(bytes_to_save)
                        offset = data.nbytes
                        tensor["name"] = key
                        model[name] = tensor
                    elif name.isdigit():
                        if self.qtype == "q8_0" and value.ndim == 2:
                            offset = self.quantize_block_32(
                                weight_file,
                                data,
                                csinn_mem_type_enum.CSINN_MEM_TYPE_BLOCK_Q8_0.value,
                            )
                            tensor["dtype"] = csinn_dtype_enum.CSINN_DTYPE_INT8.value
                            tensor["mtype"] = csinn_mem_type_enum.CSINN_MEM_TYPE_BLOCK_Q8_0.value
                        elif self.qtype == "q4_0" and value.ndim == 2:
                            offset = self.quantize_block_32(
                                weight_file,
                                data,
                                csinn_mem_type_enum.CSINN_MEM_TYPE_BLOCK_Q4_0.value,
                            )
                            tensor["dtype"] = csinn_dtype_enum.CSINN_DTYPE_INT4.value
                            tensor["mtype"] = csinn_mem_type_enum.CSINN_MEM_TYPE_BLOCK_Q4_0.value
                        elif self.qtype in ["q8_0", "q4_0"]:
                            tensor["dtype"] = StrToSHLDtype[str(data.dtype)]
                            tensor[
                                "mtype"
                            ] = csinn_mem_type_enum.CSINN_MEM_TYPE_CPU_NOT_ALIGNED.value

                            bytes_to_save = data.tobytes()
                            weight_file.write(bytes_to_save)
                            offset = data.nbytes
                        else:
                            logger.error("Unsupported quantization scheme")
                            sys.exit()

                        tensor["name"] = key
                        layers[int(name)][key] = tensor
                    data_offset += offset
            weight_file.close()

            logger.log(LOG, "data_offset:{}".format(data_offset))
            content["config"] = {}
            del self.config["_name_or_path"]
            content["config"]["model_params"] = self.config
            content["config"]["shl_params"] = {}
            content["config"]["shl_params"]["shl_model_type"] = "weight_only"
            model["layer"] = layers
            model["layers_num"] = self.config["num_layers"]
            content["model"] = model
            out_file = open(json_file_path, "w")
            json.dump(content, out_file, indent=4)

    logger.log(LOG, "Quantize model to {}".format(quantization_scheme))
    model_quantizer = Quantizer(model.state_dict(), config.to_dict(), quantization_scheme, save_dir)
    model_quantizer.from_pretrained()


def llm_quantize(name_or_dir, quantization_scheme, save_dir):
    """_summary_

    Args:
        name_or_dir (_type_): _description_
        quantization_scheme (_type_): _description_
        save_dir (_type_): _description_
    """

    if isinstance(name_or_dir, list):
        name_or_dir = name_or_dir[0]

    if quantization_scheme == "unset":
        logger.error("params 'quantization_scheme' must be setted!")
        sys.exit()

    if not isinstance(save_dir, str):
        logger.warning("save_path only support type string, not {}".format(type(save_dir)))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    try:
        model = AutoModelForCausalLM.from_pretrained(name_or_dir, trust_remote_code=True)
    except ValueError:
        model = AutoModel.from_pretrained(name_or_dir, trust_remote_code=True)
    base_quantize(model, model.config, quantization_scheme, save_dir)
