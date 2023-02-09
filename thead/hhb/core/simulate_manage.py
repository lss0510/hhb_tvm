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
# pylint: disable=unnecessary-comprehension
"""Manage simulate"""
import os
import logging

import numpy as np

from .preprocess_manage import DatasetLoader
from .common import print_top5, ensure_dir

# pylint: disable=invalid-name
LOG = 25
logger = logging.getLogger("HHB")


def inference_model(graph_module, data_loader: DatasetLoader, postprocess="top5", output_dir="."):
    """Inference GraphModule by specified data loader

    Parameters
    ----------
    graph_module: GraphModule
        Runtime graph module that can be used to execute the graph.

    data_loader: DatasetLoader
        Data loader machine that can load specified data.

    postprocess: Optional[str]
        How to deal with output data:
        "top5": print the top5 values of outputs;
        "save": save the output values into file;
        "save_and_top5": print the top5 values and save the values into file.

    """
    dataset = data_loader.get_data()
    index = 0
    for data in dataset:
        graph_module.run(**data)
        for i in range(graph_module.get_num_outputs()):
            output = graph_module.get_output(i).asnumpy()
            out_shape = output.shape
            out = np.reshape(output, [np.prod(output.size)])
            out_basename = (
                data_loader.all_file_path[index]
                if len(data_loader.all_file_path) > index
                else data_loader.all_file_path[0] + "_" + str(index)
            )
            output_prefix = os.path.basename(out_basename) + "_output_" + str(i) + ".tensor"
            if postprocess == "top5":
                print_top5(out, str(i), out_shape)
            elif postprocess == "save":
                output_dir = ensure_dir(output_dir)
                np.savetxt(
                    os.path.join(output_dir, output_prefix),
                    out,
                    fmt="%f",
                    delimiter="\n",
                    newline="\n",
                )
            else:
                print_top5(out, str(i), out_shape)
                output_dir = ensure_dir(output_dir)
                np.savetxt(
                    os.path.join(output_dir, output_prefix),
                    out,
                    fmt="%f",
                    delimiter="\n",
                    newline="\n",
                )
        index += 1


def inference_elf(elf_file, dataset, input_name_list, all_file_path, output_dir="."):
    """Inference elf on x86 by specified data loader

    Parameters
    ----------
    elf: str
        elf file that build to execute the graph.

    data_loader: DatasetLoader
        Data loader machine that can load specified data.

    postprocess: Optional[str]
        How to deal with output data:
        "top5": print the top5 values of outputs;
        "save": save the output values into file;
        "save_and_top5": print the top5 values and save the values into file.

    """
    command_line_base = "cd " + output_dir + "; " + elf_file + " ./hhb.bm "

    index = 0
    for data in dataset:
        command_line = command_line_base
        data_count = 0
        for k in input_name_list:
            if len(all_file_path) == 1:
                if index == 0:
                    input_name = all_file_path[0]
                else:
                    input_name = all_file_path[0] + "_" + str(index)
            else:
                input_name = all_file_path[index]
            v = data[k]
            v = v.astype("float32")
            file = os.path.basename(input_name) + ".{}.bin".format(data_count)
            file_path = os.path.join(output_dir, file)
            v.tofile(file_path)
            command_line += file + " "
            data_count += 1

        logger.log(LOG, command_line)
        os.system(command_line)

        index += 1
