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
import sys
import os
import argparse

import yarr


MODEL_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(MODEL_DIR)
import model_wrapper


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", "-i", type=str, help="IP address of your board.")
    parser.add_argument("--port", "-p", type=int, help="Select a port number for this server.")

    opt = parser.parse_args()
    return opt


def main(opt):
    ip = opt.ip
    port = opt.port
    yarr.yarr(
        (ip, port),
        [
            model_wrapper.get_input_number,
            model_wrapper.get_output_number,
            model_wrapper.load_model,
            model_wrapper.get_output_size_by_index,
            model_wrapper.get_output_by_index,
            model_wrapper.session_run,
            model_wrapper.get_output_dim_num_by_index,
            model_wrapper.get_output_shape_by_index,
        ],
    )


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
