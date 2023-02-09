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
import yarr
import numpy as np


class SHLHostInference:
    """A simple wrapper class for model inference. It operates in
    client-server mode.

    server: target board, core inference program of the model is on the board.
    client: host, pre/post-process program of the model is on the host.

    Example server:
    python3 model_server.py --ip "10.63.58.150" --port 5555

    Example client:

        .. code-block:: python

            # load data or preprocess from original images
            data = np.fromfile("images.0.bin", dtype=np.float32)

            # connect servert and inference model
            host_inf = SHLHostInference("shl.hhb.bm", ip="10.63.58.150", port=5555)
            outputs = host_inf.run([data])

            # postprocess
            for i, out in enumerate(outputs):
                print(out.shape)
                out.tofile(f"yarr_output_{i}.txt", "\n")

    """

    def __init__(self, path, ip, port) -> None:
        self.address = (ip, port)
        self.sess = yarr.call(self.address, "load_model", path)

    def run(self, inputs):
        if not inputs:
            raise ValueError("There is no input!")
        if not isinstance(inputs, (tuple, list)):
            inputs = [inputs]
        final_inputs = []
        for idx, i in enumerate(inputs):
            if isinstance(i, np.ndarray) and i.dtype == np.float32:
                final_inputs.append(i.flatten())
            elif isinstance(i, (tuple, list)):
                curr_value = np.array(i, dtype=np.float32)
                final_inputs.append(curr_value.flatten())
            else:
                raise ValueError(f"inputs[{idx}] is invalid.")

        # get some useful info
        in_num = yarr.call(self.address, "get_input_number", self.sess)
        assert in_num == len(
            final_inputs
        ), f"There are {in_num} inputs in model, but you give {len(final_inputs)} input data."
        out_num = yarr.call(self.address, "get_output_number", self.sess)

        # inference model
        yarr.call(self.address, "session_run", self.sess, final_inputs)

        # get output
        output = []
        for o_idx in range(out_num):
            out = yarr.call(self.address, "get_output_by_index", self.sess, o_idx)

            out_shape = yarr.call(self.address, "get_output_shape_by_index", self.sess, o_idx)
            out = np.reshape(out, out_shape.tolist())

            output.append(out)

        return output


if __name__ == "__main__":
    data = np.fromfile("images.0.bin", dtype=np.float32)

    host_inf = SHLHostInference("shl.hhb.bm", ip="10.63.58.150", port=5555)
    outputs = host_inf.run([data])

    for i, out in enumerate(outputs):
        print(out.shape)
        out.tofile(f"yarr_output_{i}.txt", "\n")
