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
import ctypes
import numpy as np
import os

MODEL_DIR = os.path.dirname(os.path.realpath(__file__))
LIB_PATH = os.path.join(MODEL_DIR, "libmodel_wrapper.so")
model_lib = ctypes.CDLL(LIB_PATH)


def load_model(model_path):
    model_lib.load_model.argtypes = [ctypes.c_char_p]
    model_lib.load_model.restype = ctypes.c_void_p

    if not os.path.isabs(model_path):
        model_path = os.path.join(MODEL_DIR, model_path)

    model_path = bytes(model_path, encoding="utf8")
    model_path = ctypes.create_string_buffer(model_path, size=(len(model_path) + 1))
    sess = model_lib.load_model(model_path)

    return sess


def get_input_number(sess):
    model_lib.get_input_number.argtypes = [ctypes.c_void_p]
    model_lib.get_input_number.restype = ctypes.c_int

    in_num = model_lib.get_input_number(sess)

    return in_num


def get_output_number(sess):
    model_lib.get_output_number.argtypes = [ctypes.c_void_p]
    model_lib.get_output_number.restype = ctypes.c_int

    out_num = model_lib.get_output_number(sess)

    return out_num


def get_output_size_by_index(sess, index):
    model_lib.get_output_size_by_index.argtypes = [ctypes.c_void_p, ctypes.c_int]
    model_lib.get_output_size_by_index.restype = ctypes.c_int

    res = model_lib.get_output_size_by_index(sess, index)
    return res


def get_output_dim_num_by_index(sess, index):
    model_lib.get_output_dim_num_by_index.argtypes = [ctypes.c_void_p, ctypes.c_int]
    model_lib.get_output_dim_num_by_index.restype = ctypes.c_int

    res = model_lib.get_output_dim_num_by_index(sess, index)
    return res


def session_run(sess, inputs):
    if not isinstance(inputs, (tuple, list)):
        raise ValueError("Inputs needs: [ndarray, ndarray, ...]")

    actual_in_num = get_input_number(sess)
    assert actual_in_num == len(inputs), "Actual input number: {}, but get {}".format(
        actual_in_num, len(inputs)
    )

    model_lib.session_run.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int,
    ]
    model_lib.session_run.restype = None
    f_ptr = ctypes.POINTER(ctypes.c_float)
    data = (f_ptr * len(inputs))(*[single.ctypes.data_as(f_ptr) for single in inputs])

    model_lib.session_run(sess, data, len(inputs))


def get_output_by_index(sess, index):
    out_size = get_output_size_by_index(sess, index)

    model_lib.get_output_by_index.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    model_lib.get_output_by_index.restype = None

    out = np.zeros(out_size, np.float32)
    model_lib.get_output_by_index(sess, index, out.ctypes.data_as(ctypes.c_void_p))

    return out


def get_output_shape_by_index(sess, index):
    out_dim_num = get_output_dim_num_by_index(sess, index)

    model_lib.get_output_shape_by_index.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    model_lib.get_output_shape_by_index.restype = None

    out = np.zeros(out_dim_num, np.int32)
    model_lib.get_output_shape_by_index(sess, index, out.ctypes.data_as(ctypes.c_void_p))

    return out


if __name__ == "__main__":
    model_path = "./shl.hhb.bm"
    sess = load_model(model_path)
    in_num = get_input_number(sess)
    out_num = get_output_number(sess)
    print(f"input num: {in_num}")
    print(f"output num: {out_num}")

    out_size = get_output_size_by_index(sess, 0)
    print(f"output size: {out_size}")

    data = np.fromfile("./images.0.bin", dtype=np.float32)
    session_run(sess, [data])

    out = get_output_by_index(sess, 0)

    out.tofile("output_python.txt", "\n")
    # print(out)
