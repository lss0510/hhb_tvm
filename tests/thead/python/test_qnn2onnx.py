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

"""Qnn to ONNX serialization test cases"""

import numpy as np
import onnxruntime as rt

import tvm
from tvm import relay
from tvm.relay.quantize import qnn_to_onnx
from tvm.relay.frontend.common import infer_shape

import hhb

hhb.set_debug_level("ERROR")


def qmod_to_onnx(qmod, name):
    onnx_model = qnn_to_onnx(qmod, {}, name, path=None)
    return onnx_model.SerializeToString()


def run_onnx(onnx_model, input_data):
    sess = rt.InferenceSession(onnx_model)
    input_names = {}
    for input, data in zip(sess.get_inputs(), input_data):
        input_names[input.name] = data
    output_names = [out.name for out in sess.get_outputs()]
    res = sess.run(output_names, input_names)
    return res


def run_relay(func, data_tuple, is_dyn=False):
    target = "llvm"
    dev = tvm.device("llvm", 0)
    kind = "graph" if not is_dyn else "vm"
    relay_res = relay.create_executor(kind, device=dev, target=target).evaluate(func)(*data_tuple)

    result = []
    relay_res = relay_res if isinstance(relay_res, list) else [relay_res]
    for res in relay_res:
        result.append(res.numpy())

    return result


def verify_results(relay_ir, qnn_ir, indata, test_name, rtol=1e-7, atol=0, is_dyn=False):
    relay_results = run_relay(relay_ir["main"], indata, is_dyn)
    onnx_results = run_onnx(qmod_to_onnx(qnn_ir, test_name), indata)
    for relay_res, onnx_res in zip(relay_results, onnx_results):
        np.testing.assert_allclose(relay_res, onnx_res, rtol=rtol, atol=atol)


def quantize_relay_ir(relay_ir, board="x86_ref", quant_scheme="float32"):
    compiler = hhb.Compiler(board)
    hhb_relay_ir = compiler.create_relay_ir()
    hhb_relay_ir.set_model(relay_ir, None)
    compiler.relay_ir = hhb_relay_ir
    compiler._init_session()

    compiler.config.quantize.quantization_scheme.value = quant_scheme
    compiler.quantize()
    qnn_ir, _ = compiler.qnn_ir.get_model()
    return qnn_ir


def test_relu():
    def verify_relu(shape, quant_scheme="float32"):
        # relay ir
        x = relay.var("x", relay.TensorType(shape, "float32"))
        z = relay.nn.relu(x)
        func = relay.Function([x], z)
        relay_ir = tvm.IRModule()
        relay_ir["main"] = func

        # convert to qnn
        qnn_ir = quantize_relay_ir(relay_ir, quant_scheme=quant_scheme)

        x_data = np.random.uniform(low=-1, high=1, size=shape).astype("float32")
        verify_results(relay_ir, qnn_ir, [x_data], "test_relu", rtol=1e-5, atol=1e-5)

    verify_relu((2,))
    verify_relu((2, 3))
    verify_relu((2, 3, 4))
    verify_relu((1, 3, 112, 112))


def test_conv2d():
    def verify_conv2d(
        scale,
        dshape,
        kshape,
        padding=(1, 1),
        groups=1,
        dilation=(1, 1),
        with_bias=True,
        quant_scheme="float32",
        **attrs,
    ):
        x = relay.var("x", shape=dshape, dtype="float32")

        w_value = np.random.uniform(-scale, scale, size=kshape).astype("float32")
        w = relay.const(w_value)
        y = relay.nn.conv2d(x, w, padding=padding, dilation=dilation, groups=groups, **attrs)

        if with_bias:
            y_shape = infer_shape(y)
            b_value = np.random.uniform(-scale, scale, size=(y_shape[1],)).astype("float32")
            b = relay.const(b_value)

            z = relay.nn.bias_add(y, b)

            func = relay.Function([x], z)
        else:
            func = relay.Function([x], y)

        relay_ir = tvm.IRModule()
        relay_ir["main"] = func
        qnn_ir = quantize_relay_ir(relay_ir, quant_scheme=quant_scheme)

        data = np.random.uniform(-scale, scale, size=dshape).astype("float32")
        verify_results(relay_ir, qnn_ir, [data], "test_conv2d", rtol=1e-5, atol=1e-5, is_dyn=True)

    dshape = (1, 32, 18, 18)
    kshape = (32, 1, 3, 3)
    verify_conv2d(1, dshape, kshape, padding=(1, 1), channels=32, groups=32, kernel_size=(3, 3))

    dshape = (1, 32, 18, 18)
    kshape = (32, 4, 3, 3)
    verify_conv2d(1, dshape, kshape, padding=(1, 1), channels=32, groups=8, kernel_size=(3, 3))

    # also group conv2d
    dshape = (1, 32, 18, 18)
    kshape = (64, 1, 3, 3)
    verify_conv2d(1, dshape, kshape, padding=(1, 1), channels=64, groups=32, kernel_size=(3, 3))

    # normal conv2d
    dshape = (1, 3, 224, 224)
    kshape = (10, 3, 3, 3)
    verify_conv2d(1, dshape, kshape, padding=(1, 1), channels=10, kernel_size=(3, 3))

    dshape = (1, 3, 224, 224)
    kshape = (10, 3, 3, 3)
    verify_conv2d(1, dshape, kshape, padding=(2, 2), channels=10, kernel_size=(3, 3))

    dshape = (1, 3, 18, 18)
    kshape = (10, 3, 3, 3)
    verify_conv2d(
        1,
        dshape,
        kshape,
        padding=(1, 1),
        channels=10,
        kernel_size=(3, 3),
        dilation=(3, 3),
    )

    dshape = (1, 3, 18, 18)
    kshape = (10, 3, 2, 2)
    verify_conv2d(
        1,
        dshape,
        kshape,
        padding=(2, 2),
        channels=10,
        kernel_size=(2, 2),
        dilation=(1, 1),
    )

    dshape = (1, 3, 18, 18)
    kshape = (10, 3, 4, 4)
    verify_conv2d(1, dshape, kshape, padding=(1, 1), channels=10, kernel_size=(4, 4))

    dshape = (1, 3, 18, 18)
    kshape = (10, 3, 4, 4)
    verify_conv2d(1, dshape, kshape, padding=(1, 1), channels=10, kernel_size=(4, 4))
    verify_conv2d(
        1, dshape, kshape, padding=(1, 1), channels=10, kernel_size=(4, 4), with_bias=False
    )


def test_conv2d_relu():
    def verify_conv2d_relu(
        scale,
        dshape,
        kshape,
        channels,
        kernel_size,
        strides=(1, 1),
        padding=(1, 1),
        groups=1,
        dilation=(1, 1),
        with_bias=True,
        quant_scheme="float32",
    ):
        x = relay.var("x", shape=dshape, dtype="float32")

        w_value = np.random.uniform(-scale, scale, size=kshape).astype("float32")
        w = relay.const(w_value)

        if with_bias:
            y_shape = dshape
            b_value = np.random.uniform(-scale, scale, size=(y_shape[1],)).astype("float32")
            b = relay.const(b_value)
            y = relay.qnn.op.csi_conv2d_relu(
                x,
                w,
                b,
                strides=strides,
                padding=padding,
                dilation=dilation,
                groups=groups,
                channels=channels,
                kernel_size=kernel_size,
                data_layout="NCHW",
                kernel_layout="OIHW",
                out_layout="",
                out_dtype="float32",
                q_params=None,
            )

            func = relay.Function([x], y)
        else:
            y_shape = dshape
            b_value = np.zeros((y_shape[1],)).astype("float32")
            b = relay.const(b_value)
            y = relay.qnn.op.csi_conv2d_relu(
                x,
                w,
                b,
                strides=strides,
                padding=padding,
                dilation=dilation,
                groups=groups,
                channels=channels,
                kernel_size=kernel_size,
                data_layout="NCHW",
                kernel_layout="OIHW",
                out_layout="",
                out_dtype="float32",
                q_params=None,
            )

            func = relay.Function([x], y)

        relay_ir = tvm.IRModule()
        relay_ir["main"] = func
        qnn_ir = relay_ir
        qnn_ir = relay.transform.InferType()(qnn_ir)

        data = np.random.uniform(-scale, scale, size=dshape).astype("float32")
        onnx_results = run_onnx(
            qmod_to_onnx(qnn_ir, "test_conv2d_relu"),
            [
                data,
            ],
        )

    dshape = (1, 32, 18, 18)
    kshape = (32, 1, 3, 3)
    verify_conv2d_relu(
        1,
        dshape,
        kshape,
        channels=32,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding=(1, 1, 1, 1),
        groups=32,
    )

    dshape = (1, 32, 18, 18)
    kshape = (32, 1, 3, 3)
    verify_conv2d_relu(
        1,
        dshape,
        kshape,
        channels=32,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding=(1, 1, 1, 1),
        groups=32,
        with_bias=False,
    )


def test_conv2d_relu6():
    def verify_conv2d_relu6(
        scale,
        dshape,
        kshape,
        channels,
        kernel_size,
        strides=(1, 1),
        padding=(1, 1),
        groups=1,
        dilation=(1, 1),
        with_bias=True,
        quant_scheme="float32",
    ):
        x = relay.var("x", shape=dshape, dtype="float32")

        w_value = np.random.uniform(-scale, scale, size=kshape).astype("float32")
        w = relay.const(w_value)

        if with_bias:
            y_shape = dshape
            b_value = np.random.uniform(-scale, scale, size=(y_shape[1],)).astype("float32")
            b = relay.const(b_value)
            y = relay.qnn.op.csi_conv2d_relu6(
                x,
                w,
                b,
                strides=strides,
                padding=padding,
                dilation=dilation,
                groups=groups,
                channels=channels,
                kernel_size=kernel_size,
                data_layout="NCHW",
                kernel_layout="OIHW",
                out_layout="",
                out_dtype="float32",
                q_params=None,
            )

            func = relay.Function([x], y)
        else:
            y_shape = dshape
            b_value = np.zeros((y_shape[1],)).astype("float32")
            b = relay.const(b_value)
            y = relay.qnn.op.csi_conv2d_relu6(
                x,
                w,
                b,
                strides=strides,
                padding=padding,
                dilation=dilation,
                groups=groups,
                channels=channels,
                kernel_size=kernel_size,
                data_layout="NCHW",
                kernel_layout="OIHW",
                out_layout="",
                out_dtype="float32",
                q_params=None,
            )

            func = relay.Function([x], y)

        relay_ir = tvm.IRModule()
        relay_ir["main"] = func
        qnn_ir = relay_ir
        qnn_ir = relay.transform.InferType()(qnn_ir)
        data = np.random.uniform(-scale, scale, size=dshape).astype("float32")
        onnx_results = run_onnx(
            qmod_to_onnx(qnn_ir, "test_conv2d_relu6"),
            [
                data,
            ],
        )

    dshape = (1, 32, 18, 18)
    kshape = (32, 1, 3, 3)
    verify_conv2d_relu6(
        1,
        dshape,
        kshape,
        channels=32,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding=(1, 1, 1, 1),
        groups=32,
    )

    dshape = (1, 32, 18, 18)
    kshape = (32, 1, 3, 3)
    verify_conv2d_relu6(
        1,
        dshape,
        kshape,
        channels=32,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding=(1, 1, 1, 1),
        groups=32,
        with_bias=False,
    )


def test_sigmoid():
    def verify_sigmoid(shape, quant_scheme="float32"):
        # relay ir
        x = relay.var("x", relay.TensorType(shape, "float32"))
        z = relay.sigmoid(x)
        func = relay.Function([x], z)
        relay_ir = tvm.IRModule()
        relay_ir["main"] = func

        # convert to qnn
        qnn_ir = quantize_relay_ir(relay_ir, quant_scheme=quant_scheme)

        x_data = np.random.uniform(low=-1, high=1, size=shape).astype("float32")
        verify_results(relay_ir, qnn_ir, [x_data], "test_sigmoid", rtol=1e-5, atol=1e-5)

    verify_sigmoid((2,))
    verify_sigmoid((3, 32))
    verify_sigmoid((6, 112, 112))
    verify_sigmoid((2, 4, 32, 32))


def test_add():
    def verify_add(shape, quant_scheme="float32"):
        # relay ir
        x = relay.var("x", relay.TensorType(shape, "float32"))
        y = relay.const(np.random.rand(*shape).astype("float32"))
        z = relay.add(x, x)
        func = relay.Function([x], z)
        relay_ir = tvm.IRModule()
        relay_ir["main"] = func

        # convert to qnn
        qnn_ir = quantize_relay_ir(relay_ir, quant_scheme=quant_scheme)

        x_data = np.random.uniform(low=-1, high=1, size=shape).astype("float32")
        verify_results(relay_ir, qnn_ir, [x_data], "test_add", rtol=1e-5, atol=1e-5)

    verify_add((3,))
    verify_add((2, 32, 32))
    verify_add((1, 112, 112))
    verify_add((4, 2, 28, 28))


def test_subtract():
    def verify_subtract(shape, quant_scheme="float32"):
        # relay ir
        x = relay.var("x", relay.TensorType(shape, "float32"))
        y = relay.const(np.random.rand(*shape).astype("float32"))
        z = relay.subtract(x, y)
        func = relay.Function([x], z)
        relay_ir = tvm.IRModule()
        relay_ir["main"] = func

        # convert to qnn
        qnn_ir = quantize_relay_ir(relay_ir, quant_scheme=quant_scheme)

        x_data = np.random.uniform(low=-1, high=1, size=shape).astype("float32")
        verify_results(relay_ir, qnn_ir, [x_data], "test_subtract", rtol=1e-5, atol=1e-5)

    verify_subtract((4,))
    verify_subtract((1, 2))
    verify_subtract((1, 12, 32))
    verify_subtract((4, 1, 112, 112))


def test_div():
    def verify_div(shape, quant_scheme="float32"):
        # relay ir
        x = relay.var("x", relay.TensorType(shape, "float32"))
        # y = relay.const(np.random.rand(*shape).astype("float32"))
        z = relay.divide(x, x)
        func = relay.Function([x], z)
        relay_ir = tvm.IRModule()
        relay_ir["main"] = func

        # convert to qnn
        qnn_ir = quantize_relay_ir(relay_ir, quant_scheme=quant_scheme)

        x_data = np.random.uniform(low=-1, high=1, size=shape).astype("float32")
        verify_results(relay_ir, qnn_ir, [x_data], "test_div", rtol=1e-5, atol=1e-5)

    verify_div((4,))
    verify_div((2, 2))
    verify_div((3, 32, 32))
    verify_div((1, 4, 26, 26))


def test_mul():
    def verify_mul(shape, quant_scheme="float32"):
        # relay ir
        x = relay.var("x", relay.TensorType(shape, "float32"))
        y = relay.const(np.random.rand(*shape).astype("float32"))
        z = relay.multiply(x, x)
        func = relay.Function([x], z)
        relay_ir = tvm.IRModule()
        relay_ir["main"] = func

        # convert to qnn
        qnn_ir = quantize_relay_ir(relay_ir, quant_scheme=quant_scheme)

        x_data = np.random.uniform(low=-1, high=1, size=shape).astype("float32")
        verify_results(relay_ir, qnn_ir, [x_data], "test_mul", rtol=1e-5, atol=1e-5)

    verify_mul((112,))
    verify_mul((1, 14))
    verify_mul((1, 4, 112))
    verify_mul((1, 4, 112, 112))


def test_split():
    def verify_split(shape, indice_dshape, axis, quant_scheme="float32"):
        # relay ir
        x = relay.var("x", relay.TensorType(shape, "float32"))
        z = relay.split(x, indice_dshape, axis)
        func = relay.Function([x], z[0])
        relay_ir = tvm.IRModule()
        relay_ir["main"] = func
        # convert to qnn
        qnn_ir = quantize_relay_ir(relay_ir, quant_scheme=quant_scheme)
        x_data = np.random.uniform(low=-1, high=1, size=shape).astype("float32")
        verify_results(relay_ir, qnn_ir, [x_data], "test_split", rtol=1e-5, atol=1e-5)

    verify_split((1, 3, 2, 2), 3, 1, "float32")
    verify_split((1, 4, 2, 2), 2, 1, "float32")
    verify_split((1, 3, 4, 2), 4, 2, "float32")


def test_softmax():
    def verify_softmax(shape, axis, quant_scheme="float32"):
        # relay ir
        x = relay.var("x", relay.TensorType(shape, "float32"))
        z = relay.nn.softmax(x, axis)
        func = relay.Function([x], z)
        relay_ir = tvm.IRModule()
        relay_ir["main"] = func
        # convert to qnn
        qnn_ir = quantize_relay_ir(relay_ir, quant_scheme=quant_scheme)
        x_data = np.random.uniform(low=-1, high=1, size=shape).astype("float32")
        verify_results(relay_ir, qnn_ir, [x_data], "test_softmax", rtol=1e-5, atol=1e-5)

    verify_softmax((2,), 0)
    verify_softmax((3, 9), 1)
    verify_softmax((4, 112, 87), 1)
    verify_softmax((2, 4, 32, 32), 2)


def test_relu6():
    def verify_relu6(shape, quant_scheme="float32"):
        # construct qnn ir 'relu6'
        x = relay.var("x", relay.TensorType(shape, "float32"))
        out_dtype = "float32"
        q_params = None
        qnn_ir = relay.qnn.op.csi_relu6(
            x,
            out_dtype,
            q_params,
            layer_name="relu6",
        )

        x_data = np.random.uniform(low=-10, high=10, size=shape).astype("float32")
        onnx_results = run_onnx(qmod_to_onnx(qnn_ir, "test_relu6"), [x_data])

    verify_relu6((2,))
    verify_relu6((2, 3))
    verify_relu6((2, 3, 4))
    verify_relu6((1, 3, 112, 112))


def test_log_softmax():
    def verify_log_softmax(shape, quant_scheme="float32"):
        # relay ir
        x = relay.var("x", relay.TensorType(shape, "float32"))
        z = relay.nn.log_softmax(x)
        func = relay.Function([x], z)
        relay_ir = tvm.IRModule()
        relay_ir["main"] = func

        # convert to qnn
        qnn_ir = quantize_relay_ir(relay_ir, quant_scheme=quant_scheme)

        x_data = np.random.uniform(low=-1, high=1, size=shape).astype("float32")
        verify_results(relay_ir, qnn_ir, [x_data], "test_log_softmax", rtol=1e-5, atol=1e-5)

    verify_log_softmax((1, 3), "float32")
    verify_log_softmax((2, 32), "float32")
    verify_log_softmax((32, 2), "float32")


def test_avgpool2d():
    def verify_avgpool2d(shape, pool_size, strides, dilation, padding, quant_scheme="float32"):
        # relay ir
        x = relay.var("x", relay.TensorType(shape, "float32"))
        z = relay.nn.avg_pool2d(x, pool_size, strides, dilation, padding)
        func = relay.Function([x], z)
        relay_ir = tvm.IRModule()
        relay_ir["main"] = func

        # convert to qnn
        qnn_ir = quantize_relay_ir(relay_ir, quant_scheme=quant_scheme)

        x_data = np.random.uniform(low=-1, high=1, size=shape).astype("float32")
        verify_results(relay_ir, qnn_ir, [x_data], "test_avgpool2d", rtol=1e-5, atol=1e-5)

    verify_avgpool2d((1, 3, 112, 112), (1, 1), (1, 1), (1, 1), (0, 0))
    verify_avgpool2d((2, 6, 32, 32), (2, 2), (1, 1), (1, 1), (0, 0))
    verify_avgpool2d((1, 3, 224, 224), (1, 1), (2, 2), (1, 1), (0, 0))
    verify_avgpool2d((1, 4, 64, 64), (1, 1), (1, 1), (2, 2), (0, 0))
    verify_avgpool2d((1, 4, 3, 3), (3, 3), (1, 1), (1, 1), (2, 2))


def test_maxpool2d():
    def verify_maxpool2d(shape, pool_size, strides, dilation, padding, quant_scheme="float32"):
        # relay ir
        x = relay.var("x", relay.TensorType(shape, "float32"))
        z = relay.nn.max_pool2d(x, pool_size, strides, dilation, padding)
        func = relay.Function([x], z)
        relay_ir = tvm.IRModule()
        relay_ir["main"] = func

        # convert to qnn
        qnn_ir = quantize_relay_ir(relay_ir, quant_scheme=quant_scheme)

        x_data = np.random.uniform(low=-1, high=1, size=shape).astype("float32")
        verify_results(relay_ir, qnn_ir, [x_data], "test_maxpool2d", rtol=1e-5, atol=1e-5)

    verify_maxpool2d((1, 3, 112, 112), (1, 1), (1, 1), (1, 1), (0, 0))
    verify_maxpool2d((1, 3, 112, 112), (2, 2), (1, 1), (1, 1), (0, 0))
    verify_maxpool2d((1, 3, 112, 112), (1, 1), (2, 2), (1, 1), (0, 0))
    verify_maxpool2d((1, 3, 112, 112), (2, 2), (1, 1), (1, 1), (1, 1))
    verify_maxpool2d((1, 3, 32, 24), (1, 1), (1, 1), (1, 1), (0, 0))


def test_global_max_pool2d():
    def verify_global_max_pool2d(shape, quant_scheme="float32"):
        # relay ir
        x = relay.var("x", relay.TensorType(shape, "float32"))
        z = relay.nn.global_max_pool2d(x)
        func = relay.Function([x], z)
        relay_ir = tvm.IRModule()
        relay_ir["main"] = func

        # convert to qnn
        qnn_ir = quantize_relay_ir(relay_ir, quant_scheme=quant_scheme)

        x_data = np.random.uniform(low=-1, high=1, size=shape).astype("float32")
        verify_results(relay_ir, qnn_ir, [x_data], "test_global_max_pool2d", rtol=1e-5, atol=1e-5)

    verify_global_max_pool2d((1, 3, 112, 112))
    verify_global_max_pool2d((1, 4, 43, 43))
    verify_global_max_pool2d((1, 3, 32, 28))
    verify_global_max_pool2d((1, 3, 34, 112))


def test_global_avg_pool2d():
    def verify_global_avg_pool2d(shape, quant_scheme="float32"):
        # relay ir
        x = relay.var("x", relay.TensorType(shape, "float32"))
        z = relay.nn.global_avg_pool2d(x)
        func = relay.Function([x], z)
        relay_ir = tvm.IRModule()
        relay_ir["main"] = func

        # convert to qnn
        qnn_ir = quantize_relay_ir(relay_ir, quant_scheme=quant_scheme)

        x_data = np.random.uniform(low=-1, high=1, size=shape).astype("float32")
        verify_results(relay_ir, qnn_ir, [x_data], "test_global_avg_pool2d", rtol=1e-5, atol=1e-5)

    verify_global_avg_pool2d((1, 3, 112, 112))
    verify_global_avg_pool2d((1, 4, 43, 43))
    verify_global_avg_pool2d((1, 3, 32, 28))
    verify_global_avg_pool2d((1, 3, 34, 112))


def test_leaky_relu():
    def verify_leaky_relu(shape, alpha, quant_scheme="float32"):
        x = relay.var("x", relay.TensorType(shape, "float32"))
        z = relay.nn.leaky_relu(x, alpha)
        func = relay.Function([x], z)
        relay_ir = tvm.IRModule()
        relay_ir["main"] = func

        # convert to qnn
        qnn_ir = quantize_relay_ir(relay_ir, quant_scheme=quant_scheme)

        x_data = np.random.uniform(low=-1, high=1, size=shape).astype("float32")
        verify_results(relay_ir, qnn_ir, [x_data], "test_leaky_relu", rtol=1e-5, atol=1e-5)

    verify_leaky_relu((2,), 0.5)
    verify_leaky_relu((2, 3), 0.6)
    verify_leaky_relu((1, 4, 5), 0.4)
    verify_leaky_relu((2, 5, 32, 32), 0.3)


def test_concatenate():
    def verify_concatenate(ishape1, ishape2, axis, quant_scheme="float32"):
        data = relay.var("data", relay.TensorType(ishape1, "float32"))
        data_ = relay.var("data_", relay.TensorType(ishape2, "float32"))

        z = relay.concatenate((data, data_), axis)
        func = relay.Function([data, data_], z)
        relay_ir = tvm.IRModule()
        relay_ir["main"] = func
        # convert to qnn
        qnn_ir = quantize_relay_ir(relay_ir, quant_scheme=quant_scheme)

        x_data = np.random.uniform(low=-1, high=1, size=ishape1).astype("float32")
        x_data_ = np.random.uniform(low=-1, high=1, size=ishape2).astype("float32")
        verify_results(
            relay_ir, qnn_ir, [x_data, x_data_], "test_concatenate", rtol=1e-5, atol=1e-5
        )

    verify_concatenate((3, 2, 2, 4), (3, 7, 2, 4), 1)
    verify_concatenate((1, 3, 2, 3), (1, 3, 2, 3), 1)
    verify_concatenate((1, 3, 3, 3), (1, 3, 6, 3), 2)
    verify_concatenate((1, 3, 2, 6), (1, 3, 2, 3), 3)


def test_expand_dims():
    def verify_expand_dims(shape, axis, nums_newshape, quant_scheme="float32"):
        # relay ir
        x = relay.var("x", relay.TensorType(shape, "float32"))
        z = relay.expand_dims(x, axis, nums_newshape)
        func = relay.Function([x], z)
        relay_ir = tvm.IRModule()
        relay_ir["main"] = func
        # convert to qnn
        qnn_ir = quantize_relay_ir(relay_ir, quant_scheme=quant_scheme)
        x_data = np.random.uniform(low=-1, high=1, size=shape).astype("float32")
        verify_results(relay_ir, qnn_ir, [x_data], "test_expand_dims", rtol=1e-5, atol=1e-5)

    verify_expand_dims((3, 1, 2), 1, 6, "float32")
    verify_expand_dims((3, 1), 0, 3, "float32")
    verify_expand_dims((3, 1, 2, 3), 3, 1, "float32")


def test_reshape():
    def verify_reshape(shape, newshape, quant_scheme="float32"):
        # relay ir
        x = relay.var("x", relay.TensorType(shape, "float32"))
        z = relay.reshape(x, newshape)
        func = relay.Function([x], z)
        relay_ir = tvm.IRModule()
        relay_ir["main"] = func
        # convert to qnn
        qnn_ir = quantize_relay_ir(relay_ir, quant_scheme=quant_scheme)
        x_data = np.random.uniform(low=-1, high=1, size=shape).astype("float32")
        verify_results(relay_ir, qnn_ir, [x_data], "test_reshape", rtol=1e-5, atol=1e-5)

    verify_reshape((3, 2, 2), (2, 3, 2), "float32")
    verify_reshape((3, 2, 2, 4), (4, 3, 4), "float32")
    verify_reshape((3, 2, 2, 10), (10, 3, 2, 2), "float32")
    verify_reshape((3, 2), (6,), "float32")
    verify_reshape((6, 6), (2, 18), "float32")


def test_flatten():
    def verify_flatten(shape, quant_scheme="float32"):
        # relay ir
        x = relay.var("x", relay.TensorType(shape, "float32"))
        z = relay.nn.batch_flatten(x)
        func = relay.Function([x], z)
        relay_ir = tvm.IRModule()
        relay_ir["main"] = func

        # convert to qnn

        x_data = np.random.uniform(low=-1, high=1, size=shape).astype("float32")
        new_shape = [shape[0], -1]
        qnn_ir = relay.qnn.op.csi_reshape(
            x,
            new_shape,
            "float32",
            None,
            layer_name="daf",
        )

        verify_results(relay_ir, qnn_ir, [x_data], "test_flatten", rtol=1e-5, atol=1e-5)

    verify_flatten((3,), "float32")
    verify_flatten((3, 32), "float32")
    verify_flatten((3, 2, 36), "float32")
    verify_flatten((3, 2, 3, 3), "float32")


def test_transpose():
    def verify_transpose(shape, oshape, dtype, quant_scheme):
        x = relay.var("data", relay.TensorType(shape, "float32"))
        z = relay.transpose(x, oshape)
        func = relay.Function([x], z)
        relay_ir = tvm.IRModule()
        relay_ir["main"] = func
        # convert to qnn
        qnn_ir = quantize_relay_ir(relay_ir, quant_scheme=quant_scheme)

        x_data = np.random.uniform(low=-1, high=1, size=shape).astype("float32")
        verify_results(relay_ir, qnn_ir, [x_data], "test_transpose", rtol=1e-5, atol=1e-5)

    verify_transpose((1, 2, 3), [1, 2, 0], "float32", "float32")
    verify_transpose((1, 2, 3, 4), [1, 2, 0, 3], "float32", "float32")
    verify_transpose((2, 3), [1, 0], "float32", "float32")


def test_dense():
    def verify_dense(x_shape, k_shape, units, dtype, quant_scheme):
        x = relay.var("x", shape=(x_shape), dtype=dtype)
        kernel = relay.const(np.random.rand(*k_shape).astype(dtype))
        z = relay.nn.dense(x, kernel, units=units, out_dtype=dtype)
        relay_ir = tvm.IRModule().from_expr(z)

        x_data = np.random.uniform(low=-1, high=1, size=x_shape).astype("float32")

        qnn_ir = quantize_relay_ir(relay_ir, quant_scheme=quant_scheme)

        verify_results(relay_ir, qnn_ir, [x_data], "test_dense", rtol=1e-5, atol=1e-5)

    verify_dense((4, 3), (2, 3), 2, "float32", "float32")
    verify_dense((2, 4), (3, 4), 3, "float32", "float32")
    verify_dense((1, 2), (2, 2), 2, "float32", "float32")


def test_argmax():
    def verify_argmax(shape, axis, keepdims, exclude, quant_scheme="float32"):
        # relay ir
        x = relay.var("x", relay.TensorType(shape, "float32"))
        z = relay.argmax(x, axis, keepdims, exclude)
        func = relay.Function([x], z)
        relay_ir = tvm.IRModule().from_expr(func)

        # convert to qnn
        qnn_ir = quantize_relay_ir(relay_ir, quant_scheme=quant_scheme)
        x_data = np.random.uniform(low=-1, high=1, size=shape).astype("float32")

        verify_results(relay_ir, qnn_ir, [x_data], "test_argmax", rtol=1e-5, atol=1e-5)

    verify_argmax((2,), 0, True, False)
    verify_argmax((2, 54), 1, True, False)
    verify_argmax((1, 4, 6), 1, False, False)
    verify_argmax((2, 5, 2, 3), 3, True, False)


def test_batch_norm():
    def verify_batch_norm(shape, axis, quant_scheme="float32"):
        x = relay.var("data", relay.TensorType(shape, "float32"))
        beta = relay.var("beta", relay.TensorType((shape[axis],), "float32"))
        gamma = relay.var("gamma", relay.TensorType((shape[axis],), "float32"))
        moving_mean = relay.var("moving_mean", relay.TensorType((shape[axis],), "float32"))
        moving_var = relay.var("moving_var", relay.TensorType((shape[axis],), "float32"))

        act = relay.nn.batch_norm(x, gamma, beta, moving_mean, moving_var)
        func = relay.Function(relay.analysis.free_vars(act[0]), act[0])
        relay_ir = tvm.ir.IRModule.from_expr(func)

        qnn_ir = relay.qnn.op.csi_batch_norm(
            x, gamma, beta, moving_mean, moving_var, 1, 0.00001, True, True, None, "btnorm"
        )

        x_data = np.random.uniform(low=-1, high=1, size=shape).astype("float32")
        beta = np.random.uniform(low=-1, high=1, size=(shape[axis],)).astype("float32")
        gamma = np.random.uniform(low=-1, high=1, size=(shape[axis],)).astype("float32")
        moving_mean = np.random.uniform(low=-1, high=1, size=(shape[axis],)).astype("float32")
        moving_var = np.random.uniform(low=-1, high=1, size=(shape[axis],)).astype("float32")

        verify_results(
            relay_ir,
            qnn_ir,
            [x_data, beta, gamma, moving_mean, moving_var],
            "test_batch_norm",
            rtol=1e-5,
            atol=1e-5,
        )

    verify_batch_norm((1, 5, 32, 32), 1)
    verify_batch_norm((1, 3, 2, 2), 1)
    verify_batch_norm((2, 4, 2, 2), 1)
    verify_batch_norm((1, 32, 4, 6), 1)


def test_bias_add():
    def verify_bias_add(shape, axis, quant_scheme="float32"):
        # relay ir
        x = relay.var("x", relay.TensorType(shape, "float32"))
        bias_value = np.random.uniform(-1, 1, (shape[axis])).astype("float32")
        bias_value = relay.const(bias_value)

        z = relay.nn.bias_add(x, bias_value, axis)
        func = relay.Function([x], z)
        relay_ir = tvm.IRModule()
        relay_ir["main"] = func
        # convert to qnn
        qnn_ir = quantize_relay_ir(relay_ir, quant_scheme=quant_scheme)
        x_data = np.ones(shape).astype("float32")
        verify_results(relay_ir, qnn_ir, [x_data], "test_bias_add", rtol=1e-5, atol=1e-5)

    verify_bias_add((1, 2, 3), 1)
    verify_bias_add((1, 3, 5, 4), 1)
    verify_bias_add((1, 3, 32, 23), 1)
    verify_bias_add((2, 6, 32, 32), 1)
    verify_bias_add((2, 6, 32, 28, 34), 1)
    verify_bias_add((1, 3, 32), 1)
    verify_bias_add((1, 6), 1)
    verify_bias_add((2, 3, 3), 1)
    verify_bias_add((3, 6), 1)


def test_batch_matmul():
    def verify_matmul(x_shape=(12, 128, 64), y_shape=(12, 128, 64), transa=False, transb=True):
        x = relay.var("x", shape=(x_shape), dtype="float32")
        y = relay.var("y", shape=(y_shape), dtype="float32")
        out = relay.nn.batch_matmul(x, y, transpose_a=transa, transpose_b=transb)
        f = relay.Function([x, y], out)
        relay_ir = tvm.IRModule().from_expr(f)
        x_data = np.random.uniform(low=-100, high=1000, size=x_shape).astype("float32")
        y_data = np.random.uniform(low=-10, high=10, size=y_shape).astype("float32")

        quant_scheme = "float32"
        qnn_ir = quantize_relay_ir(relay_ir, quant_scheme=quant_scheme)
        verify_results(
            relay_ir, qnn_ir, [x_data, y_data], "test_batch_matmul", rtol=1e-5, atol=1e-5
        )

    verify_matmul(x_shape=(2, 3, 4), y_shape=(2, 4, 3), transa=True, transb=True)
    verify_matmul(x_shape=(2, 4, 3), y_shape=(2, 3, 4), transa=False, transb=False)
    verify_matmul(x_shape=(2, 4, 3), y_shape=(2, 4, 3), transa=False, transb=True)
    verify_matmul(x_shape=(2, 4, 3), y_shape=(2, 4, 3), transa=True, transb=False)


def test_conv1d():
    def verify_conv1d(
        scale,
        dshape,
        kshape,
        padding=0,
        groups=1,
        dilation=1,
        with_bias=True,
        quant_scheme="float32",
        **attrs,
    ):
        x = relay.var("x", shape=dshape, dtype="float32")

        w_value = np.random.uniform(-scale, scale, size=kshape).astype("float32")
        w = relay.const(w_value)
        y = relay.nn.conv1d(x, w, padding=padding, dilation=dilation, groups=groups, **attrs)

        if with_bias:
            y_shape = infer_shape(y)
            b_value = np.random.uniform(-scale, scale, size=(y_shape[1],)).astype("float32")
            b = relay.const(b_value)

            z = relay.nn.bias_add(y, b)

            func = relay.Function([x], z)
        else:
            func = relay.Function([x], y)

        relay_ir = tvm.IRModule()
        relay_ir["main"] = func
        qnn_ir = quantize_relay_ir(relay_ir, quant_scheme=quant_scheme)

        data = np.random.uniform(-scale, scale, size=dshape).astype("float32")

        verify_results(relay_ir, qnn_ir, [data], "test_conv1d", rtol=1e-5, atol=1e-5, is_dyn=True)

    dshape = (1, 1, 3)
    kshape = (1, 1, 2)
    verify_conv1d(1, dshape, kshape, padding=0, channels=1, groups=1, kernel_size=2)
    dshape = (1, 2, 6)
    kshape = (1, 2, 3)
    verify_conv1d(1, dshape, kshape, padding=0, channels=1, groups=1, kernel_size=3)
    dshape = (1, 2, 64)
    kshape = (2, 2, 5)
    verify_conv1d(1, dshape, kshape, padding=0, channels=2, groups=1, kernel_size=5)


def test_cast():
    def verify_cast(shape, dtype, quant_scheme="float32"):
        # relay ir
        x = relay.var("x", relay.TensorType(shape, "float32"))
        z = relay.cast(x, dtype)
        func = relay.Function([x], z)
        relay_ir = tvm.IRModule()
        relay_ir["main"] = func

        # convert to qnn
        qnn_ir = quantize_relay_ir(relay_ir, quant_scheme=quant_scheme)

        x_data = np.random.uniform(low=-1, high=1, size=shape).astype("float32")
        verify_results(relay_ir, qnn_ir, [x_data], "test_cast", rtol=1e-5, atol=1e-5)

    verify_cast((1, 4, 112, 112), "float16")
    verify_cast((1, 3, 32, 112), "float16")
    verify_cast((2, 4, 112), "float16")
    verify_cast((2, 8), "float16")
    verify_cast((4,), "float16")


def test_clip():
    def verify_clip(shape, a_min, a_max, quant_scheme="float32"):
        # relay ir
        x = relay.var("x", relay.TensorType(shape, "float32"))
        z = relay.clip(x, a_min, a_max)
        func = relay.Function([x], z)
        relay_ir = tvm.IRModule()
        relay_ir["main"] = func

        # convert to qnn
        qnn_ir = quantize_relay_ir(relay_ir, quant_scheme=quant_scheme)
        x_data = np.random.uniform(low=-5, high=5, size=shape).astype("float32")
        verify_results(relay_ir, qnn_ir, [x_data], "test_clip", rtol=1e-5, atol=1e-5)

    verify_clip((1, 4), -1, 2)
    verify_clip((1, 4, 32), -0.5, 0.5)
    verify_clip((1, 4, 32, 32), -2, 3)
    verify_clip((1, 4, 1, 1), -5, 2)


def test_deconv2d():
    def verify_deconv2d(
        scale,
        dshape,
        kshape,
        padding=(1, 1),
        groups=1,
        dilation=(1, 1),
        with_bias=True,
        quant_scheme="float32",
        **attrs,
    ):
        x = relay.var("x", shape=dshape, dtype="float32")

        w_value = np.random.uniform(-scale, scale, size=kshape).astype("float32")
        w = relay.const(w_value)
        y = relay.nn.conv2d_transpose(
            x, w, padding=padding, dilation=dilation, groups=groups, **attrs
        )

        if with_bias:
            y_shape = infer_shape(y)
            b_value = np.random.uniform(-scale, scale, size=(y_shape[1],)).astype("float32")
            b = relay.const(b_value)

            z = relay.nn.bias_add(y, b)

            func = relay.Function([x], z)
        else:
            func = relay.Function([x], y)

        relay_ir = tvm.IRModule()
        relay_ir["main"] = func
        qnn_ir = quantize_relay_ir(relay_ir, quant_scheme=quant_scheme)

        data = np.random.uniform(-scale, scale, size=dshape).astype("float32")
        verify_results(relay_ir, qnn_ir, [data], "test_deconv2d", rtol=1e-5, atol=1e-5, is_dyn=True)

    dshape = (1, 32, 18, 18)
    kshape = (32, 1, 3, 3)
    verify_deconv2d(1, dshape, kshape, padding=(1, 1), channels=32, groups=32, kernel_size=(3, 3))

    dshape = (3, 64, 112, 112)
    kshape = (64, 2, 5, 5)
    verify_deconv2d(1, dshape, kshape, padding=(1, 1), channels=64, groups=32, kernel_size=(5, 5))

    dshape = (1, 16, 112, 112)
    kshape = (16, 2, 1, 1)
    verify_deconv2d(1, dshape, kshape, padding=(1, 1), channels=16, groups=8, kernel_size=(1, 1))


def test_depth_to_space():
    def verify_depth_to_space(shape, block_size, layout, mode, quant_scheme="float32"):
        # relay ir
        x = relay.var("x", relay.TensorType(shape, "float32"))
        z = relay.nn.depth_to_space(x, block_size, layout, mode)
        func = relay.Function([x], z)
        relay_ir = tvm.IRModule()
        relay_ir["main"] = func

        # convert to qnn
        qnn_ir = quantize_relay_ir(relay_ir, quant_scheme=quant_scheme)

        x_data = np.random.uniform(low=-1, high=1, size=shape).astype("float32")
        verify_results(relay_ir, qnn_ir, [x_data], "test_depth_to_space", rtol=1e-5, atol=1e-5)

    verify_depth_to_space((1, 4, 4, 4), 2, "NCHW", "DCR")
    verify_depth_to_space((1, 9, 112, 112), 3, "NCHW", "DCR")
    verify_depth_to_space((1, 16, 32, 32), 4, "NCHW", "DCR")


def test_space_to_depth():
    def verify_space_to_depth(shape, block_size, layout, quant_scheme="float32"):
        # relay ir
        x = relay.var("x", relay.TensorType(shape, "float32"))
        z = relay.nn.space_to_depth(x, block_size, layout)
        func = relay.Function([x], z)
        relay_ir = tvm.IRModule()
        relay_ir["main"] = func

        # convert to qnn
        qnn_ir = quantize_relay_ir(relay_ir, quant_scheme=quant_scheme)

        x_data = np.random.uniform(low=-1, high=1, size=shape).astype("float32")
        verify_results(relay_ir, qnn_ir, [x_data], "test_space_to_depth", rtol=1e-5, atol=1e-5)

    verify_space_to_depth((1, 4, 4, 4), 2, "NCHW")
    verify_space_to_depth((1, 3, 9, 9), 3, "NCHW")
    verify_space_to_depth((1, 16, 16, 32), 4, "NCHW")
    verify_space_to_depth((1, 4, 8, 16), 2, "NCHW")


def test_lrn():
    def verify_lrn(shape, size, axis, bias, alpha, beta, dtype, quant_scheme="float32"):
        data = tvm.nd.array((np.random.uniform(low=-1, high=1, size=shape).astype(dtype)))
        data = relay.var("x", relay.TensorType(shape, dtype))
        func = relay.nn.lrn(data, size=size, axis=axis, bias=bias, alpha=alpha, beta=beta)
        relay_ir = tvm.IRModule().from_expr(func)
        qnn_ir = quantize_relay_ir(relay_ir, quant_scheme=quant_scheme)

        x_data = np.random.uniform(low=-1, high=1, size=shape).astype("float32")
        verify_results(relay_ir, qnn_ir, [x_data], "test_lrn", rtol=1e-5, atol=1e-5)

    verify_lrn((3, 2, 1, 1), 5, 1, 2, 1e-05, 0.75, "float32")
    verify_lrn((3, 2, 3, 4), 5, 1, 2, 1e-03, 0.5, "float32")
    verify_lrn((3, 2, 1, 4), 5, 1, 2, 1e-04, 0.25, "float32")


def test_pad():
    def verify_pad(shape, pad_width, pad_value=0.0, pad_mode="constant", quant_scheme="float32"):
        # relay ir
        x = relay.var("x", relay.TensorType(shape, "float32"))
        z = relay.nn.pad(x, pad_width, pad_value, pad_mode)
        func = relay.Function([x], z)
        relay_ir = tvm.IRModule()
        relay_ir["main"] = func

        # convert to qnn
        qnn_ir = quantize_relay_ir(relay_ir, quant_scheme=quant_scheme)
        x_data = np.random.uniform(low=-1, high=1, size=shape).astype("float32")
        verify_results(relay_ir, qnn_ir, [x_data], "test_pad", rtol=1e-5, atol=1e-5)

    verify_pad((3, 2), ((0, 0), (2, 0)), 1.0, "constant", "float32")
    verify_pad((3, 2, 4), ((0, 0), (2, 0), (0, 3)), 1.0, "constant", "float32")
    verify_pad((3, 2, 4, 2), ((0, 0), (2, 0), (0, 3), (0, 0)), 0.0, "constant", "float32")
    verify_pad((3, 4), ((0, 0), (0, 2)), 1.0, "reflect", "float32")
    verify_pad((3, 2, 4), ((0, 0), (2, 0), (0, 3)), 1.0, "edge", "float32")


def test_prelu():
    def verify_prelu(shape, alpha_shape, axis, quant_scheme="float32"):
        x = relay.var("x", relay.TensorType(shape, "float32"))
        alpha = relay.var("alpha", relay.TensorType(alpha_shape, "float32"))
        z = relay.nn.prelu(x, alpha, axis)
        func = relay.Function([x, alpha], z)
        relay_ir = tvm.IRModule()
        relay_ir["main"] = func

        # convert to qnn
        qnn_ir = quantize_relay_ir(relay_ir, quant_scheme=quant_scheme)
        x_data = np.random.uniform(low=-1, high=1, size=shape).astype("float32")
        p_data = np.random.uniform(low=-1, high=1, size=alpha_shape).astype("float32")
        verify_results(relay_ir, qnn_ir, [x_data, p_data], "test_prelu", rtol=1e-5, atol=1e-5)

    verify_prelu((3, 2, 32, 32), (32,), 3, "float32")
    verify_prelu((3, 2, 32), (32,), 2, "float32")
    verify_prelu(
        (
            3,
            2,
        ),
        (2,),
        1,
        "float32",
    )


def test_strided_slice():
    def verify_strided_slice(
        shape,
        begin,
        end,
        strides,
        axes=None,
        slice_mode="end",
        layout="NCHW",
        quant_scheme="float32",
    ):
        x = relay.var("data", relay.TensorType(shape, "float32"))
        ndim = len(shape)
        begin = begin if begin else [0] * ndim
        end = end if end else list(shape)
        x_data = np.random.uniform(size=shape).astype("float32")
        if strides:
            z = relay.strided_slice(
                x, begin=begin, end=end, strides=strides, axes=axes, slice_mode=slice_mode
            )
        else:
            z = relay.strided_slice(x, begin=begin, end=end, axes=axes, slice_mode=slice_mode)

        func = relay.Function([x], z)
        relay_ir = tvm.IRModule()
        relay_ir["main"] = func
        relay_results = run_relay(relay_ir["main"], [x_data], "false")
        # convert to qnn
        qnn_ir = quantize_relay_ir(relay_ir, quant_scheme=quant_scheme)
        x_data = np.random.uniform(low=-1, high=1, size=shape).astype("float32")
        verify_results(relay_ir, qnn_ir, [x_data], "test_strided_slice", rtol=1e-5, atol=1e-5)

    verify_strided_slice((1, 3, 4, 4), [0, 1, 0, 0], [1, 3, 2, 2], [1, 1, 1, 2])
    verify_strided_slice((2, 4), [0, 1], [2, 3], [1, 1])
    verify_strided_slice((2, 4), [0, 1], [-1, 1000], [1, 2])


def test_take():
    def verify_take(src_shape, indices_src, axis, mode, indices_dtype, quant_scheme="float32"):
        src_dtype = "float32"
        indices_src = np.array(indices_src, dtype=indices_dtype)
        x = relay.var("x", relay.TensorType(src_shape, src_dtype))
        indices = relay.var("indices", relay.TensorType(indices_src.shape, indices_dtype))
        z = relay.take(x, indices, axis=axis, mode=mode)
        func = relay.Function([x, indices], z)
        relay_ir = tvm.IRModule().from_expr(func)
        indices_src = np.array(indices_src, dtype=indices_dtype)
        x_data = np.random.uniform(low=-1, high=1, size=src_shape).astype(src_dtype)
        qnn_ir = quantize_relay_ir(relay_ir, quant_scheme=quant_scheme)
        verify_results(relay_ir, qnn_ir, [x_data, indices_src], "test_take", rtol=1e-5, atol=1e-5)

    verify_take((4,), [1], 0, "clip", "int64")
    verify_take((3, 2), [[[0, 1], [1, 2]]], 0, "clip", "int64")
    verify_take((2, 2), [[[1, 0], [0, 1]]], 1, "clip", "int64")
    verify_take((4, 3, 5, 6), [[2, 1, 0, 0]], -2, "clip", "int64")


def test_upsampling():
    def verify_upsampling(in_shape, value_range, scale, quant_scheme="float32"):
        x = relay.var("x", shape=in_shape, dtype="float32")
        out_h = int(in_shape[2] * scale)
        out_w = int(in_shape[3] * scale)
        y = relay.image.resize2d(x, (out_h, out_w))

        func = relay.Function([x], y)
        relay_ir = tvm.IRModule()
        relay_ir["main"] = func
        qnn_ir = quantize_relay_ir(relay_ir, quant_scheme=quant_scheme)

        data = np.random.uniform(-value_range, value_range, size=in_shape).astype("float32")
        verify_results(relay_ir, qnn_ir, [data], "test_upsampling", rtol=1e-5, atol=1e-4)

    verify_upsampling((1, 3, 224, 224), 255, 2)
    verify_upsampling((1, 3, 12, 12), 255, 4)


def test_mean():
    def verify_mean(
        in_shape, value_range, axis=None, keepdims=False, exclude=False, quant_scheme="float32"
    ):
        x = relay.var("x", shape=in_shape, dtype="float32")

        y = relay.mean(x, axis=axis, keepdims=keepdims, exclude=exclude)

        func = relay.Function([x], y)
        relay_ir = tvm.IRModule()
        relay_ir["main"] = func
        qnn_ir = quantize_relay_ir(relay_ir, quant_scheme=quant_scheme)

        data = np.random.uniform(-value_range, value_range, size=in_shape).astype("float32")
        verify_results(relay_ir, qnn_ir, [data], "test_upsampling", rtol=1e-5, atol=1e-5)

    verify_mean((1, 3, 224, 224), 255)
    verify_mean((1, 3, 224, 224), 255, 1)
    verify_mean((1, 3, 224, 224), 255, (2, 3))
    verify_mean((1, 3, 224, 224), 255, 1, True)


def test_squeeze():
    def verify_squeeze(shape, axis, layout="NCHW", quant_scheme="float32"):
        # relay ir
        x = relay.var("x", relay.TensorType(shape, "float32"))
        if axis == None:
            z = relay.squeeze(x, axis=None)
        else:
            z = relay.squeeze(x, axis=[axis])
        func = relay.Function([x], z)
        relay_ir = tvm.IRModule()
        relay_ir["main"] = func

        # convert to qnn
        qnn_ir = quantize_relay_ir(relay_ir, quant_scheme=quant_scheme)
        x_data = np.random.uniform(low=-1, high=1, size=shape).astype("float32")
        verify_results(relay_ir, qnn_ir, [x_data], "test_squeeze", rtol=1e-5, atol=1e-5)

    verify_squeeze((2, 1, 2, 3), 1)
    verify_squeeze((2, 1, 32), 1)
    verify_squeeze((2, 3, 1), 2)


def test_quantize():
    def verify_quantize(shape, axis, out_dtype, quant_scheme="int8"):
        # relay ir
        x = relay.var("x", relay.TensorType(shape, "float32"))
        scale = relay.var("scale", relay.TensorType((shape[axis],), "float32"))
        # zero_point = relay.var("zp", relay.TensorType((shape[axis], ), "int32"))
        zero_point = relay.const(np.random.rand((shape[axis])).astype("int32"))
        z = relay.qnn.op.csi_quantize(
            x,
            scale,
            zero_point,
            axis,
            out_dtype,
            None,
        )
        relay_ir = tvm.IRModule().from_expr(z)

        x_data = np.random.uniform(low=-10, high=10, size=shape).astype("float32")
        scale_data = np.random.uniform(low=-10, high=10, size=(shape[axis],)).astype("float32")
        zp_data = np.random.uniform(low=-10, high=10, size=(shape[axis],)).astype("int32")

        qnn_ir = relay_ir
        qnn_ir = relay.transform.InferType()(qnn_ir)
        onnx_results = run_onnx(
            qmod_to_onnx(qnn_ir, "test_quantize"), [x_data, scale_data, zp_data]
        )

    verify_quantize((1, 3, 2, 1), 1, "int8")
    verify_quantize((1, 4, 3, 3), 1, "uint8")


def test_dequantize():
    def verify_dequantize(shape, axis, out_dtype):
        # relay ir
        x = relay.const(np.random.randint(10, size=shape).astype("int8"))
        scale = relay.const(np.random.rand((shape[axis])).astype("float32"))
        zero_point = relay.const(np.random.rand((shape[axis])).astype("int32"))
        z = relay.qnn.op.csi_dequantize(x, scale, zero_point, axis, out_dtype, None)
        relay_ir = tvm.IRModule().from_expr(z)

        x_data = np.random.uniform(low=-100, high=1000, size=shape).astype("int8")
        scale_data = np.random.uniform(low=-10, high=10, size=(shape[axis],)).astype("float32")
        zp_data = np.random.uniform(low=-1000, high=1000, size=(shape[axis],)).astype("int32")

        qnn_ir = relay_ir
        qnn_ir = relay.transform.InferType()(qnn_ir)
        onnx_results = run_onnx(
            qmod_to_onnx(qnn_ir, "test_dequantize"), [x_data, scale_data, zp_data]
        )

    verify_dequantize((1, 3, 2, 2), 1, "float32")
    verify_dequantize((3, 32, 224, 224), 1, "float32")


if __name__ == "__main__":
    test_argmax()
    test_add()
    test_avgpool2d()
    test_batch_norm()
    test_bias_add()
    test_batch_matmul()
    test_cast()
    test_clip()
    test_concatenate()
    test_conv1d()
    test_conv2d()
    test_conv2d_relu()
    test_conv2d_relu6()
    test_deconv2d()
    test_dequantize()
    test_depth_to_space()
    test_dense()
    test_div()
    test_expand_dims()
    test_flatten()
    test_global_max_pool2d()
    test_global_avg_pool2d()
    test_leaky_relu()
    test_lrn()
    test_log_softmax()
    test_maxpool2d()
    test_mean()
    test_mul()
    test_pad()
    test_prelu()
    test_quantize()
    test_relu()
    test_relu6()
    test_reshape()
    test_split()
    test_subtract()
    test_softmax()
    test_space_to_depth()
    test_strided_slice()
    test_squeeze()
    test_sigmoid()
    test_transpose()
    test_take()
    test_upsampling()
