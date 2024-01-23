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
# pylint: disable=invalid-name, unused-argument
"""Convert csinn model layout."""
import logging
import numpy as np

from tvm import relay
from tvm.relay import op as _op
from tvm.relay import function as _function
from tvm.relay.expr import Constant, Tuple

from .base import _qnn_attrs
from ..quantization.calibrate import PER_TENSOR, PER_CHANNEL, USE_MINMAX, USE_SCALE

LOG = 25
logger = logging.getLogger("HHB")
RELAY_FUNCS = {}


def attrs_changer(attrs):
    """Remove useless attributes"""
    attrs = _qnn_attrs(attrs)
    if "layer_name" in attrs:
        attrs.pop("layer_name")
    if "out_dtype" in attrs:
        attrs.pop("out_dtype")
    return attrs, attrs.pop("q_params")


def relay_func_register(func_name):
    """Register func in RELAY_FUNCS"""

    def decorator(func):
        RELAY_FUNCS[func_name] = func.__name__

        def wrapper(self, call, op_args):
            attrs, q_params = attrs_changer(call.attrs)
            return func(self, op_args, attrs, q_params)

        return wrapper

    return decorator


def convert_to_relay(mod):
    """Convert quanted model to float"""

    class Convert2Relay(relay.ExprMutator):
        """Convert qnn model to relay"""

        def __init__(self):
            super(Convert2Relay, self).__init__()
            self.unregistered_func = []

        def constant_convert(self, src_constat, q_params):
            """Convert constant value to relay"""
            if isinstance(src_constat, Constant):
                np_value = src_constat.data.asnumpy().astype("float32")
                # tensor_type = q_params[0] # CONST ACTIVATION
                value_type = q_params[1]  # USE_MINMAX USE_SCALE
                q_type = q_params[2]  # PER_TENSOR PER_CHANNEL
                if q_type == PER_TENSOR:
                    if value_type == USE_SCALE:
                        scale = q_params[3]
                        zp = q_params[4]
                        np_value = scale * (np_value - zp)
                    elif value_type == USE_MINMAX:
                        raise Exception(f"not registe convert const with 'USE_MINMAX'.")
                    else:
                        raise Exception(f"get err q_type {q_type}.")
                elif q_type == PER_CHANNEL:
                    if value_type == USE_SCALE:
                        r_idx = [-1 if i == 0 else 1 for i, _ in enumerate(np_value.shape)]
                        q_info = np.array(q_params[3:]).reshape([-1, 2])
                        scales = q_info[:, 0].reshape(r_idx)
                        zps = q_info[:, 1].reshape(r_idx)
                        np_value = scales * (np_value - zps)
                    elif value_type == USE_MINMAX:
                        raise Exception(f"not registe convert const with 'USE_MINMAX'.")
                    else:
                        raise Exception(f"get err q_type {q_type}.")
                else:
                    raise Exception(f"get err value_type {value_type}.")
                np_value = np_value.astype("float32")
                return relay.const(np_value, str(np_value.dtype))

            return src_constat

        def visit_call(self, call):
            op_args = [self.visit(arg) for arg in call.args]
            if call.op.name in RELAY_FUNCS:
                func = getattr(self, RELAY_FUNCS[call.op.name])
                new_call = func(call=call, op_args=op_args)
            else:
                raise Exception(f"{call.op.name} not registed.")

            return new_call

        def diso_convert(self, op_args, attrs, q_params, relay_op):
            op_args[1] = self.constant_convert(op_args[1], q_params[1])
            return relay_op(*op_args, **attrs)

        def siso_convert(self, op_args, attrs, relay_op):
            return relay_op(*op_args, **attrs)

        @relay_func_register("qnn.csi.conv2d")
        def conv2d(self, op_args, attrs, q_params):
            """convert conv2d to relay"""
            data = op_args[0]
            weight = self.constant_convert(op_args[1], q_params[1])
            bias = self.constant_convert(op_args[2], q_params[2])
            out = _op.nn.conv2d(data, weight, **attrs)
            if bias.data.numpy().size == attrs["channels"]:
                out = _op.nn.bias_add(out, bias)
            return out

        @relay_func_register("qnn.csi.relu")
        def relu(self, op_args, attrs, q_params):
            """convert relu to relay"""
            return _op.nn.relu(*op_args)

        @relay_func_register("qnn.csi.relu6")
        def relu6(self, op_args, attrs, q_params):
            """convert relu6 to relay"""
            return _op.clip(*op_args, 0.0, 6.0)

        @relay_func_register("qnn.csi.reshape")
        def reshape(self, op_args, attrs, q_params):
            """convert reshape to relay"""
            return self.siso_convert(op_args, attrs, _op.reshape)

        @relay_func_register("qnn.csi.depth_to_space")
        def depth_to_space(self, op_args, attrs, q_params):
            """convert depth_to_space to relay"""
            return self.siso_convert(op_args, attrs, _op.nn.depth_to_space)

        @relay_func_register("qnn.csi.softmax")
        def softmax(self, op_args, attrs, q_params):
            """convert softmax to relay"""
            return _op.nn.softmax(*op_args, **attrs)

        @relay_func_register("qnn.csi.squeeze")
        def squeeze(self, op_args, attrs, q_params):
            """convert squeeze to relay"""
            return self.siso_convert(op_args, attrs, _op.squeeze)

        # DISO
        @relay_func_register("qnn.csi.subtract")
        def subtract(self, op_args, attrs, q_params):
            """convert subtract to relay"""
            return self.diso_convert(op_args, attrs, q_params, _op.subtract)

        @relay_func_register("qnn.csi.mul")
        def mul(self, op_args, attrs, q_params):
            """convert mul to relay"""
            return self.diso_convert(op_args, attrs, q_params, _op.multiply)

        @relay_func_register("qnn.csi.add")
        def add(self, op_args, attrs, q_params):
            """convert add to relay"""
            return self.diso_convert(op_args, attrs, q_params, _op.add)

        @relay_func_register("qnn.csi.div")
        def div(self, op_args, attrs, q_params):
            """convert div to relay"""
            return self.diso_convert(op_args, attrs, q_params, _op.divide)

        @relay_func_register("qnn.csi.minimum")
        def minimum(self, op_args, attrs, q_params):
            """convert minimum to relay"""
            return self.diso_convert(op_args, attrs, q_params, _op.minimum)

        @relay_func_register("qnn.csi.avgpool2d")
        def avgpool2d(self, op_args, attrs, q_params):
            """convert avgpool2d to relay"""
            return self.siso_convert(op_args, attrs, _op.nn.avg_pool2d)

        @relay_func_register("qnn.csi.concatenate")
        def concatenate(self, op_args, attrs, q_params):
            """convert concatenate to relay"""
            new_args = []
            for i, arg in enumerate(op_args[0]):
                new_args.append(self.constant_convert(arg, q_params[i]))
            return _op.concatenate(Tuple(new_args), **attrs)

        @relay_func_register("qnn.csi.dense")
        def dense(self, op_args, attrs, q_params):
            """convert dense to relay"""
            units = attrs["units"]
            data = op_args[0]
            weight = self.constant_convert(op_args[1], q_params[1])
            bias = self.constant_convert(op_args[2], q_params[2])
            out = _op.nn.dense(data, weight, units)
            if bias.data.numpy().size == units:
                out = _op.nn.bias_add(out, bias)
            return out

        def visit_function(self, fn):
            new_params = [self.visit(x) for x in fn.params]
            new_body = self.visit(fn.body)
            return _function.Function(list(new_params), new_body)

    mod["main"] = Convert2Relay().visit(mod["main"])

    return mod
