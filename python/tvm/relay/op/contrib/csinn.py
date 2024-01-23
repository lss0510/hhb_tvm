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
"""csinn library supported operators.
There are two ways to registering a function for an op to indicate if it is
supported by csinn.

- The first and simplest way is to use the helper so that
users only need to provide the operator name and a boolean value to indicate if
it is supported. For example:

    .. code-block:: python

      add = _register_external_op_helper("add")
      add = _register_external_op_helper("add", True)
      add = _register_external_op_helper("add", False)

- The other way is to implement the function by themselves to
check the attributes of the op and decide if it should be offloaded to DNNL.
"""
import tvm.ir
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name


def is_csinn_runtime_enabled():
    """Check if the csinn runtime is present.

    Returns
    -------
    ret: bool
        True if present, False if not.
    """
    check_enabled = tvm.get_global_func("relay.op.is_csinn_runtime_enabled", True)
    if check_enabled:
        return check_enabled()
    return False


def _register_external_op_helper(op_name, supported=True):
    """The helper function to indicate that a given operator can be supported
    by csinn.

    Paramters
    ---------
    op_name : Str
        The name of operator that will be registered.

    Returns
    -------
    f : callable
        A function that returns if the operator is supported by DNNL.
    """

    @tvm.ir.register_op_attr(op_name, "target.csinn")
    def _func_wrapper(expr):
        return supported

    return _func_wrapper


_register_external_op_helper("qnn.csi.abs")
_register_external_op_helper("qnn.csi.acos")
_register_external_op_helper("qnn.csi.acosh")
_register_external_op_helper("qnn.csi.add")
_register_external_op_helper("qnn.csi.argmax")
_register_external_op_helper("qnn.csi.argmin")
_register_external_op_helper("qnn.csi.asin")
_register_external_op_helper("qnn.csi.asinh")
_register_external_op_helper("qnn.csi.atan")
_register_external_op_helper("qnn.csi.atanh")
_register_external_op_helper("qnn.csi.avgpool2d")
_register_external_op_helper("qnn.csi.avgpool3d")
_register_external_op_helper("qnn.csi.batch_to_space_nd")
_register_external_op_helper("qnn.csi.bias_add")
_register_external_op_helper("qnn.csi.broadcast_to")
_register_external_op_helper("qnn.csi.cast")
_register_external_op_helper("qnn.csi.ceil")
_register_external_op_helper("qnn.csi.cache_matmul")
_register_external_op_helper("qnn.csi.cache_conv1d")
_register_external_op_helper("qnn.csi.clip")
_register_external_op_helper("qnn.csi.concatenate")
_register_external_op_helper("qnn.csi.conv1d")
_register_external_op_helper("qnn.csi.conv2d")
_register_external_op_helper("qnn.csi.conv2d_relu")
_register_external_op_helper("qnn.csi.conv2d_relu6")
_register_external_op_helper("qnn.csi.conv3d")
_register_external_op_helper("qnn.csi.cos")
_register_external_op_helper("qnn.csi.cosh")
_register_external_op_helper("qnn.csi.crop_resize")
_register_external_op_helper("qnn.csi.deconv2d")
_register_external_op_helper("qnn.csi.deconv3d")
_register_external_op_helper("qnn.csi.dense")
_register_external_op_helper("qnn.csi.depth_to_space")
_register_external_op_helper("qnn.csi.dilation2d")
_register_external_op_helper("qnn.csi.div")
_register_external_op_helper("qnn.csi.equal")
_register_external_op_helper("qnn.csi.erf")
_register_external_op_helper("qnn.csi.exp")
_register_external_op_helper("qnn.csi.expand_dims")
_register_external_op_helper("qnn.csi.flatten")
_register_external_op_helper("qnn.csi.floor")
_register_external_op_helper("qnn.csi.floor_div")
_register_external_op_helper("qnn.csi.floor_mod")
_register_external_op_helper("qnn.csi.fsmn")
_register_external_op_helper("qnn.csi.full")
_register_external_op_helper("qnn.csi.global_avgpool2d")
_register_external_op_helper("qnn.csi.global_maxpool2d")
_register_external_op_helper("qnn.csi.leaky_relu")
_register_external_op_helper("qnn.csi.left_shift")
_register_external_op_helper("qnn.csi.less")
_register_external_op_helper("qnn.csi.log")
_register_external_op_helper("qnn.csi.layer_norm")
_register_external_op_helper("qnn.csi.log_softmax")
_register_external_op_helper("qnn.csi.lrn")
_register_external_op_helper("qnn.csi.max")
_register_external_op_helper("qnn.csi.maxpool3d")
_register_external_op_helper("qnn.csi.maximum")
_register_external_op_helper("qnn.csi.matmul")
_register_external_op_helper("qnn.csi.maxpool2d")
_register_external_op_helper("qnn.csi.maxpool2d_locat")
_register_external_op_helper("qnn.csi.maxpool2d_with_argmax")
_register_external_op_helper("qnn.csi.mean")
_register_external_op_helper("qnn.csi.min")
_register_external_op_helper("qnn.csi.minimum")
_register_external_op_helper("qnn.csi.mod")
_register_external_op_helper("qnn.csi.mul")
_register_external_op_helper("qnn.csi.negative")
_register_external_op_helper("qnn.csi.one_hot")
_register_external_op_helper("qnn.csi.pad")
_register_external_op_helper("qnn.csi.power")
_register_external_op_helper("qnn.csi.prelu")
_register_external_op_helper("qnn.csi.prod")
_register_external_op_helper("qnn.csi.proposal")
_register_external_op_helper("qnn.csi.psroipooling")
_register_external_op_helper("qnn.csi.relu")
_register_external_op_helper("qnn.csi.relu6")
_register_external_op_helper("qnn.csi.reshape")
_register_external_op_helper("qnn.csi.reverse")
_register_external_op_helper("qnn.csi.right_shift")
_register_external_op_helper("qnn.csi.roipooling")
_register_external_op_helper("qnn.csi.round")
_register_external_op_helper("qnn.csi.scatter_nd")
_register_external_op_helper("qnn.csi.segment_max")
_register_external_op_helper("qnn.csi.segment_mean")
_register_external_op_helper("qnn.csi.segment_min")
_register_external_op_helper("qnn.csi.segment_prod")
_register_external_op_helper("qnn.csi.segment_sum")
_register_external_op_helper("qnn.csi.sigmoid")
_register_external_op_helper("qnn.csi.sign")
_register_external_op_helper("qnn.csi.sin")
_register_external_op_helper("qnn.csi.sinh")
_register_external_op_helper("qnn.csi.softmax")
_register_external_op_helper("qnn.csi.space_to_batch_nd")
_register_external_op_helper("qnn.csi.space_to_depth")
_register_external_op_helper("qnn.csi.split")
_register_external_op_helper("qnn.csi.sqrt")
_register_external_op_helper("qnn.csi.rsqrt")
_register_external_op_helper("qnn.csi.squeeze")
_register_external_op_helper("qnn.csi.strided_slice")
_register_external_op_helper("qnn.csi.subtract")
_register_external_op_helper("qnn.csi.sum")
_register_external_op_helper("qnn.csi.take")
_register_external_op_helper("qnn.csi.tan")
_register_external_op_helper("qnn.csi.tanh")
_register_external_op_helper("qnn.csi.tile")
_register_external_op_helper("qnn.csi.transpose")
_register_external_op_helper("qnn.csi.unpooling")
_register_external_op_helper("qnn.csi.upsampling")
_register_external_op_helper("qnn.csi.where")
_register_external_op_helper("qnn.csi.where_softmax")
_register_external_op_helper("qnn.csi.quantize")
_register_external_op_helper("qnn.csi.dequantize")
_register_external_op_helper("qnn.csi.data_convert")


def partition_for_csinn(mod, params=None, **opts):
    """Partition the graph greedily offloading supported operators to csinn.

    Parameters
    ----------
    mod : Module
        The module to run passes on.
    params : Optional[Dict[str, NDArray]]
        Constant input parameters.

    Returns
    -------
    ret : annotated and partitioned module.
    """
    if params:
        mod["main"] = bind_params_by_name(mod["main"], params)

    seq = tvm.transform.Sequential(
        [
            transform.InferType(),
            transform.AnnotateTarget("csinn", True),
            transform.MergeCompilerRegions(),
            transform.PartitionGraph(),
        ]
    )
    return seq(mod)
