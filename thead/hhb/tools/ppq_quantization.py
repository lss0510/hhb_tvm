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
"""
Quantization tools with PPQ.
"""
import os
import logging
from typing import Any, Callable, List, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from ppq import (
    BaseGraph,
    BaseQuantizer,
    Operation,
    OperationQuantizationConfig,
    QuantizationPolicy,
    QuantizationProperty,
    QuantizationStates,
    TargetPlatform,
    TorchExecutor,
    graphwise_error_analyse,
    layerwise_error_analyse,
)
from ppq.api import load_onnx_graph, export_ppq_graph
from ppq.core import RoundingPolicy, PASSIVE_OPERATIONS
from ppq.scheduler import DISPATCHER_TABLE, GraphDispatcher
from ppq.core.data import convert_any_to_torch_tensor
from ppq.core import empty_ppq_cache, ppq_warning
from ppq.api.setting import DispatchingTable, QuantizationSetting, QuantizationSettingFactory
from ppq.quantization.optim import *

from ..core.common import AttributeDict
from ..core.arguments_manage import QuantizeArguments, ArgSpecHelper


LOG = 25
logger = logging.getLogger("HHB")


NEED_TO_BE_QUANTIZED_OPS = {
    "Conv",
    "ConvTranspose",
    "Gemm",
    "Relu",
    "PRelu",
    "Clip",
    "Pad",
    "Resize",
    "MaxPool",
    "AveragePool",
    "GlobalMaxPool",
    "GlobalAveragePool",
    "Softmax",
    "Mul",
    "Add",
    "Max",
    "Sub",
    "Div",
    "Reshape",
    "LeakyRelu",
    "Concat",
    "Sigmoid",
    "Interp",
    "ReduceMean",
    "Transpose",
    "Slice",
    "Flatten",
    "HardSwish",
    "HardSigmoid",
    "MatMul",
}


class PPQFixedPointQuantizer(BaseQuantizer):
    def __init__(
        self,
        graph: BaseGraph,
        per_channel: bool = False,
        sym: bool = False,
        power_of_2: bool = False,
        num_of_bits: int = 8,
        observer_algorithm="percentile",
    ) -> None:
        """A Generalized fixed-point Quantizer."""
        assert 16 >= num_of_bits >= 2, "Unacceptable bit-width."

        self.num_of_bits = num_of_bits
        self.power_of_2 = power_of_2
        self.per_channel = per_channel
        self.symmetric = sym
        self.observer_algorithm = observer_algorithm

        if sym:
            self.quant_min = -pow(2, num_of_bits - 1)
            self.quant_max = pow(2, num_of_bits - 1) - 1
            self.policy = QuantizationPolicy(
                QuantizationProperty.PER_TENSOR
                + QuantizationProperty.LINEAR
                + QuantizationProperty.SYMMETRICAL
            )
        else:
            self.quant_min = 0
            self.quant_max = pow(2, num_of_bits) - 1
            self.policy = QuantizationPolicy(
                QuantizationProperty.PER_TENSOR
                + QuantizationProperty.LINEAR
                + QuantizationProperty.ASYMMETRICAL
            )

        if power_of_2:
            self.policy = QuantizationPolicy(self.policy._policy + QuantizationProperty.POWER_OF_2)

        super().__init__(graph, True)

    def init_quantize_config(self, operation: Operation) -> OperationQuantizationConfig:
        """
        When implementing a custom quantizer, you need to initialize the quantization
        information structure(TQC) for each type of operators.

        Check Predefined Quantizers within ppq.quantization.quantizer folder, see how to implements a
        customized quantizer.

        TQC is made up of input_quantization_config and output_quantization_config.
        The quantization information includes
            quantization policy,
            quantization bit width,
            quantization maximum and minimum values,
            and scale & offset.

        Scale and offset are generated and maintained by the calibration pass.
        """
        OQC = self.create_default_quant_config(
            op=operation,
            num_of_bits=self.num_of_bits,
            quant_min=self.quant_min,
            quant_max=self.quant_max,
            observer_algorithm=self.observer_algorithm,
            policy=self.policy,
            rounding=self.rounding_policy,
            exponent_bits=0,
        )

        if operation.type in {"Conv", "ConvTranspose", "MatMul", "Gemm"}:

            if operation.num_of_input == 3:  # has bias
                # modify quantization of bias
                bias_config = OQC.input_quantization_config[-1]

                bias_config.policy = QuantizationPolicy(
                    QuantizationProperty.SYMMETRICAL
                    + QuantizationProperty.LINEAR
                    + QuantizationProperty.PER_TENSOR
                )
                if operation.type in {"Conv", "ConvTranspose"} and self.per_channel:
                    bias_config.policy = QuantizationPolicy(
                        QuantizationProperty.SYMMETRICAL
                        + QuantizationProperty.LINEAR
                        + QuantizationProperty.PER_CHANNEL
                    )
                    OQC.input_quantization_config[-1].channel_axis = 0
                bias_config.state = QuantizationStates.PASSIVE_INIT
                bias_bits = 30
                bias_config.quant_min = -1 << (bias_bits - 1)
                bias_config.quant_max = 1 << (bias_bits - 1)
                bias_config.num_of_bits = bias_bits
                bias_config.observer_algorithm = "minmax"

            # modify calibration method of parameter(for higher accuracy)
            OQC.input_quantization_config[1].observer_algorithm = "minmax"

            # # for both SYMMETRICAL and ASYMMETRICAL quantization,
            # # weight should always be quantized symmetrically.
            # OQC.input_quantization_config[1].quant_min = - pow(2, self.num_of_bits - 1)
            # OQC.input_quantization_config[1].quant_max = pow(2, self.num_of_bits - 1) - 1
            # OQC.input_quantization_config[1].policy = QuantizationPolicy(
            #     QuantizationProperty.PER_TENSOR +
            #     QuantizationProperty.LINEAR +
            #     QuantizationProperty.SYMMETRICAL +
            #     (QuantizationProperty.POWER_OF_2 if self.power_of_2 else 0))

            if operation.type in {"Conv", "ConvTranspose"} and operation.num_of_parameter > 1:
                # Per-channel Variation
                if self.per_channel:
                    OQC.input_quantization_config[1].policy = QuantizationPolicy(
                        QuantizationProperty.PER_CHANNEL
                        + QuantizationProperty.LINEAR
                        + QuantizationProperty.SYMMETRICAL
                        + (QuantizationProperty.POWER_OF_2 if self.power_of_2 else 0)
                    )
                    OQC.input_quantization_config[1].channel_axis = 0

                    if operation.type == "ConvTranspose":
                        OQC.input_quantization_config[1].channel_axis = 1

        elif operation.type in {"LayerNormalization", "Clip"}:
            # LayerNormalization only take input & output quantization, parameter shall not been quantized.
            for input_config in OQC.input_quantization_config[1:]:
                input_config.state = QuantizationStates.FP32

        return OQC

    @property
    def quant_operation_types(self) -> set:
        QUANTTYPE = NEED_TO_BE_QUANTIZED_OPS
        QUANTTYPE.update(PASSIVE_OPERATIONS)
        return QUANTTYPE

    @property
    def rounding_policy(self) -> RoundingPolicy:
        return RoundingPolicy.ROUND_HALF_EVEN

    @property
    def activation_fusion_types(self) -> set:
        return {
            "Relu",
            "Clip",
            # 'Sigmoid',
            # 'Swish',
            # 'Mish',
            # 'LeakyRelu'
        }

    def build_quant_pipeline(
        self, setting: QuantizationSetting
    ) -> QuantizationOptimizationPipeline:
        assert isinstance(setting, QuantizationSetting), (
            f"PPQ needs a OptimSetting instance to initialize optimization pipeline,"
            f" however {type(setting)} was given."
        )

        if setting.matrix_factorization == True:
            ppq_warning(
                "PPQ Matrix Factorization Pass has been removed from QuantizationSetting since 0.6.5, this pass must be called manually now."
            )
            ppq_warning(
                "PPQ Matrix Factorization Pass 已经不能通过 QuantizationSetting 调用，现在你必须手动调用该优化过程"
            )

        list_of_passes = []
        if setting.ssd_equalization:
            equalization_setting = setting.ssd_setting
            list_of_passes.append(
                SSDEqualizationPass(
                    optimize_level=equalization_setting.opt_level,
                    channel_ratio=equalization_setting.channel_ratio,
                    loss_threshold=equalization_setting.loss_threshold,
                    layer_norm=equalization_setting.layer_norm,
                    iteration=equalization_setting.iteration,
                )
            )

        if setting.fusion:
            fusion_setting = setting.fusion_setting
            list_of_passes.append(
                QuantizeFusionPass(
                    fuse_activation=fusion_setting.fuse_activation,
                    fuse_passive_op=fusion_setting.fuse_passive_op,
                    activation_type=self.activation_fusion_types,
                )
            )

            if fusion_setting.remove_useless_quantization:
                list_of_passes.append(QuantizeSimplifyPass())

        if setting.quantize_parameter:
            param_setting = setting.quantize_parameter_setting
            list_of_passes.append(ParameterQuantizePass(method=param_setting.calib_algorithm))

        if setting.quantize_activation:
            act_setting = setting.quantize_activation_setting
            list_of_passes.append(RuntimeCalibrationPass(method=act_setting.calib_algorithm))

        if setting.fusion:
            if fusion_setting.align_quantization:
                list_of_passes.append(
                    QuantAlignmentPass(
                        elementwise_alignment=fusion_setting.align_elementwise_to,
                        concat_alignment=fusion_setting.align_concat_to,
                        pooling_alignment=fusion_setting.align_avgpooling_to,
                        resize_alignment=fusion_setting.align_resize_to,
                        force_overlap=fusion_setting.force_alignment_overlap,
                    )
                )

        if setting.quantize_parameter:
            param_setting = setting.quantize_parameter_setting
            if param_setting.quantize_passive_parameter:
                list_of_passes.append(PassiveParameterQuantizePass(process_clip=False))

        if setting.bias_correct:
            bias_correct_setting = setting.bias_correct_setting
            list_of_passes.append(
                BiasCorrectionPass(
                    block_size=bias_correct_setting.block_size,
                    interested_layers=bias_correct_setting.interested_layers,
                    steps=bias_correct_setting.steps,
                    collecting_device=bias_correct_setting.collecting_device,
                )
            )

        if setting.lsq_optimization:
            lsq_setting = setting.lsq_optimization_setting
            list_of_passes.append(
                LearnedStepSizePass(
                    interested_layers=lsq_setting.interested_layers,
                    lr=lsq_setting.lr,
                    collecting_device=lsq_setting.collecting_device,
                    steps=lsq_setting.steps,
                    gamma=lsq_setting.gamma,
                    is_scale_trainable=lsq_setting.is_scale_trainable,
                    block_size=lsq_setting.block_size,
                )
            )
            # requant passive parameters
            list_of_passes.append(PassiveParameterQuantizePass(process_clip=False))

        if setting.blockwise_reconstruction:
            blockwise_reconstruction_setting = setting.blockwise_reconstruction_setting
            list_of_passes.append(
                AdaroundPass(
                    interested_layers=blockwise_reconstruction_setting.interested_layers,
                    lr=blockwise_reconstruction_setting.lr,
                    collecting_device=blockwise_reconstruction_setting.collecting_device,
                    steps=blockwise_reconstruction_setting.steps,
                    gamma=blockwise_reconstruction_setting.gamma,
                    is_scale_trainable=blockwise_reconstruction_setting.is_scale_trainable,
                    block_size=blockwise_reconstruction_setting.block_size,
                )
            )
            # requant passive parameters
            list_of_passes.append(PassiveParameterQuantizePass())

        if setting.quantize_parameter:
            if param_setting.baking_parameter:
                list_of_passes.append(ParameterBakingPass())

        if setting.extension:
            list_of_passes.append(ExtensionPass(setting.extension_setting.my_first_parameter))

        return QuantizationOptimizationPipeline(passes=list_of_passes)


def dispatch_hhb_graph(
    graph: BaseGraph,
    quant_types: set,
    dispatcher: Union[str, GraphDispatcher] = "conservative",
    dispatching_table: DispatchingTable = None,
) -> BaseGraph:
    """
    This function will cut your graph into a series of subgraph and send them to different device.
    PPQ provides an automatic dispatcher which, will generate different dispatching scheme on your TargetPlatform.
    A dispatching table can be passed via QuantizationSetting to override
        the default dispatching logic of ppq dispatcher manually.
    """
    dispatching_override = dispatching_table

    if isinstance(dispatcher, str):
        dispatcher = dispatcher.lower()
        if dispatcher not in DISPATCHER_TABLE:
            raise ValueError(
                f'Can not found dispatcher type "{dispatcher}", check your input again.'
            )
        dispatcher = DISPATCHER_TABLE[dispatcher](graph)
    else:
        if not isinstance(dispatcher, GraphDispatcher):
            raise TypeError(
                'Parameter "dispachter" of function ppq.api.dispatch_graph must be String or GraphDispatcher, '
                f"however {type(dispatcher)} was given."
            )
        dispatcher = dispatcher

    assert isinstance(dispatcher, GraphDispatcher)
    dispatching_table = dispatcher.dispatch(
        graph=graph,
        quant_types=quant_types,
        quant_platform=TargetPlatform.UNSPECIFIED,  # MUST BE UNSPECIFIED, 这里的意思是交由 Quantizer 决定是否量化这个算子
        fp32_platform=TargetPlatform.FP32,
        SOI_platform=TargetPlatform.SOI,
    )

    # override dispatching result
    if dispatching_override is not None:
        if not isinstance(dispatching_override, DispatchingTable):
            raise TypeError(
                'Parameter "dispatching_table" of function ppq.api.dispatch_graph must be DispatchingTable, '
                f"however {type(dispatching_override)} was given."
            )

        for opname, platform in dispatching_override.dispatchings.items():
            if opname not in graph.operations:
                continue
            assert isinstance(platform, int), (
                f"Your dispatching table contains a invalid setting of operation {opname}, "
                "All platform setting given in dispatching table is expected given as int, "
                f"however {type(platform)} was given."
            )
            dispatching_table[opname] = TargetPlatform(platform)

    for operation in graph.operations.values():
        assert (
            operation.name in dispatching_table
        ), f"Internal Error, Can not find operation {operation.name} in dispatching table."
        operation.platform = dispatching_table[operation.name]
    return graph


@empty_ppq_cache
def quantize_hhb_model(
    model: BaseGraph,
    calib_dataloader: DataLoader,
    calib_steps: int,
    input_shape: List[int],
    input_dtype: torch.dtype = torch.float,
    inputs: List[Any] = None,
    setting: QuantizationSetting = None,
    collate_fn: Callable = None,
    device: str = "cuda",
    verbose: int = 0,
    do_quantize: bool = True,
    num_of_bits: int = 8,
    sym: bool = False,
    per_channel: bool = False,
    power_of_2: bool = False,
    observer_algorithm: str = "percentile",
) -> BaseGraph:
    """
    quantize ppq model, input ppq graph and return quantized ppq graph.

    Parameters
    ----------
    model : BaseGraph
        Quantized ppq graph

    calib_dataloader : DataLoader
        Calibration data loader

    calib_steps : int
        Calibration steps

    collate_fn : Callable)
        Batch collate func for preprocessing

    input_shape : List[int]
        A list of ints indicating size of input, for multiple inputs, please use
        keyword arg inputs for direct parameter passing and this should be set to None

    input_dtype : torch.dtype
        The torch datatype of input, for multiple inputs, please use keyword arg inputs
        for direct parameter passing and this should be set to None

    inputs : List[Any], optional
        For multiple inputs, please give the specified inputs directly in the form of
        a list of arrays

    setting : OptimSetting
        Quantization setting, default setting will be used when set None

    do_quantize : Bool, optional
        Whether to quantize the model, defaults to True.

    device : str, optional
        Execution device, defaults to 'cuda'.

    verbose : int, optional
        Whether to print details, defaults to 0.

    num_of_bits : int, optional
        The number of bits to quantize tensor.

    sym : bool, optional
        Whether it is symmetric quantization, defaults to False

    per_channel : bool, optional
        Whether it is per-channel quantization, defaults to False

    power_of_2 : bool, optional
        The power of 2 to quantize tensor.

    observer_algorithm : str, optional
        How to get the range of tensor, defaults to 'percentile'

    Raises:
        ValueError: the given platform doesn't support quantization
        KeyError: the given platform is not supported yet

    Returns
    -------
    BaseGraph:
        The quantized IR, containing all information needed for backend execution
    """
    if do_quantize:
        if calib_dataloader is None or calib_steps is None:
            raise TypeError("Quantization needs a valid calib_dataloader and calib_steps setting.")

    if setting is None:
        setting = QuantizationSettingFactory.default_setting()
    # quantizer = PFL.Quantizer(platform=platform, graph=graph)
    quantizer = PPQFixedPointQuantizer(
        model,
        per_channel=per_channel,
        sym=sym,
        power_of_2=power_of_2,
        num_of_bits=num_of_bits,
        observer_algorithm=observer_algorithm,
    )

    ppq_ir = dispatch_hhb_graph(
        graph=model,
        quant_types=quantizer.quant_operation_types,
        dispatcher=setting.dispatcher,
        dispatching_table=setting.dispatching_table,
    )

    if inputs is None:
        dummy_input = torch.zeros(size=input_shape, device=device, dtype=input_dtype)
    else:
        dummy_input = inputs

    quantizer = PPQFixedPointQuantizer(
        ppq_ir,
        per_channel=per_channel,
        sym=sym,
        power_of_2=power_of_2,
        num_of_bits=num_of_bits,
        observer_algorithm=observer_algorithm,
    )
    executor = TorchExecutor(graph=quantizer._graph, device=device)

    if do_quantize:
        quantizer.quantize(
            inputs=dummy_input,
            calib_dataloader=calib_dataloader,
            executor=executor,
            setting=setting,
            calib_steps=calib_steps,
            collate_fn=collate_fn,
        )
        if verbose:
            quantizer.report()
        return quantizer._graph
    else:
        executor = TorchExecutor(graph=ppq_ir, device=device)
        executor.tracing_operation_meta(inputs=dummy_input)
        return quantizer._graph


def create_ppq_quantization_setting(
    config: Union[AttributeDict, QuantizeArguments], target: str
) -> QuantizationSetting:
    ppq_qs = QuantizationSettingFactory.default_setting()
    hhb_config = None
    if isinstance(config, AttributeDict):
        hhb_config = config
    elif isinstance(config, QuantizeArguments):
        hhb_config = AttributeDict()
        for k, v in config.__dict__.items():
            if isinstance(v, ArgSpecHelper):
                hhb_config[k] = v.value
    else:
        raise ValueError(
            "Parameter 'config' must be AttributeDict or QuantizeArguments, "
            f"but get {type(config)}"
        )
    ppq_qs.lsq_optimization = hhb_config.lsq
    ppq_qs.lsq_optimization_setting.steps = hhb_config.lsq_steps
    ppq_qs.lsq_optimization_setting.lr = hhb_config.lsq_lr
    ppq_qs.lsq_optimization_setting.collecting_device = hhb_config.quant_device

    if target == "th1520":
        ppq_qs.fusion_setting.align_avgpooling_to = "Align to Input"

    ppq_qs.fusion_setting.align_elementwise_to = hhb_config.align_elementwise

    return ppq_qs


def convert_dtype_numpy2torch(dtype):
    np2torch = {np.float32: torch.float32, np.int32: torch.int32, np.int64: torch.int64}
    res = torch.float32
    if dtype.type not in np2torch:
        warn_content = f"The dtype of numpy data is {dtype}, don't know how to convert to torch."
        logger.warning(warn_content)
    else:
        res = np2torch[dtype.type]
    return res


def load_calibrate_dataset_for_ppq(hhb_data, batch_size=1):
    """hhb_data: [[np.ndarray, ...], ...]"""
    assert isinstance(hhb_data, (list, tuple)), f"Need list or tuple but get {type(hhb_data)}"
    logger.log(LOG, f"Load {len(hhb_data)} calibrate data")
    samples = []
    for single_data in hhb_data:
        assert isinstance(
            single_data, (list, tuple)
        ), f"single batch data should be list, but get{type(single_data)}"
        sample = []
        for s in single_data:
            sample.append(convert_any_to_torch_tensor(s, dtype=convert_dtype_numpy2torch(s.dtype)))
        samples.append(sample)

    num_in = len(samples[0])
    batches, batch = [], []
    b = list([[] for _ in range(num_in)])
    if batch_size != 1:
        for sample in samples:
            for idx in range(num_in):
                if len(b[idx]) < batch_size:
                    b[idx].append(sample[idx])
                else:
                    batch.append(torch.cat(b[idx], dim=0))
                    b[idx] = [sample[idx]]
                    if idx == num_in - 1:
                        batches.append(batch)
                        batch = []

        for idx in range(num_in):
            if len(b[idx]) != 0:
                batch.append(torch.cat(b[idx], dim=0))
        if len(batch) != 0:
            batches.append(batch)
    else:
        batches = samples

    data_shape = []
    for d in batches[0]:
        data_shape.append(d.size())
    logger.log(LOG, f"Convert dataset into {len(batches)} groups and with shape:")
    logger.log(LOG, f"\t{data_shape}")

    assert len(batches) > 0, "Empty calibration dataset."
    new_batches = []
    for batch in batches:
        if len(batch) == 1:
            new_batches.append(batch[0])
        else:
            new_batches.append(batch)
    return new_batches


def quantize_ppq(
    model_file: str,
    model_input_shape: List[List[int]],
    config: Union[AttributeDict, QuantizeArguments],
    cali_dataset,
    batch_size=1,
    device="cpu",
    output_dir=".",
    target="x86_ref",
):
    if not model_file or not isinstance(model_file, str) or not os.path.exists(model_file):
        raise ValueError(f"Model file is invalid or not exists: {model_file}")
    assert os.path.splitext(model_file)[-1] == ".onnx", "Only support for .onnx file"
    graph = load_onnx_graph(model_file)
    assert graph is not None, "Graph Loading Error, Check your input again."

    if isinstance(model_input_shape, (list, tuple)):
        for i in model_input_shape:
            if not isinstance(i, (list, tuple)):
                raise ValueError(f"Parameters 'input_shape' should be list[list[int]]")
    else:
        raise ValueError(f"Parameters 'input_shape' should be list[list[int]]")

    if len(model_input_shape) == 1:
        input_shape = model_input_shape[0]
        inputs = None
    else:
        inputs = []
        for i in model_input_shape:
            inputs.append(torch.zeros(*i))
        input_shape = None

    if isinstance(config, AttributeDict):
        quant_scheme = config.quantization_scheme
        cali_mode = config.calibrate_mode
    elif isinstance(config, QuantizeArguments):
        quant_scheme = config.quantization_scheme.value
        cali_mode = config.calibrate_mode.value
    else:
        raise ValueError(
            "Parameter 'config' must be AttributeDict or QuantizeArguments, "
            f"but get {type(config)}"
        )
    observer_algorithm = "percentile"
    power_of_2 = False
    if cali_mode == "maxmin":
        observer_algorithm = "minmax"
    elif cali_mode == "pow2":
        power_of_2 = True

    if quant_scheme == "uint8_asym":
        num_of_bits = 8
        sym = False
        per_channel = False
    elif quant_scheme == "int8_sym":
        num_of_bits = 8
        sym = True
        per_channel = False
    elif quant_scheme == "int8_asym_w_sym":
        num_of_bits = 8
        sym = True
        logger.warning(
            "%s quant scheme expect that activation is quantized with int8_asym, "
            "but int8 can only be used with sym in ppq, so we set activation with int8_sym.",
            quant_scheme,
        )
        per_channel = True
    else:
        raise ValueError(f"Current quantization scheme is not supported: {quant_scheme}")

    ppq_qs = create_ppq_quantization_setting(config, target)
    data_loader = load_calibrate_dataset_for_ppq(cali_dataset, batch_size=batch_size)

    quant_info_str = "\n--------- PPQ Quantization info ---------\n"
    quant_info_str += f"Num of bits:              {num_of_bits}\n"
    quant_info_str += f"Sym or Asym:              {'sym' if sym else 'asym'}\n"
    quant_info_str += (
        f"Granularity:              {'per-channel' if per_channel else 'per-tensor'}\n"
    )
    quant_info_str += f"Pow of two:               {power_of_2}\n"
    quant_info_str += f"Observer Algorithm:       {observer_algorithm}\n"
    quant_info_str += f"Num of calibrate dataset: {len(data_loader)}\n"
    quant_info_str += f"Shape of inputs:          {model_input_shape}\n"
    logger.log(LOG, quant_info_str)

    collate_fn = (
        lambda x: list([data.to(device) for data in x])
        if isinstance(x, (list, tuple))
        else x.to(device)
    )
    quantized = quantize_hhb_model(
        model=graph,
        setting=ppq_qs,
        calib_dataloader=data_loader,
        calib_steps=32,
        input_shape=input_shape,
        inputs=inputs,
        collate_fn=collate_fn,
        device=device,
        do_quantize=True,
        num_of_bits=num_of_bits,
        sym=sym,
        per_channel=per_channel,
        power_of_2=power_of_2,
        observer_algorithm=observer_algorithm,
    )

    # analysis quantization errors
    if logger.level <= logging.DEBUG:
        logger.debug(
            "Calculating model network quantization errors(SNR), "
            "SNR of last layer should be less than 0.1 to keep accuracy:"
        )
        reports = graphwise_error_analyse(
            graph=quantized,
            running_device=device,
            steps=32,
            dataloader=data_loader,
            collate_fn=collate_fn,
        )
        for op, snr in reports.items():
            if snr > 0.1:
                logger.debug(
                    f"layer {op}'s cumulative quantization error is significant, "
                    "please optimize it."
                )

        logger.debug(
            "Calculating layer-wise quantization errors(SNR),"
            "SNR of each layer should be less than 0.1 to keep accuracy:"
        )
        layerwise_error_analyse(
            graph=quantized,
            running_device=device,
            interested_outputs=None,
            dataloader=data_loader,
            collate_fn=collate_fn,
        )

    # force to insert qdq ops for all ops
    from ppq import OperationQuantizationConfig, QuantizationVisibility

    for _, op in quantized.operations.items():
        if op.type in NEED_TO_BE_QUANTIZED_OPS:
            if hasattr(op, "config") and op.config:
                OQC = op.config
                assert isinstance(OQC, OperationQuantizationConfig)
                for iqc in OQC.input_quantization_config:
                    iqc.visibility = QuantizationVisibility.FORCE_EXPORT
                for oqc in OQC.output_quantization_config:
                    oqc.visibility = QuantizationVisibility.FORCE_EXPORT

    # export to onnx
    onnx_path = os.path.join(output_dir, "model_ppq_quantized.onnx")
    cfg_path = os.path.join(output_dir, "model_ppq_qunatized_cfg.json")
    export_ppq_graph(
        graph=quantized,
        platform=TargetPlatform.ONNXRUNTIME,
        graph_save_to=onnx_path,
        config_save_to=cfg_path,
        quantized_param=True,
        remove_activation=False,
    )

    return onnx_path, cfg_path
