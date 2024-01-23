<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

## HHB Command line tools
### 1. Introduction
HHB is a set of function related command line tools, its usage is similar
to gcc. There are two ways to use it: single-stage mode and multi-stages mode.

### 1.1 Single-stage command

> Usage: python hhb.py [OPTION]...

All arguments are listed as follows:

| Arguments | choices | default | Note |
| ------------|---------|---------| ---- |
| -v, --verbose | None | 0 | Increase verbosity |
| -h, --help | None | False | Show this help information |
| --version | None | None | Print the version and exit |
| -E | None | False | Convert model into relay ir. |
| -Q | None | False | Quantize the relay ir. |
| -C | None | False | codegen the model. |
| --simulate | None | False | Simulate model on x86 device. |
| --no-quantize | None | False | If set, don't quantize the model. |
| -f, --model-file | None | None | Path to the input model file, can pass multi files |
| --save-temps | None | False | Save temp files. |
| --generate-dataset | None | False | Generate dataset according to provided preprocess parameters. |
| -in, --input-name | None | None | Set the name of input node. If '--input-name'is None, default value is 'Placeholder'. Multiple values are separated by semicolon(;). |
| -is, --input-shape | None | None | Set the shape of input nodes. Multiple shapes are separated by semicolon(;) and the dims between shape are separated by space. |
| -on, --output-name | None | None | Set the name of output nodes. Multiple shapes are separated by semicolon(;). |
| --model-format | ['keras', 'onnx', 'pb', 'tflite', 'pytorch', 'caffe'] | None | Specify input model format:['keras', 'onnx', 'pb', 'tflite', 'pytorch', 'caffe'] |
| --reorder-pixel-format | None | False | If original model's input data pixel format is rgb, then covert it to bgr;otherwise, then convert it to rgb. |
| --board | ['th1520', "e907", 'c906', 'c908', 'c920', "c920v2", 'x86_ref', 'unset'] | unset | Set target device, default is unset. |
| -cd, --calibrate-dataset | None | None | Provide with dataset for the input of model in reference step. Support dir or .npz .jpg .png .JPEG or .txt in which there are path of images. Note: only one image path in one line if .txt. |
| --quantization-scheme | ['int4_asym_w_sym', 'uint8_asym', 'int8_asym', 'int8_sym', 'int8_original', 'int8_asym_w_sym', 'int16_sym', 'float16', 'float32', 'unset'] | unset | Scheme of quantization. default is unset, and select scheme by --board. |
| --auto-hybrid-quantization | None | False | If set, quantize model automatically. |
| --quantization-loss-algorithm | ['cos_similarity', 'mse', 'kl_divergence', 'cross_entropy', 'gini'] | cos_similarity | How to calculate accuracy loss for every layer. |
| --quantization-loss-threshold | None | 0.99 | The threshold that will determin thich layer will be quantized with hybrid way.If it is None, we will select threshold automatically. |
| --dump-quantization-loss | None | False | If set, dump quantizaiton loss into file. |
| --hybrid-quantization-scheme | ['int4_asym_w_sym', 'uint8_asym', 'int8_asym', 'int8_sym', 'int8_original', 'int8_asym_w_sym', 'int16_sym', 'float16', 'unset'] | unset | Scheme of hybrid quantization. default is unset. |
| --hybrid-layer-name | None | None | Layer buffer name to use hybrid quantization. |
| --fuse-conv-relu | None | False | Fuse the convolution and relu layer. |
| --fuse-reshape-dense | None | False | Fuse the reshape and dense layer. |
| --broadcast-quantization | None | False | Broadcast quantization parameters for special ops. |
| --fuse-clip | None | False | Fuse clip's attr into pre layer's quantitative information. This flag is only valid when quantization is used. |
| --fuse-zp2bias | None | False | Merge conv2d/dense zp to bias. |
| --target-layout | ['NCHW', 'NHWC'] | NCHW | Set target layout. |
| -m, --data-mean | None | 0 | Set the mean value of input, multiple values are separated by space, default is 0. |
| -s, --data-scale | None | 1 | Scale number(mul) for inputs normalization(data=img*scale), default is 1. |
| -sv, --data-scale-div | None | 1 | Scale number(div) for inputs normalization(data=img/scale), default is 1. |
| -r, --data-resize | None | None | Resize base size for input image to resize. |
| --pixel-format | ['RGB', 'BGR'] | RGB | The pixel format of image data, defalut is RGB |
| --add-preprocess-node | None | False | Add preprocess node for model. |
| --config-file | None | None | Configue more complex parameters for executing the model. |
| --generate-config | None | False | Generate  config file |
| -o, --output | None | hhb_out | The directory that holds the outputs. |
| -sd, --simulate-data | None | None | Provide with dataset for the input of model in reference step. Support dir or .npz .jpg .png .JPEG or .txt in which there are path of images. Note: only one image path in one line if .txt. |
| --postprocess | ['top5', 'save', 'save_and_top5'] | top5 | Set the mode of postprocess: 'top5' show top5 of output; 'save' save output to file;'save_and_top5' show top5 and save output to file. Default is top5 |
| --model-save | ['run_only', 'save_only', 'save_and_run'] | run_only | Whether save binary graph or run only. run_only: execute model only, not save binary graph. save_only: save binary graph only. save_and_run: execute and save model. |
| --model-priority | None | 0 | Set model priority, only for th1520 now. 0 is lowest, 1 is medium, 2 is highest. |
| --without-preprocess | None | False | Do not generate preprocess codes. |
| --input-memory-type | [0, 1, 2] | None | Set the memory type for input tensor, support for multi-values. 0: allocated by CPU and not aligned; 1: allocated by CPU and aligned; 2: dma buffer. |
| --output-memory-type | [0, 1, 2] | None | Set the memory type for output tensor, support for multi-values. 0: allocated by CPU and not aligned; 1: allocated by CPU and aligned; 2: dma buffer. |
| --memory-type | [0, 1, 2] | None | Set the memory type for input and output tensors. 0: allocated by CPU and not aligned; 1: allocated by CPU and aligned; 2: dma buffer. |

### 1.2 Multi-stages command

#### 1.2.1 import subcommand

> Usage: python hhb.py import [OPTION]...

All arguments are listed as follows:

| Arguments | choices | default | Note |
| ------------|---------|---------| ---- |
| -h, --help | None | None | show this help message and exit |
| -in, --input-name | None | None | Set the name of input node. If '--input-name'is None, default value is 'Placeholder'. Multiple values are separated by semicolon(;). |
| -is, --input-shape | None | None | Set the shape of input nodes. Multiple shapes are separated by semicolon(;) and the dims between shape are separated by space. |
| -on, --output-name | None | None | Set the name of output nodes. Multiple shapes are separated by semicolon(;). |
| --model-format | ['keras', 'onnx', 'pb', 'tflite', 'pytorch', 'caffe'] | None | Specify input model format:['keras', 'onnx', 'pb', 'tflite', 'pytorch', 'caffe'] |
| --reorder-pixel-format | None | False | If original model's input data pixel format is rgb, then covert it to bgr;otherwise, then convert it to rgb. |
| --config-file | None | None | Configue more complex parameters for executing the model. |
| --generate-config | None | False | Generate  config file |
| -o, --output | None | hhb_out | The directory that holds the outputs. |
| -v, --verbose | None | 0 | Increase verbosity |
| FILE | None | None | Path to the input model file, can pass multi files |

#### 1.2.2 quantize subcommand

> Usage: python hhb.py quantize [OPTION]...

All arguments are listed as follows:

| Arguments | choices | default | Note |
| ------------|---------|---------| ---- |
| -h, --help | None | None | show this help message and exit |
| -m, --data-mean | None | 0 | Set the mean value of input, multiple values are separated by space, default is 0. |
| -s, --data-scale | None | 1 | Scale number(mul) for inputs normalization(data=img*scale), default is 1. |
| -sv, --data-scale-div | None | 1 | Scale number(div) for inputs normalization(data=img/scale), default is 1. |
| -r, --data-resize | None | None | Resize base size for input image to resize. |
| --pixel-format | ['RGB', 'BGR'] | RGB | The pixel format of image data, defalut is RGB |
| --add-preprocess-node | None | False | Add preprocess node for model. |
| -cd, --calibrate-dataset | None | None | Provide with dataset for the input of model in reference step. Support dir or .npz .jpg .png .JPEG or .txt in which there are path of images. Note: only one image path in one line if .txt. |
| --quantization-scheme | ['int4_asym_w_sym', 'uint8_asym', 'int8_asym', 'int8_sym', 'int8_original', 'int8_asym_w_sym', 'int16_sym', 'float16', 'float32', 'unset'] | unset | Scheme of quantization. default is unset, and select scheme by --board. |
| --auto-hybrid-quantization | None | False | If set, quantize model automatically. |
| --quantization-loss-algorithm | ['cos_similarity', 'mse', 'kl_divergence', 'cross_entropy', 'gini'] | cos_similarity | How to calculate accuracy loss for every layer. |
| --quantization-loss-threshold | None | 0.99 | The threshold that will determin thich layer will be quantized with hybrid way.If it is None, we will select threshold automatically. |
| --dump-quantization-loss | None | False | If set, dump quantizaiton loss into file. |
| --hybrid-quantization-scheme | ['int4_asym_w_sym', 'uint8_asym', 'int8_asym', 'int8_sym', 'int8_original', 'int8_asym_w_sym', 'int16_sym', 'float16', 'unset'] | unset | Scheme of hybrid quantization. default is unset. |
| --hybrid-layer-name | None | None | Layer buffer name to use hybrid quantization. |
| --fuse-conv-relu | None | False | Fuse the convolution and relu layer. |
| --fuse-reshape-dense | None | False | Fuse the reshape and dense layer. |
| --broadcast-quantization | None | False | Broadcast quantization parameters for special ops. |
| --fuse-clip | None | False | Fuse clip's attr into pre layer's quantitative information. This flag is only valid when quantization is used. |
| --fuse-zp2bias | None | False | Merge conv2d/dense zp to bias. |
| --target-layout | ['NCHW', 'NHWC'] | NCHW | Set target layout. |
| --board | ['th1520', "e907", 'c906', 'c908', 'c920', "c920v2", 'x86_ref', 'unset'] | unset | Set target device, default is unset. |
| --model-save | ['run_only', 'save_only', 'save_and_run'] | run_only | Whether save binary graph or run only. run_only: execute model only, not save binary graph. save_only: save binary graph only. save_and_run: execute and save model. |
| --model-priority | None | 0 | Set model priority, only for th1520 now. 0 is lowest, 1 is medium, 2 is highest. |
| --without-preprocess | None | False | Do not generate preprocess codes. |
| --input-memory-type | [0, 1, 2] | None | Set the memory type for input tensor, support for multi-values. 0: allocated by CPU and not aligned; 1: allocated by CPU and aligned; 2: dma buffer. |
| --output-memory-type | [0, 1, 2] | None | Set the memory type for output tensor, support for multi-values. 0: allocated by CPU and not aligned; 1: allocated by CPU and aligned; 2: dma buffer. |
| --memory-type | [0, 1, 2] | None | Set the memory type for input and output tensors. 0: allocated by CPU and not aligned; 1: allocated by CPU and aligned; 2: dma buffer. |
| --config-file | None | None | Configue more complex parameters for executing the model. |
| --generate-config | None | False | Generate  config file |
| -o, --output | None | hhb_out | The directory that holds the outputs. |
| -v, --verbose | None | 0 | Increase verbosity |
| FILE | None | None | Directory to the model file |

#### 1.2.3 codegen subcommand

> Usage: python hhb.py codegen [OPTION]...

All arguments are listed as follows:

| Arguments | choices | default | Note |
| ------------|---------|---------| ---- |
| -h, --help | None | None | show this help message and exit |
| --board | ['th1520', "e907", 'c906', 'c908', 'c920', "c920v2", 'x86_ref', 'unset'] | unset | Set target device, default is unset. |
| --postprocess | ['top5', 'save', 'save_and_top5'] | top5 | Set the mode of postprocess: 'top5' show top5 of output; 'save' save output to file;'save_and_top5' show top5 and save output to file. Default is top5 |
| --model-save | ['run_only', 'save_only', 'save_and_run'] | run_only | Whether save binary graph or run only. run_only: execute model only, not save binary graph. save_only: save binary graph only. save_and_run: execute and save model. |
| --model-priority | None | 0 | Set model priority, only for th1520 now. 0 is lowest, 1 is medium, 2 is highest. |
| --without-preprocess | None | False | Do not generate preprocess codes. |
| --input-memory-type | [0, 1, 2] | None | Set the memory type for input tensor, support for multi-values. 0: allocated by CPU and not aligned; 1: allocated by CPU and aligned; 2: dma buffer. |
| --output-memory-type | [0, 1, 2] | None | Set the memory type for output tensor, support for multi-values. 0: allocated by CPU and not aligned; 1: allocated by CPU and aligned; 2: dma buffer. |
| --memory-type | [0, 1, 2] | None | Set the memory type for input and output tensors. 0: allocated by CPU and not aligned; 1: allocated by CPU and aligned; 2: dma buffer. |
| --config-file | None | None | Configue more complex parameters for executing the model. |
| --generate-config | None | False | Generate  config file |
| -o, --output | None | hhb_out | The directory that holds the outputs. |
| -v, --verbose | None | 0 | Increase verbosity |
| FILE | None | None | Directory to the model file |

#### 1.2.4 simulate subcommand

> Usage: python hhb.py simulate [OPTION]...

All arguments are listed as follows:

| Arguments | choices | default | Note |
| ------------|---------|---------| ---- |
| -h, --help | None | None | show this help message and exit |
| -sd, --simulate-data | None | None | Provide with dataset for the input of model in reference step. Support dir or .npz .jpg .png .JPEG or .txt in which there are path of images. Note: only one image path in one line if .txt. |
| -m, --data-mean | None | 0 | Set the mean value of input, multiple values are separated by space, default is 0. |
| -s, --data-scale | None | 1 | Scale number(mul) for inputs normalization(data=img*scale), default is 1. |
| -sv, --data-scale-div | None | 1 | Scale number(div) for inputs normalization(data=img/scale), default is 1. |
| -r, --data-resize | None | None | Resize base size for input image to resize. |
| --pixel-format | ['RGB', 'BGR'] | RGB | The pixel format of image data, defalut is RGB |
| --add-preprocess-node | None | False | Add preprocess node for model. |
| --postprocess | ['top5', 'save', 'save_and_top5'] | top5 | Set the mode of postprocess: 'top5' show top5 of output; 'save' save output to file;'save_and_top5' show top5 and save output to file. Default is top5 |
| --config-file | None | None | Configue more complex parameters for executing the model. |
| --generate-config | None | False | Generate  config file |
| -o, --output | None | hhb_out | The directory that holds the outputs. |
| -v, --verbose | None | 0 | Increase verbosity |
| FILE | None | None | Directory to the model file |

#### 1.2.5 profiler subcommand

> Usage: python hhb.py profiler [OPTION]...

All arguments are listed as follows:

| Arguments | choices | default | Note |
| ------------|---------|---------| ---- |
| -h, --help | None | None | show this help message and exit |
| -in, --input-name | None | None | Set the name of input node. If '--input-name'is None, default value is 'Placeholder'. Multiple values are separated by semicolon(;). |
| -is, --input-shape | None | None | Set the shape of input nodes. Multiple shapes are separated by semicolon(;) and the dims between shape are separated by space. |
| -on, --output-name | None | None | Set the name of output nodes. Multiple shapes are separated by semicolon(;). |
| --model-format | ['keras', 'onnx', 'pb', 'tflite', 'pytorch', 'caffe'] | None | Specify input model format:['keras', 'onnx', 'pb', 'tflite', 'pytorch', 'caffe'] |
| --reorder-pixel-format | None | False | If original model's input data pixel format is rgb, then covert it to bgr;otherwise, then convert it to rgb. |
| --ir-type | ['relay', 'th1520'] | relay | The ir type that will be profiled, default is relay |
| --indicator | ['cal', 'mem', 'cycle', 'all'] | cal | Select indicator to profile, default is cal(calculation). cal: calculation, how many operations to be executed for current op. mem: memory, how many memory to be used for current op. cycle: how many cycles to execute op. all: include all indicators above. |
| --output-type | ['json', 'binary', 'print', 'total', 'all'] | total | How to show results, default is show summary result. |
| --npu-frequency | None | 1000000000 | NPU frequency(HZ). |
| --config-file | None | None | Configue more complex parameters for executing the model. |
| --generate-config | None | False | Generate  config file |
| -o, --output | None | hhb_out | The directory that holds the outputs. |
| -v, --verbose | None | 0 | Increase verbosity |
| -f, --model-file | None | None | Path to the input model file, can pass multi files |

### 2. How to use

HHB support two kind of modes: single-stage mode and multi-stages mode. And both modes can parser all command parameters from specify file.

#### 2.1 single-stage mode

For example, you can import, quantize and simulate the specified model by a single command as follows:

```Python
python hhb.py --simulate \
	-v -v -v \
	--data-mean "103.94,116.98,123.68" \
	--data-scale 0.017 \
	--data-resize 256 \
	--calibrate-dataset quant.txt \
	--simulate-data n01440764_188.JPEG \
	--opt-level 3 \
	--board x86_ref \
	--model-file mobilenetv1.prototxt \
	mobilenetv1.caffemodel \
	--postprocess top5 \
```



#### 2.2 multi-stages mode

In this mode, you can compile model by executing multiply sub command.

##### 2.2.1 import model

```Python
python hhb.py import alexnet.prototxt alexnet.caffemodel -o model.relay --opt-level -1
```

##### 2.2.2 quantize model

```Python
python hhb.py quantize \
	--data-mean "103.94,116.98,123.68" \
	--data-scale 1 \
	--data-resize 256 \
	--calibrate-dataset quant.txt \
    -o model_qnn \
	model.relay
```

##### 2.2.3 codegen

```Python
python hhb.py codegen \
	--board x86_ref \
	-o quant_codegen \
	model_qnn \
```

##### 2.2.4 simulate

```Python
python hhb.py simulate \
	--simulate-data /lhome/fern/aone/hhb/tests/thead/images/n01440764_188.JPEG \
	--data-mean "103.94,116.98,123.68" \
	--data-scale 1 \
	--data-resize 256 \
	--postprocess top5 \
	-o output \
	quant_codegen \
```


#### 2.3 using config file

You can generate a template config file by:

```bash
python hhb.py --generate-config -o config.yaml
```

change the part parameters...

The use the config file by:

```Bash
python hhb.py --config-file config.yaml --file mobilenetv1.prototxt mobilenetv1.caffemodel
```

#### 2.4 generate preprocessed dataset

You can generate dataset by provided preprocess parameters

```bash
python hhb.py --generate-dataset \
	-v -v -v \
	-sd /lhome/fern/aone/hhb/tests/thead/images \
	--input-shape "1 3 224 224" \
	-o gen_dataset_o \
	--data-resize 256 \
	--data-mean "103.94 116.78 123.68" \
	--data-scale 0.017 \
```

#### 2.5 benchmark test

You can do accuracy testing by 'benchmark' subcommand. Currently, we support classfication model only.

```bash
python hhb.py benchmark \
	-v -v -v \
	--board x86_ref \
	-cd quant.txt \
	-sd ILSVRC2012_img_val \
	--reference-label val.txt \
	--print-interval 50 \
	--save-temps \
	--data-scale 1 \
	--data-mean "103.94 116.98 123.68" \
	--data-resize 256 \
	-o alexnet_caffe_benchmark_o \
	--no-quantize \
	alexnet.prototxt \
	alexnet.caffemodel \
```

#### 2.6 profiler

You can profile model by 'profiler' subcommand.

```bash
python hhb.py profiler \
	-v -v -v \
	--ir-type relay \
	--indicator cal \
	--output-type binary json \
	--model-file alexnet.caffemodel alexnet.prototxt \
```
