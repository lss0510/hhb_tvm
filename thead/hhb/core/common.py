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
"""Some common utils"""
import os
import importlib
import functools
import argparse
import re
import yaml
import logging
import collections
import sys
import subprocess
import json

import numpy as np


ArgInfo = collections.namedtuple("_ArgInfo", ["name", "choices", "default_value", "help"])

HHB_REGISTERED_PARSER = []
ALL_ARGUMENTS_INFO = {}
HHB_IR = []
ALL_ARGUMENTS_DESC = {}
ARGS_DEST_TO_OPTIONS_STRING = {}
logger = logging.getLogger("HHB")


def hhb_version():
    """Version information"""
    __version__ = "2.9.5"
    __build_time__ = "20231226"
    return "HHB version: " + __version__ + ", build " + __build_time__


def hhb_exit(message):
    logger.error(message)
    sys.exit()


def parse_mean(mean):
    """Parse the mean value .

    Parameters
    ----------
    mean : str or list
        The provided mean value

    Returns
    -------
    mean_list : list[int]
        The mean list
    """
    if isinstance(mean, list):
        return mean
    if "," in mean:
        mean = mean.replace(",", " ")
    mean_list = mean.strip().split(" ")
    mean_list = list([float(n) for n in mean_list if n])
    # if len(mean_list) == 1:
    #     mean_list = mean_list * 3
    return mean_list


class HHBArgumentParser(argparse.ArgumentParser):
    """HHB ArgumentParser that avoids to allow_abbrev=False invalid for short options
    since Python3.8
    """

    def _get_option_tuples(self, option_string):
        res = []
        if self.allow_abbrev:
            res = super()._get_option_tuples(option_string)
        return res


class HHBException(Exception):
    """HHB Exception"""


class AttributeDict(dict):
    def __init__(self, **kwargs):
        super(AttributeDict, self).__init__()
        for name in kwargs:
            setattr(self, name, kwargs[name])

    def __getattr__(self, name):
        try:
            return self[name]
        except:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value


def hhb_register_parse(make_subparser):
    """
    Utility function to register a subparser for HHB.

    Functions decorated with `hhb_register_parse` will be invoked
    with a parameter containing the subparser instance they need to add itself to,
    as a parser.

    Example
    -------

        @hhb_register_parse
        def _example_parser(main_subparser):
            subparser = main_subparser.add_parser('example', help='...')
            ...

    """
    HHB_REGISTERED_PARSER.append(make_subparser)
    return make_subparser


def hhb_ir_helper(hhb_ir):
    """
    Utility function to register a HHB_IR.

    Classes decorated with `hhb_register_parse` will be put into HHB_IR

    Example
    -------

        @hhb_ir_helper
        class HHBIR_E(HHBIRBase):
            pass

    """
    HHB_IR.append(hhb_ir)
    return hhb_ir


def get_parameters_info(params_name="unknown"):
    def decorate(func):
        @functools.wraps(func)
        def inner_wrapper(parser):
            if not isinstance(parser, argparse.ArgumentParser):
                raise HHBException("invalid parser:{}".format(parser))
            if parser.prog == "hhb_api":
                before_args = vars(parser.parse_known_args([])[0])
            else:
                before_args = vars(parser.parse_known_args()[0])
            value = func(parser)
            if parser.prog == "hhb_api":
                after_args = vars(parser.parse_known_args([])[0])
            else:
                after_args = vars(parser.parse_known_args()[0])
            ALL_ARGUMENTS_INFO[params_name] = {
                key: value for key, value in after_args.items() if key not in before_args
            }
            return value

        return inner_wrapper

    return decorate


def argument_filter_helper(func):
    @functools.wraps(func)
    def inner_wrapper(filtered_args, extra=None):
        if not isinstance(filtered_args, AttributeDict):
            raise HHBException("invalid filtered_args:{}".format(filtered_args))
        if extra is not None and not isinstance(extra, AttributeDict):
            raise HHBException("invalid extra: {}".format(extra))
        value = func(filtered_args, extra)
        return value

    return inner_wrapper


def import_module_for_register(all_modules):
    """Dynamic importing libraries"""
    if not all_modules:
        return
    for m in all_modules:
        importlib.import_module(m)


def ensure_dir(directory):
    """Create a directory if not exists

    Parameters
    ----------

    directory : str
        File path to create
    """
    if directory is None:
        directory = "hhb_out"
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def ensure_compiler(command):
    """Call compiler command, and return version

    Parameters
    ----------

    command : str
        command to call
    """
    cmd_line = [command, "-v"]
    cmd_output = subprocess.check_output(cmd_line, stderr=subprocess.STDOUT)
    output_str = cmd_output.decode()
    if "command not found" in output_str:
        raise HHBException("Cannot find compiler.\n{}".format(output_str))
    # only for gcc
    version_index = output_str.find("gcc Toolchain") + 15
    version_str = output_str[version_index : version_index + 3]
    if version_str:
        return float(version_str)
    return 0


def get_target(board):
    """Get the target info accorrding to the board type."""
    if board == "th1520":
        target = "c -device=th1520"
    elif board == "hth1520":
        target = "c -device=hth1520"
    elif board == "e907":
        target = "c -device=e907"
    elif board == "c906":
        target = "c -device=c906"
    elif board == "rvm":
        target = "c -device=rvm"
    elif board == "c908":
        target = "c -device=c908"
    elif board == "c920":
        target = "c -device=c920"
    elif board == "c920v2":
        target = "c -device=c920v2"
    elif board == "x86_ref":
        target = "c"
    return target


def print_top5(value, output_name=None, shape=None):
    """Print the top5 infomation

    Parameters
    ----------
    value : numpy.ndarray
        The data need to be print.
    output_name : str
        The name of value.
    shape : List[int]
        The original shape of value.
    """
    if not isinstance(value, np.ndarray):
        raise HHBException("Unsupport for {}, please input ndarray".format(type(value)))
    len_t = np.prod(value.size)
    pre = np.reshape(value, [len_t])
    ind = np.argsort(pre)
    if len_t > 5:
        ind = ind[len_t - 5 :]
    value = pre[ind]
    ind = ind[::-1]
    value = value[::-1]
    print("=== tensor info ===")
    print(f"shape: {shape}")
    print(f"The max_value of output: {pre.max():.6f}")
    print(f"The min_value of output: {pre.min():.6f}")
    print(f"The mean_value of output: {pre.mean():.6f}")
    print(f"The var_value of output: {pre.var():.6f}")
    print(f"====== index:{output_name}, shape:{shape}, top5: ======")
    for (i, v) in zip(ind, value):
        print("{}:{}".format(i, v))


def convert_invalid_symbol(input_str):
    """Convert invalid symbols in string into '_'

    Parameters
    ----------
    input_str : str
        Input string

    Returns
    -------
    new_str : str
        Modified string.

    """
    invalid_symbol = r"[/:\s\.]"
    new_str = re.sub(invalid_symbol, "_", input_str)
    return new_str


def find_index(data, item):
    """Find item index in data

    Parameters
    ----------
    data: list or tuple
        The data that will be searched

    item: object
        To be searched item

    Returns
    -------
    res: int
        The index that item in data. If item is not in data, return -1.

    """
    if not isinstance(data, (list, tuple)):
        raise HHBException("Unsupport for type: {}".format(type(data)))

    res = -1
    if item in data:
        res = data.index(item)
    return res


def generate_config_file(config_file):
    logger.debug("save the cmd parameters info into: %s", config_file)
    with open(config_file, "w") as f:
        yaml.safe_dump(ALL_ARGUMENTS_INFO, f, default_flow_style=False)


def collect_arguments_info(actions):
    """Convert argparse.Actions into ArgInfo."""
    res = list()
    for a in actions:
        if a.dest not in ARGS_DEST_TO_OPTIONS_STRING:
            ARGS_DEST_TO_OPTIONS_STRING[a.dest] = a.option_strings
        # ignore hidden arguments.
        if a.help == "==SUPPRESS==":
            continue
        default_value = a.default
        if default_value == "==SUPPRESS==":
            default_value = None

        if not a.option_strings:
            name = [a.dest]
        else:
            name = a.option_strings

        tmp = ArgInfo(
            name=", ".join(name), choices=a.choices, default_value=default_value, help=a.help
        )
        res.append(tmp)
    return res


def generate_readme_file(output_dir="."):
    """Automatically README.md file for command line tools."""

    def _gen_table(key):
        res = list()
        res.append("\n| Arguments | choices | default | Note |\n")
        res.append("| ------------|---------|---------| ---- |\n")
        if key in ALL_ARGUMENTS_DESC:
            for arg in ALL_ARGUMENTS_DESC[key]:
                new_help = arg.help.replace("\n", " ")
                res.append(f"| {arg.name} | {arg.choices} | {arg.default_value} | {new_help} |\n")
        return res

    content = list(
        """<!--- Licensed to the Apache Software Foundation (ASF) under one -->
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
"""
    )

    # generate main command arguments.
    content.extend(_gen_table("main_command"))

    # generate import subcommand arguments.
    content.append(
        """
### 1.2 Multi-stages command

#### 1.2.1 import subcommand

> Usage: python hhb.py import [OPTION]...

All arguments are listed as follows:
"""
    )
    content.extend(_gen_table("import"))

    # generate quantize subcommand arguments.
    content.append(
        """
#### 1.2.2 quantize subcommand

> Usage: python hhb.py quantize [OPTION]...

All arguments are listed as follows:
"""
    )
    content.extend(_gen_table("quantize"))

    # generate codegen subcommand arguments.
    content.append(
        """
#### 1.2.3 codegen subcommand

> Usage: python hhb.py codegen [OPTION]...

All arguments are listed as follows:
"""
    )
    content.extend(_gen_table("codegen"))

    # generate simulate subcommand arguments.
    content.append(
        """
#### 1.2.4 simulate subcommand

> Usage: python hhb.py simulate [OPTION]...

All arguments are listed as follows:
"""
    )
    content.extend(_gen_table("simulate"))

    # generate profiler subcommand arguments.
    content.append(
        """
#### 1.2.5 profiler subcommand

> Usage: python hhb.py profiler [OPTION]...

All arguments are listed as follows:
"""
    )
    content.extend(_gen_table("profiler"))

    # generate examples
    content.append(
        r"""
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
"""
    )

    if not os.path.exists(output_dir):
        logger.error("%s dose not exist." % output_dir)
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.writelines(content)


def hhb_deprecated_check(deprecated_content, since_version=None, substitute=""):
    """check deprecated args."""
    msg = f"{deprecated_content} will be deprecated"
    if since_version:
        msg += f" since {since_version}"
    if substitute:
        msg += f", recommend using {substitute} instead"
    msg += "."
    logger.warning(msg)


def to_json(data, path, with_format=False):
    """save json file with beautifier format."""
    if with_format:
        import jsbeautifier

        options = jsbeautifier.default_options()
        options.indent_size = 2
        res = jsbeautifier.beautify(json.dumps(data, sort_keys=True), options)
        with open(path, "w") as f:
            f.write(res)
    else:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


def from_json(path: str):
    with open(path, "r") as f:
        data = json.load(f)
    return data
