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
Import networks into relay IR.
"""
import logging
import os
import sys

import tvm

from .core.common import hhb_register_parse
from .core.common import ensure_dir
from .core.common import generate_config_file
from .core.common import ALL_ARGUMENTS_DESC
from .core.common import collect_arguments_info
from .core.arguments_manage import add_import_argument
from .core.arguments_manage import add_optimize_argument
from .core.arguments_manage import add_common_argument
from .core.frontend_manage import import_model
from .core.hhbir_manage import HHBRelayIR


# pylint: disable=invalid-name
logger = logging.getLogger("HHB")


def save_imported_model(mod, params, output_dir="."):
    """Save imported model into file.

    Parameters
    ----------
    mod : tvm.IRModule
        The relay module for compilation
    params : dict of str to tvm.nd.NDArray
        The parameter dict to be used by relay
    output_dir : str
        The output directory holding file.
    """
    mod_path = os.path.join(output_dir, "relay.txt")
    params_path = os.path.join(output_dir, "relay.params")

    with open(mod_path, "w") as f:
        f.write(mod.astext())

    with open(params_path, "wb") as f:
        f.write(tvm.relay.save_param_dict(params))


def hhb_import(
    path, model_format=None, input_name=None, input_shape=None, output_name=None, save_to_dir=None
) -> HHBRelayIR:
    """Import a model from a supported framework into relay ir.

    Parameters
    ----------
    path : str or list[str]
        Path to a model file. There may be two files(.caffemodel, .prototxt) for Caffe model
    model_format : str, optional
        A string representing input model format
    input_name : list[str], optional
        The names of input node in the graph
    input_shape : list[list[int]], optional
        The shape of input node in the graph
    output_name : list[str], optional
        The name of output node in the graph
    save_to_dir : str, optional
        save model into specified directory

    Returns
    -------
    relay_ir : HHBRelayIR
        Relay ir wrapper that holds module and params
    """
    if isinstance(path, str):
        path = [path]
    mod, params = import_model(path, model_format, input_name, input_shape, output_name)

    if save_to_dir is not None:
        save_to_dir = ensure_dir(save_to_dir)
        save_imported_model(mod, params, save_to_dir)

    relay_ir = HHBRelayIR()
    relay_ir.set_model(mod, params)

    return relay_ir


@hhb_register_parse
def add_import_parser(subparsers):
    """Include parser for 'import' subcommand"""

    parser = subparsers.add_parser("import", help="Import a model into relay ir")
    parser.set_defaults(func=driver_import)

    add_import_argument(parser)
    # add_optimize_argument(parser)
    add_common_argument(parser)

    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity")
    parser.add_argument(
        "FILE", nargs="+", help="Path to the input model file, can pass multi files"
    )

    ALL_ARGUMENTS_DESC["import"] = collect_arguments_info(parser._actions)


def driver_import(args_filter):
    """Driver import command"""
    args = args_filter.filtered_args
    mod, params = import_model(
        args.FILE,
        args.model_format,
        args.input_name,
        args.input_shape,
        args.output_name,
    )

    relay_ir = HHBRelayIR()
    relay_ir.set_model(mod, params)
    args.output = ensure_dir(args.output)

    if args.generate_config:
        generate_config_file(os.path.join(args.output, "cmd_import_params.yml"))

    relay_ir.save_model(args.output)
    return 0
