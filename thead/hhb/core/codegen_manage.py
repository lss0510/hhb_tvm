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
"""Manage Codegen"""
import logging
import struct
import sys
import os
import shutil

from tvm import relay
from math import ceil

from .common import argument_filter_helper, hhb_exit
from .common import ALL_ARGUMENTS_INFO
from .common import AttributeDict
from .common import HHBException
from .common import parse_mean
from .hhbir_manage import HHBBoardQnnCodegenIR
from tvm.relay.backend.contrib.csinn_backend import emit_binary_model


logger = logging.getLogger("HHB")


@argument_filter_helper
def collect_codegen_config(filtered_args, extra=None):
    """collect codegen arguments"""
    filtered_args.codegen_config = AttributeDict()
    for k in ALL_ARGUMENTS_INFO["codegen"]:
        filtered_args.codegen_config[k] = filtered_args[k]

    filtered_args.codegen_config.th1520_input_fix_size = parse_mean(
        filtered_args.codegen_config.th1520_input_fix_size
    )
    if filtered_args.codegen_config.model_save == "save_only":
        raise HHBException("Unsupport --model-save = save_only.\n")


@argument_filter_helper
def set_codegen_config(filtered_args, extra=None):
    """set codegen arguments"""

    def _set_memory_type(io_memory_type, io_num, unify_type=None):
        res = io_memory_type
        if io_memory_type is None:
            if unify_type is None:
                res = [0] * io_num
            else:
                res = [unify_type] * io_num
        else:
            if len(io_memory_type) == 1:
                res = io_memory_type * io_num
            else:
                if len(io_memory_type) != io_num:
                    hhb_exit(
                        "There are {} input/output, but get {} input/output memory".format(
                            io_num, len(io_memory_type)
                        )
                    )
        return res

    if not hasattr(filtered_args, "codegen_config"):
        raise HHBException("Please execute 'collect_codegen_config' filter first.")
    if not hasattr(extra, "input_num"):
        raise HHBException("extra has no input_num attr")
    if not hasattr(extra, "output_num"):
        raise HHBException("extra has no output_num attr")

    filtered_args.codegen_config.input_memory_type = _set_memory_type(
        filtered_args.codegen_config.input_memory_type,
        extra.input_num,
        filtered_args.codegen_config.memory_type,
    )

    filtered_args.codegen_config.output_memory_type = _set_memory_type(
        filtered_args.codegen_config.output_memory_type,
        extra.output_num,
        filtered_args.codegen_config.memory_type,
    )


def get_execute_path():
    if hasattr(sys, "_MEIPASS"):
        execute_path = os.path.dirname(os.path.realpath(sys.executable))
    else:
        execute_path, _ = os.path.split(os.path.abspath(__file__))
        execute_path = os.path.join(execute_path, "..")
    return execute_path


def entry_c_codegen(
    codegen_obj: HHBBoardQnnCodegenIR,
    input_shape,
    output_shape,
    board,
    output_path,
    main_file,
    postprocess="top5",
    model_save="run_only",
    without_preprocess=False,
    preprocess_params=None,
    multithread=False,
    input_memory_type=None,
    q_scheme=None,
    dynamic_shape=None,
    hhb_gen=False,
):
    """Generate the main.c file"""

    execute_path = get_execute_path()

    with open(main_file, "r") as f:
        code_str = f.read()

    template_dir = os.path.join(execute_path, "config", "template")

    # check options setting
    if preprocess_params.calibrate_data_format == "npz":
        without_preprocess = True

    if board != "anole" and board != "th1520":
        # disable_nbg = True
        model_save = "run_only"

    #######################################################################
    #
    # Header Codegen
    #
    with open(os.path.join(template_dir, "header.tp"), "r") as f:
        header_str = f.read()
    if board == "anole":
        header_str += '\n#include "shl_ovx.h"'
    else:
        header_str += '\n#include "shl_ref.h"'

    if not without_preprocess:
        header_str += '\n#include "process.h"'
        process_c_path = os.path.join(execute_path, "config", "process", "src", "process.c")
        process_c = os.path.join(output_path, codegen_obj.preprocess_source_name)
        process_h_path = os.path.join(execute_path, "config", "process", "include", "process.h")
        process_h = os.path.join(output_path, codegen_obj.preprocess_header_name)
        logger.info("write process header to %s", process_h)
        logger.info("write process source to %s", process_c)
        shutil.copy(process_h_path, process_h)
        shutil.copy(process_c_path, process_c)
    io_c_path = os.path.join(execute_path, "config", "process", "src", "io.c")
    io_c = os.path.join(output_path, codegen_obj.preio_source_name)
    io_h_path = os.path.join(execute_path, "config", "process", "include", "io.h")
    io_h = os.path.join(output_path, codegen_obj.preio_header_name)
    logger.info("write io header to %s", io_h)
    logger.info("write io source to %s", io_c)
    shutil.copy(io_h_path, io_h)
    shutil.copy(io_c_path, io_c)

    code_str = code_str.replace("#_hhb_header_files_#", header_str)

    #######################################################################
    #
    # Macro Codegen
    #
    with open(os.path.join(template_dir, "macro_def.tp"), "r") as f:
        macro_str = f.read()
    code_str = code_str.replace("#_hhb_macro_def_#", macro_str)

    #######################################################################
    #
    # Function Declaration Codegen
    #
    with open(os.path.join(template_dir, "function_decl.tp"), "r") as f:
        function_str = f.read()
    # if disable_nbg == False:
    if model_save != "run_only":
        if multithread and board == "anole":
            function_str += "\nvoid *csinn_nbg(const char *nbg_file_name, int deviceIndex);"
        else:
            function_str += "\nvoid *csinn_nbg(const char *nbg_file_name);"
    else:
        function_str += "\n#define csinn_nbg(...) NULL"
    csinn_args = ""
    for i in range(len(input_shape)):
        csinn_args += "void *data" + str(i) + ", "
    function_str = function_str.replace("#_csinn_args#", csinn_args)
    if multithread and board == "anole":
        function_str = function_str.replace(
            "void *csinn_(char *params);", "void *csinn_(char *params, int deviceIndex);"
        )
    if board == "th1520" and model_save == "save_only":
        function_str = function_str.replace(
            "void *csinn_(char *params);", "#define csinn_(...) NULL"
        )
    code_str = code_str.replace("#_hhb_function_decl_#", function_str)

    #######################################################################
    #
    # Global Variable Codegen
    #
    with open(os.path.join(template_dir, "global_var_decl.tp"), "r") as f:
        global_var_str = f.read()

    def _convert_shape2str(shape_list):
        res = ""
        for shape in shape_list:
            shape = shape if len(shape) != 0 else [1]
            tmp_str = list(map(str, shape))
            tmp_str = " * ".join(tmp_str)
            if q_scheme == "int16_sym":
                tmp_str += " * 2"
            res += tmp_str + ", "
        return res

    global_var_str = global_var_str.replace("#_input_size_define#", _convert_shape2str(input_shape))
    global_var_str = global_var_str.replace("#_model_name_define#", "network")
    code_str = code_str.replace("#_hhb_global_var_decl_#", global_var_str)

    #######################################################################
    #
    # Preprocess Codegen
    #
    preprocess_str = ""
    if not without_preprocess:
        with open(os.path.join(template_dir, "preprocess_def.tp"), "r") as f:
            preprocess_str = f.read()
        preprocess_str = _preprocess_macro_define(preprocess_params, preprocess_str)
    code_str = code_str.replace("#_hhb_preprocess_def_#", preprocess_str)

    #######################################################################
    #
    # Utils Codegen
    #
    with open(os.path.join(template_dir, "utils_def.tp"), "r") as f:
        utils_str = f.read()
    code_str = code_str.replace("#_hhb_utils_def_#", utils_str)

    #######################################################################
    #
    # Postprocess Codegen
    #
    with open(os.path.join(template_dir, "postprocess_def.tp"), "r") as f:
        postprocess_str = f.read()

    convert_fouput = ""
    if board != "anole":
        convert_fouput = "struct csinn_tensor *foutput = shl_ref_tensor_transform_f32(output);"

    postprocess_str = postprocess_str.replace("#_convert_fouput_#", convert_fouput)

    show_top5 = ""
    if "top5" in postprocess:
        if board == "anole":
            show_top5 = "shl_ovx_show_top5(i, sess);"
        else:
            show_top5 = "shl_show_top5(foutput, sess);"
    postprocess_str = postprocess_str.replace("#_show_top5_stats_#", show_top5)

    free_anole_input_data = ""
    free_output_data = ""
    if board == "anole":
        free_anole_input_data = "free(input->data);"
        free_output_data = "free(output->data);"
    if board in ("th1520", "hth1520", "c906", "rvm", "c908", "c920"):
        free_output_data = "shl_ref_tensor_transform_free_f32(foutput);\n"
        if board in ("c906", "rvm", "c908", "c920"):
            free_output_data += " " * 8 + "if (!output->is_const) {\n"
            free_output_data += " " * 12 + "shl_mem_free(output->data);\n"
            free_output_data += " " * 8 + "}"
    postprocess_str = postprocess_str.replace("#_free_anole_input_data_#", free_anole_input_data)
    postprocess_str = postprocess_str.replace("#_free_output_data_#", free_output_data)

    save_output = ""
    if "save" in postprocess:
        save_output = "char filename[FILE_LENGTH] = {0};\n"
        save_output += " " * 8 + "char shape[SHAPE_LENGHT] = {0};\n"
        save_output += (
            " " * 8 + "shape2string(output->dim, output->dim_count, shape, SHAPE_LENGHT);\n"
        )
        save_output += (
            " " * 8
            + 'snprintf(filename, FILE_LENGTH, "%s_output%u_%s.txt", filename_prefix, i, shape);\n'
        )
        if board == "anole":
            save_output += " " * 8 + "shl_ovx_save_output(i, filename, sess);\n"
        else:
            save_output += " " * 8 + "int output_size = csinn_tensor_size(foutput);\n"
            save_output += (
                " " * 8 + "save_data_to_file(filename, (float*)foutput->data, output_size);\n"
            )
    postprocess_str = postprocess_str.replace("#_save_output_stats_#", save_output)
    code_str = code_str.replace("#_hhb_postprocess_def_#", postprocess_str)

    if board == "c920":
        if "save" in postprocess:
            code_str = code_str.replace("#_hhb_c920_postprocess_#", "1")
        else:
            code_str = code_str.replace("#_hhb_c920_postprocess_#", "0")

    #######################################################################
    #
    # Main Codegen
    #
    code_str = code_str.replace("#_input_num#", str(len(input_shape)))
    code_str = code_str.replace("#_output_num#", str(len(output_shape)))

    hhb_gen_register = ""
    if hhb_gen:
        hhb_gen_register = "void hhb_gen_register(); hhb_gen_register();"
    code_str = code_str.replace("#_hhb_gen_register_#", hhb_gen_register)

    aligned_buffer_stats = ""
    if input_memory_type and (1 in input_memory_type):
        aligned_buffer_stats += "void *input_aligned[input_num];\n"
        aligned_buffer_stats += " " * 4 + "for (i = 0; i < input_num; i++) {\n"
        aligned_buffer_stats += (
            " " * 8
            + "input_size[i] = csinn_tensor_byte_size(((struct csinn_session *)sess)->input[i]);\n"
        )
        aligned_buffer_stats += (
            " " * 8 + "input_aligned[i] = shl_mem_alloc_aligned(input_size[i], 0);\n"
        )
        aligned_buffer_stats += " " * 4 + "}\n"
    code_str = code_str.replace("#_aligned_buffer_stats_#", aligned_buffer_stats)

    aligned_buffer_copy = ""
    aligned_buffer_free = ""
    if input_memory_type:
        for i in range(len(input_shape)):
            if input_memory_type[i] == 1:  # cpu aligned
                if i != 0:
                    aligned_buffer_copy += " " * 8
                aligned_buffer_copy += (
                    "memcpy(input_aligned["
                    + str(i)
                    + "], input["
                    + str(i)
                    + "], input_size["
                    + str(i)
                    + "]);\n"
                )
                aligned_buffer_copy += " " * 8
                aligned_buffer_copy += (
                    "input_tensors[" + str(i) + "]->data = input_aligned[" + str(i) + "];\n"
                )
                if i == 0:
                    aligned_buffer_free += "shl_mem_free(input_aligned[j]);"
            else:
                if i != 0:
                    aligned_buffer_copy += " " * 8
                aligned_buffer_copy += (
                    "input_tensors[" + str(i) + "]->data = input[" + str(i) + "];\n"
                )
    code_str = code_str.replace("#_aligned_buffer_copy_#", aligned_buffer_copy)
    code_str = code_str.replace("#_aligned_buffer_free_#", aligned_buffer_free)

    get_input_data_stats = ""
    if without_preprocess:
        get_input_data_stats += "if (get_file_type(data_path[i * input_num + j]) != FILE_BIN) {\n"
        get_input_data_stats += (
            " " * 16
            + 'printf("Please input binary files, since you compiled the model without preprocess.\\n");\n'
        )
        get_input_data_stats += " " * 16 + "return -1;\n"
        get_input_data_stats += " " * 12 + "}\n"
        get_input_data_stats += (
            " " * 12
            + "inputf[j] = (float*)get_binary_from_file(data_path[i * input_num + j], NULL);\n"
        )
    else:
        is_rgb = 1
        if preprocess_params["gray"]:
            is_rgb = 0

        if preprocess_params["pixel_format"] == "RGB":
            to_bgr = 0
        elif preprocess_params["pixel_format"] == "BGR":
            to_bgr = 1
        get_input_data_stats += (
            "int input_len = csinn_tensor_size(((struct csinn_session *)sess)->input[j]);\n"
        )
        get_input_data_stats += (
            " " * 12
            + "struct image_data *img = get_input_data(data_path[i * input_num + j], input_len);\n"
        )
        get_input_data_stats += (
            " " * 12
            + "if (get_file_type(data_path[i * input_num + j]) == FILE_PNG || get_file_type(data_path[i * input_num + j]) == FILE_JPEG) {\n"
        )
        get_input_data_stats += (
            " " * 16 + "preprocess(img, " + str(is_rgb) + ", " + str(to_bgr) + ");\n"
        )
        get_input_data_stats += " " * 12 + "}\n"
        get_input_data_stats += " " * 12 + "inputf[j] = img->data;\n"
        get_input_data_stats += " " * 12 + "free_image_data(img);\n"
    code_str = code_str.replace("#_get_input_data_stats_#", get_input_data_stats)

    run_csinn_stats_anole = ""
    run_csinn_stats_thead = ""
    if dynamic_shape:
        run_csinn_stats_anole += """
    if (option->input_number) {
        for (int i = 0; i < option->input_number; i++) {
            input_tensors[i] = csinn_alloc_tensor(NULL);
            input_tensors[i]->dim_count = option->input_shape[i].dim_count;
            for (int j = 0; j < input_tensors[i]->dim_count; j++) {
                input_tensors[i]->dim[j] = option->input_shape[i].dims[j];
            }
            csinn_update_input(i, input_tensors[i], sess);
        }
    } else {
        printf("Use --input-shape to set input shape\\n");
        return 0;
    }
        """
    else:
        for i in range(len(input_shape)):
            if i > 0:
                run_csinn_stats_anole += "    "
            run_csinn_stats_anole += f"input_tensors[{i}] = csinn_alloc_tensor(NULL);\n"
            run_csinn_stats_anole += f"    input_tensors[{i}]->dim_count = {len(input_shape[i])};\n"
            for j, s in enumerate(input_shape[i]):
                run_csinn_stats_anole += f"    input_tensors[{i}]->dim[{j}] = {s};\n"

            run_csinn_stats_thead += "input[" + str(i) + "], "
    code_str = code_str.replace("#_tensor_shape_#", run_csinn_stats_anole)

    return code_str


def main_c_codegen(
    codegen_obj: HHBBoardQnnCodegenIR,
    input_shape,
    output_shape,
    board,
    output_path,
    postprocess="top5",
    model_save="run_only",
    without_preprocess=False,
    preprocess_params=None,
    multithread=False,
    input_memory_type=None,
    q_scheme=None,
    dynamic_shape=None,
    hhb_gen=False,
):
    """Generate the main.c file"""

    execute_path = get_execute_path()
    if board == "anole":
        if multithread:
            main_file = os.path.join(execute_path, "config", "anole_multithread.tp")
        else:
            main_file = os.path.join(execute_path, "config", "anole.tp")
    elif board in ("th1520", "hth1520"):
        main_file = os.path.join(execute_path, "config", "th1520.tp")
    elif board in ("c906", "rvm", "c908"):
        if dynamic_shape:
            main_file = os.path.join(execute_path, "config", "c906_cmdline.tp")
        else:
            main_file = os.path.join(execute_path, "config", "c906.tp")
    elif board == "c920":
        main_file = os.path.join(execute_path, "config", "c920.tp")
    else:
        without_preprocess = True
        main_file = os.path.join(execute_path, "config", "x86_ref.tp")

    code_str = entry_c_codegen(
        codegen_obj,
        input_shape,
        output_shape,
        board,
        output_path,
        main_file,
        postprocess,
        model_save,
        without_preprocess,
        preprocess_params,
        multithread,
        input_memory_type,
        q_scheme,
        dynamic_shape,
        hhb_gen,
    )

    logger.info("save main souce to %s", os.path.join(output_path, codegen_obj.main_source_name))
    with open(os.path.join(output_path, codegen_obj.main_source_name), "w") as f:
        f.write(code_str)


def jit_c_codegen(
    codegen_obj: HHBBoardQnnCodegenIR,
    input_shape,
    output_shape,
    board,
    output_path,
    preprocess_params=None,
    q_scheme=None,
):
    """Generate the main.c file"""

    execute_path = get_execute_path()
    if board in ("th1520", "hth1520", "c920"):
        main_file = os.path.join(execute_path, "config", "th1520_jit.tp")

    code_str = entry_c_codegen(
        codegen_obj,
        input_shape,
        output_shape,
        board,
        output_path,
        main_file,
        "top5",
        "run_only",
        True,
        preprocess_params,
        False,
        None,
        q_scheme,
        None,
        False,
    )

    logger.info("save jit souce to %s", os.path.join(output_path, "jit.c"))
    with open(os.path.join(output_path, "jit.c"), "w") as f:
        f.write(code_str)


def _preprocess_macro_define(preprocess_params, preprocess_str):
    if len(preprocess_params["data_mean"]) not in (1, 3):
        raise HHBException(
            "do not know how to deal with mean values:{}".format(preprocess_params["data_mean"])
        )
    if preprocess_params["add_preprocess_node"]:
        preprocess_params["data_mean"] = [0, 0, 0]
        preprocess_params["data_scale"] = 1.0
    if len(preprocess_params["data_mean"]) == 1:
        preprocess_params["data_mean"] = preprocess_params["data_mean"] * 3
    data_resize = preprocess_params["data_resize"]
    if isinstance(data_resize, int):
        data_resize = [data_resize, 0]
    preprocess_params_code = ""
    preprocess_params_code += "#define RESIZE_HEIGHT" + "       " + str(data_resize[0]) + "\n"
    preprocess_params_code += "#define RESIZE_WIDTH" + "        " + str(data_resize[1]) + "\n"
    preprocess_params_code += (
        "#define CROP_HEGHT" + "          " + str(preprocess_params["target_shape"][0]) + "\n"
    )
    preprocess_params_code += (
        "#define CROP_WIDTH" + "          " + str(preprocess_params["target_shape"][1]) + "\n"
    )
    data_mean = preprocess_params["data_mean"]
    if preprocess_params["pixel_format"] == "BGR":
        data_mean = data_mean[::-1]
    preprocess_params_code += "#define R_MEAN" + "              " + str(data_mean[0]) + "\n"
    preprocess_params_code += "#define G_MEAN" + "              " + str(data_mean[1]) + "\n"
    preprocess_params_code += "#define B_MEAN" + "              " + str(data_mean[2]) + "\n"
    preprocess_params_code += (
        "#define SCALE" + "               " + str(preprocess_params["data_scale"]) + "\n"
    )
    preprocess_str = preprocess_str.replace("#_preprocess_define_#", preprocess_params_code)
    return preprocess_str


class VisitLayers(relay.ExprVisitor):
    """get layer kinds"""

    def __init__(self, func):
        super(VisitLayers, self).__init__()
        self.layer_kinds = []
        self.visit(func)

    def visit_call(self, call):
        _ = [self.visit(arg) for arg in call.args]

        op_name = call.op.name
        if op_name == "qnn.csi.conv2d":
            in_shape = list(call.type_args[0].concrete_shape)
            kernel_shape = list(call.type_args[1].concrete_shape)
            if call.attrs.groups > 1:
                op_name = "group_conv2d"
                if call.attrs.out_layout == "NCHW":
                    if call.attrs.groups == in_shape[0] and kernel_shape[1] == 1:
                        op_name = "depthwise_conv2d"
        if op_name not in self.layer_kinds:
            self.layer_kinds.append(op_name)

    def get_op_kinds(self):
        return self.layer_kinds


def generate_c906_cb_reg(model, board, dump_file_path, q_scheme):

    c906_cb_reg_map = {
        "abs": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_ABS", "NULL", "shl_c906_abs_fp16", "shl_gref_abs"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_ABS", "NULL", "shl_c906_abs_f32", "shl_gref_abs"],
        },
        "add": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_ADD", "NULL", "shl_c906_add_fp16", "shl_gref_add"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_ADD", "NULL", "shl_c906_add_f32", "shl_gref_add"],
        },
        "avgpool2d": {
            "CSINN_DTYPE_FLOAT16": [
                "CSINN_OP_AVGPOOL2D",
                "shl_c906_avgpool2d_init",
                "NULL",
                "shl_gref_avgpool2d",
            ],
            "CSINN_DTYPE_FLOAT32": [
                "CSINN_OP_AVGPOOL2D",
                "shl_c906_avgpool2d_init",
                "NULL",
                "shl_gref_avgpool2d",
            ],
        },
        "cache_matmul": {
            "CSINN_DTYPE_FLOAT16": [
                "CSINN_OP_CACHE_MATMUL",
                "shl_c906_cache_matmul_init",
                "shl_c906_cache_matmul_fp16",
                "shl_gref_cache_matmul",
            ],
        },
        "cache_conv1d": {
            "CSINN_DTYPE_FLOAT16": [
                "CSINN_OP_CACHE_CONV1D",
                "shl_c906_cache_conv1d_init",
                "shl_c906_cache_conv1d_fp16",
                "shl_gref_cache_conv1d",
            ],
        },
        "clip": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_CLIP", "NULL", "shl_c906_clip_fp16", "shl_gref_clip"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_CLIP", "NULL", "shl_c906_clip_f32", "shl_gref_clip"],
        },
        "concat": {
            "CSINN_DTYPE_FLOAT16": [
                "CSINN_OP_CONCAT",
                "NULL",
                "shl_c906_concat_fp16",
                "shl_gref_concat",
            ],
            "CSINN_DTYPE_FLOAT32": [
                "CSINN_OP_CONCAT",
                "NULL",
                "shl_c906_concat_f32",
                "shl_gref_concat",
            ],
        },
        "conv1d": {
            "CSINN_DTYPE_FLOAT16": [
                "CSINN_OP_CONV1D",
                "shl_c906_conv1d_init",
                "NULL",
                "shl_gref_conv1d",
            ],
            "CSINN_DTYPE_FLOAT32": [
                "CSINN_OP_CONV1D",
                "shl_c906_conv1d_init",
                "NULL",
                "shl_gref_conv1d",
            ],
        },
        "conv2d": {
            "CSINN_DTYPE_FLOAT16": [
                "CSINN_OP_CONV2D",
                "shl_c906_conv2d_init",
                "NULL",
                "shl_gref_conv2d",
            ],
            "CSINN_DTYPE_FLOAT32": [
                "CSINN_OP_CONV2D",
                "shl_c906_conv2d_init",
                "NULL",
                "shl_gref_conv2d",
            ],
        },
        "depthwise_conv2d": {
            "CSINN_DTYPE_FLOAT16": [
                "CSINN_OP_DEPTHWISE_CONV2D",
                "shl_c906_depthwise_conv2d_init",
                "NULL",
                "shl_gref_depthwise_conv2d",
            ],
            "CSINN_DTYPE_FLOAT32": [
                "CSINN_OP_DEPTHWISE_CONV2D",
                "shl_c906_depthwise_conv2d_init",
                "NULL",
                "shl_gref_depthwise_conv2d",
            ],
        },
        "group_conv2d": {
            "CSINN_DTYPE_FLOAT16": [
                "CSINN_OP_GROUP_CONV2D",
                "shl_c906_conv2d_init",
                "NULL",
                "shl_gref_conv2d",
            ],
            "CSINN_DTYPE_FLOAT32": [
                "CSINN_OP_GROUP_CONV2D",
                "shl_c906_conv2d_init",
                "NULL",
                "shl_gref_conv2d",
            ],
        },
        "div": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_DIV", "shl_c906_div_init", "NULL", "shl_gref_div"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_DIV", "shl_c906_div_init", "NULL", "shl_gref_div"],
        },
        "fullyconnected": {
            "CSINN_DTYPE_FLOAT16": [
                "CSINN_OP_FULLYCONNECTED",
                "shl_c906_fullyconnected_init",
                "NULL",
                "shl_gref_fullyconnected",
            ],
            "CSINN_DTYPE_FLOAT32": [
                "CSINN_OP_FULLYCONNECTED",
                "shl_c906_fullyconnected_init",
                "NULL",
                "shl_gref_fullyconnected",
            ],
        },
        "global_avgpool2d": {
            "CSINN_DTYPE_FLOAT16": [
                "CSINN_OP_GLOBAL_AVGPOOL2D",
                "NULL",
                "shl_c906_global_avgpool2d_fp16",
                "shl_gref_global_avgpool2d",
            ],
            "CSINN_DTYPE_FLOAT32": [
                "CSINN_OP_GLOBAL_AVGPOOL2D",
                "NULL",
                "shl_c906_global_avgpool2d_f32",
                "shl_gref_global_avgpool2d",
            ],
        },
        "global_maxpool2d": {
            "CSINN_DTYPE_FLOAT16": [
                "CSINN_OP_GLOBAL_MAXPOOL2D",
                "NULL",
                "shl_ref_global_maxpool2d_quant",
                "shl_gref_global_maxpool2d",
            ],
            "CSINN_DTYPE_FLOAT32": [
                "CSINN_OP_GLOBAL_MAXPOOL2D",
                "NULL",
                "shl_c906_global_maxpool2d_f32",
                "shl_gref_global_maxpool2d",
            ],
        },
        "layer_norm": {
            "CSINN_DTYPE_FLOAT16": [
                "CSINN_OP_LAYER_NORM",
                "NULL",
                "shl_c906_layer_norm_fp16",
                "shl_gref_layer_norm",
            ],
        },
        "leaky_relu": {
            "CSINN_DTYPE_FLOAT16": [
                "CSINN_OP_LEAKY_RELU",
                "NULL",
                "shl_c906_leaky_relu_fp16",
                "shl_gref_leaky_relu",
            ],
            "CSINN_DTYPE_FLOAT32": [
                "CSINN_OP_LEAKY_RELU",
                "NULL",
                "shl_c906_leaky_relu_f32",
                "shl_gref_leaky_relu",
            ],
        },
        "lrn": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_LRN", "NULL", "shl_c906_lrn_fp16", "shl_gref_lrn"],
        },
        "matmul": {
            "CSINN_DTYPE_FLOAT16": [
                "CSINN_OP_MATMUL",
                "NULL",
                "shl_c906_matmul_fp16",
                "shl_gref_matmul",
            ],
        },
        "maxpool2d": {
            "CSINN_DTYPE_FLOAT16": [
                "CSINN_OP_MAXPOOL2D",
                "shl_c906_maxpool2d_init",
                "NULL",
                "shl_gref_maxpool2d",
            ],
            "CSINN_DTYPE_FLOAT32": [
                "CSINN_OP_MAXPOOL2D",
                "shl_c906_maxpool2d_init",
                "NULL",
                "shl_gref_maxpool2d",
            ],
        },
        "minimum": {
            "CSINN_DTYPE_FLOAT16": [
                "CSINN_OP_MINIMUM",
                "NULL",
                "shl_c906_minimum_fp16",
                "shl_gref_minimum",
            ],
            "CSINN_DTYPE_FLOAT32": [
                "CSINN_OP_MINIMUM",
                "NULL",
                "shl_c906_minimum_f32",
                "shl_gref_minimum",
            ],
        },
        "mul": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_MUL", "NULL", "shl_c906_mul_fp16", "shl_gref_mul"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_MUL", "NULL", "shl_c906_mul_f32", "shl_gref_mul"],
        },
        "prelu": {
            "CSINN_DTYPE_FLOAT16": [
                "CSINN_OP_PRELU",
                "NULL",
                "shl_c906_prelu_fp16",
                "shl_gref_prelu",
            ],
            "CSINN_DTYPE_FLOAT32": [
                "CSINN_OP_PRELU",
                "NULL",
                "shl_c906_prelu_f32",
                "shl_gref_prelu",
            ],
        },
        "relu": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_RELU", "NULL", "shl_c906_relu_fp16", "shl_gref_relu"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_RELU", "NULL", "shl_c906_relu_f32", "shl_gref_relu"],
        },
        "relu1": {
            "CSINN_DTYPE_FLOAT16": [
                "CSINN_OP_RELU1",
                "NULL",
                "shl_c906_relu1_fp16",
                "shl_gref_relu1",
            ],
            "CSINN_DTYPE_FLOAT32": [
                "CSINN_OP_RELU1",
                "NULL",
                "shl_c906_relu1_f32",
                "shl_gref_relu1",
            ],
        },
        "relu6": {
            "CSINN_DTYPE_FLOAT16": [
                "CSINN_OP_RELU6",
                "NULL",
                "shl_c906_relu6_fp16",
                "shl_gref_relu6",
            ],
            "CSINN_DTYPE_FLOAT32": [
                "CSINN_OP_RELU6",
                "NULL",
                "shl_c906_relu6_f32",
                "shl_gref_relu6",
            ],
        },
        "reshape": {
            "CSINN_DTYPE_FLOAT16": [
                "CSINN_OP_RESHAPE",
                "NULL",
                "shl_c906_reshape_fp16",
                "shl_gref_reshape",
            ],
        },
        "split": {
            "CSINN_DTYPE_FLOAT16": [
                "CSINN_OP_SPLIT",
                "NULL",
                "shl_c906_split_fp16",
                "shl_gref_split",
            ],
            "CSINN_DTYPE_FLOAT32": [
                "CSINN_OP_SPLIT",
                "NULL",
                "shl_c906_split_f32",
                "shl_gref_split",
            ],
        },
        "sub": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_SUB", "NULL", "shl_c906_sub_fp16", "shl_gref_sub"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_SUB", "NULL", "shl_c906_sub_f32", "shl_gref_sub"],
        },
        "sum": {
            "CSINN_DTYPE_FLOAT16": [
                "CSINN_OP_SUM",
                "NULL",
                "shl_c906_sum_stride_fp16",
                "shl_gref_sum",
            ],
        },
        "transpose": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_TRANSPOSE", "shl_c906_transpose_fp16"],
        },
    }

    op_kinds = VisitLayers(model["main"]).get_op_kinds()
    if "qnn.csi.conv2d" in op_kinds:
        op_kinds.append("depthwise_conv2d")
    execute_path = get_execute_path()
    func_file = os.path.join(execute_path, "config", "reg_rewrite", "c906.tp")
    with open(func_file, "r") as f:
        code_str = f.read()
    cb_op_str = ""
    cb_op_est_str = ""
    reg_906 = "\tshl_c906_reg_op({}, {}, {}, {});\n"
    est_906 = "\tshl_c906_reg_op_est({}, {}, {});\n"
    if q_scheme == "unset":
        qtype = "CSINN_DTYPE_FLOAT16"
    elif q_scheme == "float32":
        qtype = "CSINN_DTYPE_FLOAT32"
    elif q_scheme == "float16":
        qtype = "CSINN_DTYPE_FLOAT16"
    elif q_scheme == "int8_asym_w_sym":
        qtype = "CSINN_DTYPE_INT8"
    else:
        raise HHBException("C906 unsupport\n")

    opks = []

    for op in op_kinds:
        kind = op.split(".")[-1]
        if kind in c906_cb_reg_map:
            m = c906_cb_reg_map[kind]
            cb_op_str += reg_906.format(qtype, m[qtype][0], m[qtype][1], m[qtype][2])
            cb_op_est_str += est_906.format(qtype, m[qtype][0], m[qtype][3])
        else:
            opks.append(op)

    code_str = code_str.replace("#REG_OP#", cb_op_str)
    code_str = code_str.replace("#REG_OP_EST#", cb_op_est_str)
    code_str = code_str.replace("#TARGET#", board)
    code_str = code_str.replace("#TARGET_API#", "CSINN_C906")
    with open(dump_file_path, "w+") as f:
        f.write(code_str)

    return opks


def generate_rvv_cb_reg(model, dump_file_path, q_scheme, opks):

    rvv_cb_reg_map = {
        "add": {
            "CSINN_DTYPE_INT8": ["CSINN_OP_ADD", "NULL", "shl_rvv_add_int8", "shl_gref_add"],
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_ADD", "NULL", "shl_rvv_add_fp16", "shl_gref_add"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_ADD", "NULL", "shl_rvv_add_fp32", "shl_gref_add"],
        },
        "avgpool2d": {
            "CSINN_DTYPE_INT8": [
                "CSINN_OP_AVGPOOL2D",
                "shl_rvv_global_avgpool2d_init",
                "NULL",
                "shl_gref_global_avgpool2d",
            ],
            "CSINN_DTYPE_FLOAT16": [
                "CSINN_OP_AVGPOOL2D",
                "shl_rvv_global_avgpool2d_init",
                "NULL",
                "shl_gref_global_avgpool2d",
            ],
            "CSINN_DTYPE_FLOAT32": [
                "CSINN_OP_AVGPOOL2D",
                "shl_rvv_global_avgpool2d_init",
                "NULL",
                "shl_gref_global_avgpool2d",
            ],
        },
        "concat": {
            "CSINN_DTYPE_INT8": [
                "CSINN_OP_CONCAT",
                "NULL",
                "shl_rvv_concat_int8",
                "shl_gref_concat",
            ],
            "CSINN_DTYPE_FLOAT16": [
                "CSINN_OP_CONCAT",
                "NULL",
                "shl_rvv_concat_fp16",
                "shl_gref_concat",
            ],
            "CSINN_DTYPE_FLOAT32": [
                "CSINN_OP_CONCAT",
                "NULL",
                "shl_rvv_concat_fp32",
                "shl_gref_concat",
            ],
        },
        "conv2d": {
            "CSINN_DTYPE_INT8": [
                "CSINN_OP_CONV2D",
                "shl_rvv_conv2d_init_int8",
                "NULL",
                "shl_gref_conv2d",
            ],
            "CSINN_DTYPE_FLOAT16": [
                "CSINN_OP_CONV2D",
                "shl_rvv_conv2d_init_fp16",
                "NULL",
                "shl_gref_conv2d",
            ],
            "CSINN_DTYPE_FLOAT32": [
                "CSINN_OP_CONV2D",
                "shl_rvv_conv2d_init_fp32",
                "NULL",
                "shl_gref_conv2d",
            ],
        },
        "depthwise_conv2d": {
            "CSINN_DTYPE_INT8": [
                "CSINN_OP_DEPTHWISE_CONV2D",
                "shl_rvv_depthwise_conv2d_init_int8",
                "NULL",
                "shl_gref_depthwise_conv2d",
            ],
            "CSINN_DTYPE_FLOAT16": [
                "CSINN_OP_DEPTHWISE_CONV2D",
                "shl_rvv_depthwise_conv2d_init_fp16",
                "NULL",
                "shl_gref_depthwise_conv2d",
            ],
            "CSINN_DTYPE_FLOAT32": [
                "CSINN_OP_DEPTHWISE_CONV2D",
                "shl_rvv_depthwise_conv2d_init_fp32",
                "NULL",
                "shl_gref_depthwise_conv2d",
            ],
        },
        "group_conv2d": {
            "CSINN_DTYPE_FLOAT16": [
                "CSINN_OP_GROUP_CONV2D",
                "shl_rvv_conv2d_init_fp32",
                "NULL",
                "shl_gref_conv2d",
            ],
            "CSINN_DTYPE_FLOAT32": [
                "CSINN_OP_GROUP_CONV2D",
                "shl_rvv_conv2d_init_fp16",
                "NULL",
                "shl_gref_conv2d",
            ],
        },
        "fullyconnected": {
            "CSINN_DTYPE_INT8": [
                "CSINN_OP_FULLYCONNECTED",
                "shl_rvv_fullyconnected_init",
                "NULL",
                "shl_gref_fullyconnected",
            ],
            "CSINN_DTYPE_FLOAT16": [
                "CSINN_OP_FULLYCONNECTED",
                "shl_rvv_fullyconnected_init",
                "NULL",
                "shl_gref_fullyconnected",
            ],
            "CSINN_DTYPE_FLOAT32": [
                "CSINN_OP_FULLYCONNECTED",
                "shl_rvv_fullyconnected_init",
                "NULL",
                "shl_gref_fullyconnected",
            ],
        },
        "global_avgpool2d": {
            "CSINN_DTYPE_INT8": [
                "CSINN_OP_GLOBAL_AVGPOOL2D",
                "shl_rvv_global_avgpool2d_init",
                "NULL",
                "shl_gref_global_avgpool2d",
            ],
            "CSINN_DTYPE_FLOAT16": [
                "CSINN_OP_GLOBAL_AVGPOOL2D",
                "shl_rvv_global_avgpool2d_init",
                "NULL",
                "shl_gref_global_avgpool2d",
            ],
            "CSINN_DTYPE_FLOAT32": [
                "CSINN_OP_GLOBAL_AVGPOOL2D",
                "shl_rvv_global_avgpool2d_init",
                "NULL",
                "shl_gref_global_avgpool2d",
            ],
        },
        "global_maxpool2d": {
            "CSINN_DTYPE_INT8": [
                "CSINN_OP_GLOBAL_MAXPOOL2D",
                "shl_rvv_global_maxpool2d_init",
                "NULL",
                "shl_gref_global_maxpool2d",
            ],
            "CSINN_DTYPE_FLOAT16": [
                "CSINN_OP_GLOBAL_MAXPOOL2D",
                "shl_rvv_global_maxpool2d_init",
                "NULL",
                "shl_gref_global_maxpool2d",
            ],
            "CSINN_DTYPE_FLOAT32": [
                "CSINN_OP_GLOBAL_MAXPOOL2D",
                "shl_rvv_global_maxpool2d_init",
                "NULL",
                "shl_gref_global_maxpool2d",
            ],
        },
        "leaky_relu": {
            "CSINN_DTYPE_INT8": [
                "CSINN_OP_LEAKY_RELU",
                "NULL",
                "shl_rvv_leaky_relu_int8",
                "shl_gref_leaky_relu",
            ],
            "CSINN_DTYPE_FLOAT16": [
                "CSINN_OP_LEAKY_RELU",
                "NULL",
                "shl_rvv_leaky_relu_fp16",
                "shl_gref_leaky_relu",
            ],
            "CSINN_DTYPE_FLOAT32": [
                "CSINN_OP_LEAKY_RELU",
                "NULL",
                "shl_rvv_leaky_relu_fp32",
                "shl_gref_leaky_relu",
            ],
        },
        "maxpool2d": {
            "CSINN_DTYPE_INT8": [
                "CSINN_OP_MAXPOOL2D",
                "shl_rvv_maxpool2d_init_int8",
                "NULL",
                "shl_gref_maxpool2d",
            ],
            "CSINN_DTYPE_FLOAT16": [
                "CSINN_OP_MAXPOOL2D",
                "shl_rvv_maxpool2d_init_fp16",
                "NULL",
                "shl_gref_maxpool2d",
            ],
            "CSINN_DTYPE_FLOAT32": [
                "CSINN_OP_MAXPOOL2D",
                "shl_rvv_maxpool2d_init_fp32",
                "NULL",
                "shl_gref_maxpool2d",
            ],
        },
        "mul": {
            "CSINN_DTYPE_INT8": ["CSINN_OP_MUL", "NULL", "shl_rvv_mul_int8", "shl_gref_mul"],
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_MUL", "NULL", "shl_rvv_mul_fp16", "shl_gref_mul"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_MUL", "NULL", "shl_rvv_mul_fp32", "shl_gref_mul"],
        },
        "prelu": {
            "CSINN_DTYPE_INT8": [
                "CSINN_OP_PRELU",
                "NULL",
                "shl_rvv_prelu_int8",
                "shl_gref_prelu",
            ],
            "CSINN_DTYPE_FLOAT16": [
                "CSINN_OP_PRELU",
                "NULL",
                "shl_rvv_prelu_fp16",
                "shl_gref_prelu",
            ],
            "CSINN_DTYPE_FLOAT32": [
                "CSINN_OP_PRELU",
                "NULL",
                "shl_rvv_prelu_fp32",
                "shl_gref_prelu",
            ],
        },
        "relu": {
            "CSINN_DTYPE_INT8": ["CSINN_OP_RELU", "NULL", "shl_rvv_relu_int8", "shl_gref_relu"],
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_RELU", "NULL", "shl_rvv_relu_fp16", "shl_gref_relu"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_RELU", "NULL", "shl_rvv_relu_fp32", "shl_gref_relu"],
        },
        "relu6": {
            "CSINN_DTYPE_INT8": [
                "CSINN_OP_RELU6",
                "NULL",
                "shl_rvv_relu6_int8",
                "shl_gref_relu6",
            ],
            "CSINN_DTYPE_FLOAT16": [
                "CSINN_OP_RELU6",
                "NULL",
                "shl_c906_relu6_fp16",
                "shl_gref_relu6",
            ],
            "CSINN_DTYPE_FLOAT32": [
                "CSINN_OP_RELU6",
                "NULL",
                "shl_c906_relu6_f32",
                "shl_gref_relu6",
            ],
        },
        "reshape": {
            "CSINN_DTYPE_INT8": [
                "CSINN_OP_RESHAPE",
                "NULL",
                "shl_rvv_reshape_int8",
                "shl_gref_reshape",
            ],
            "CSINN_DTYPE_FLOAT16": [
                "CSINN_OP_RESHAPE",
                "NULL",
                "shl_c906_reshape_fp16",
                "shl_gref_reshape",
            ],
            "CSINN_DTYPE_FLOAT32": [
                "CSINN_OP_RESHAPE",
                "NULL",
                "shl_rvv_reshape_fp32",
                "shl_gref_reshape",
            ],
        },
        "sum": {
            "CSINN_DTYPE_INT8": [
                "CSINN_OP_SUM",
                "NULL",
                "shl_rvv_sum_stride_int8",
                "shl_gref_sum",
            ],
        },
        "sigmoid": {
            "CSINN_DTYPE_FLOAT16": [
                "CSINN_OP_SIGMOID",
                "NULL",
                "shl_rvv_sigmoid_fp16",
                "shl_gref_sigmoid",
            ],
        },
        "softmax": {
            "CSINN_DTYPE_FLOAT16": [
                "CSINN_OP_SOFTMAX",
                "NULL",
                "shl_rvv_softmax_fp16",
                "shl_gref_softmax",
            ],
        },
    }

    if model:
        op_kinds = VisitLayers(model["main"]).get_op_kinds()
        if "qnn.csi.conv2d" in op_kinds:
            op_kinds.append("depthwise_conv2d")
    else:
        op_kinds = opks

    execute_path = get_execute_path()
    func_file = os.path.join(execute_path, "config", "reg_rewrite", "rvv.tp")
    with open(func_file, "r") as f:
        code_str = f.read()
    cb_op_str = ""
    reg_rvv = "\tshl_rvv_reg_op({}, {}, {}, {}, {});\n"
    if q_scheme == "unset":
        qtype = "CSINN_DTYPE_FLOAT16"
    elif q_scheme == "float32":
        qtype = "CSINN_DTYPE_FLOAT32"
    elif q_scheme == "float16":
        qtype = "CSINN_DTYPE_FLOAT16"
    elif q_scheme == "int8_asym_w_sym":
        qtype = "CSINN_DTYPE_INT8"
    else:
        raise HHBException("RVV unsupport\n")

    unreg = []

    for op in op_kinds:
        kind = op.split(".")[-1]
        if kind in rvv_cb_reg_map:
            m = rvv_cb_reg_map[kind]
            cb_op_str += reg_rvv.format(qtype, m[qtype][0], m[qtype][1], m[qtype][2], m[qtype][3])
        else:
            unreg.append(op)

    code_str = code_str.replace("#REG_OP#", cb_op_str)
    with open(dump_file_path, "w+") as f:
        f.write(code_str)

    for op in unreg:
        logger.info("Can not register RVV op: %s", op)


def package_sections(board, output_path, model_save):
    model_params_section = True
    binary_graph_section = False
    graph_info = False

    if model_save == "save_only":
        model_params_section = False
        binary_graph_section = True

    if board == "th1520":
        graph_info = True

    if board not in ["th1520"]:
        model_params_section = True
        binary_graph_section = False

    bm_list = []

    if binary_graph_section:
        bm_list.append(["0", "graph", os.path.join(output_path, "csi.mbs.bin")])

    if model_params_section:
        bm_list.append(["0", "params", os.path.join(output_path, "model.params")])

    if graph_info:
        bm_list.append(["0", "info", os.path.join(output_path, "graph_info.bin")])

    emit_binary_model(os.path.join(output_path, "hhb.bm"), bm_list)
