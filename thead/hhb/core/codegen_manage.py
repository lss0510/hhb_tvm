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
import sys
import os
import shutil

from tvm import relay

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

    if board != "th1520":
        # disable_nbg = True
        model_save = "run_only"

    #######################################################################
    #
    # Header Codegen
    #
    with open(os.path.join(template_dir, "header.tp"), "r") as f:
        header_str = f.read()
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
        function_str += "\nvoid *csinn_nbg(const char *nbg_file_name);"
    else:
        function_str += "\n#define csinn_nbg(...) NULL"
    csinn_args = ""
    for i in range(len(input_shape)):
        csinn_args += "void *data" + str(i) + ", "
    function_str = function_str.replace("#_csinn_args#", csinn_args)
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
    convert_fouput = "struct csinn_tensor *foutput = shl_ref_tensor_transform_f32(output);"

    postprocess_str = postprocess_str.replace("#_convert_fouput_#", convert_fouput)

    show_top5 = ""
    if "top5" in postprocess:
        show_top5 = "shl_show_top5(foutput, sess);"
    postprocess_str = postprocess_str.replace("#_show_top5_stats_#", show_top5)

    free_output_data = ""
    if board in ("th1520", "hth1520", "c906", "rvm", "c908", "c920", "c920v2"):
        free_output_data = "shl_ref_tensor_transform_free_f32(foutput);\n"
        if board in ("c906", "rvm", "c908", "c920", "c920v2"):
            free_output_data += " " * 8 + "if (!output->is_const) {\n"
            free_output_data += " " * 12 + "shl_mem_free(output->data);\n"
            free_output_data += " " * 8 + "}"
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

    run_csinn_stats_dynamic_shape = ""
    if dynamic_shape:
        run_csinn_stats_dynamic_shape += """
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
                run_csinn_stats_dynamic_shape += "    "
            run_csinn_stats_dynamic_shape += f"input_tensors[{i}] = csinn_alloc_tensor(NULL);\n"
            if board in ("th1520", "hth1520") and input_memory_type:
                if input_memory_type[i] == 1:
                    run_csinn_stats_dynamic_shape += (
                        f"    input_tensors[{i}]->mtype = CSINN_MEM_TYPE_CPU_ALIGNED;\n"
                    )
                elif input_memory_type[i] == 2:
                    run_csinn_stats_dynamic_shape += (
                        f"    input_tensors[{i}]->mtype = CSINN_MEM_TYPE_DMABUF;\n"
                    )
            run_csinn_stats_dynamic_shape += (
                f"    input_tensors[{i}]->dim_count = {len(input_shape[i])};\n"
            )
            for j, s in enumerate(input_shape[i]):
                run_csinn_stats_dynamic_shape += f"    input_tensors[{i}]->dim[{j}] = {s};\n"

    code_str = code_str.replace("#_tensor_shape_#", run_csinn_stats_dynamic_shape)

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
    input_memory_type=None,
    q_scheme=None,
    dynamic_shape=None,
    hhb_gen=False,
):
    """Generate the main.c file"""

    execute_path = get_execute_path()
    if board in ("th1520", "hth1520"):
        main_file = os.path.join(execute_path, "config", "th1520.tp")
    elif board in ("c906", "rvm", "c908", "c920v2"):
        main_file = os.path.join(execute_path, "config", "c906.tp")
    elif board == "c920":
        main_file = os.path.join(execute_path, "config", "c920.tp")
    else:
        without_preprocess = True
        main_file = os.path.join(execute_path, "config", "c906.tp")

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
