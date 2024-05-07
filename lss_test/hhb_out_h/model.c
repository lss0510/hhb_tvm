/* auto generate by HHB_VERSION 2.6.0 */

#include <csi_nn.h>

void *csinn_npu1(char *params_base) {
  struct csinn_session *sess = csinn_alloc_session();
  sess->base_run_mode = CSINN_RM_NPU_GRAPH;
  sess->base_quant_type = CSINN_QUANT_INT8_ASYM;
  sess->model.priority = 0;
  sess->base_api = CSINN_TH1520;
  sess->base_dtype = CSINN_DTYPE_INT8;
  csinn_session_init(sess);
  csinn_set_input_number(1, sess);
  csinn_set_output_number(1, sess);

  struct csinn_tensor *input = csinn_alloc_tensor(sess);
  input->name = "input@@conv2d_/conv1/Conv_PART_0_1_fuse_bias_add_/conv1/Conv_2_0";
  input->dtype = CSINN_DTYPE_INT8;
  input->layout = CSINN_LAYOUT_NCHW;
  input->dim[0] = 1;
  input->dim[1] = 3;
  input->dim[2] = 224;
  input->dim[3] = 224;
  input->dim_count = 4;
  input->qinfo = (struct csinn_quant_info *)(params_base + 0);
  input->quant_channel = 1;
  struct csinn_tensor *output_0 = csinn_alloc_tensor(sess);
  output_0->name = "output_0";
  output_0->dtype = CSINN_DTYPE_INT8;
  output_0->layout = CSINN_LAYOUT_NCHW;
  output_0->dim[0] = 1;
  output_0->dim[1] = 1;
  output_0->dim[2] = 111;
  output_0->dim[3] = 111;
  output_0->dim_count = 4;
  output_0->qinfo = (struct csinn_quant_info *)(params_base + 24);
  output_0->quant_channel = 1;
  struct csinn_tensor *kernel_0 = csinn_alloc_tensor(sess);
  kernel_0->name = "kernel_0";
  kernel_0->data = params_base + 72;
  kernel_0->is_const = 1;
  kernel_0->dtype = CSINN_DTYPE_INT8;
  kernel_0->layout = CSINN_LAYOUT_OIHW;
  kernel_0->dim[0] = 1;
  kernel_0->dim[1] = 3;
  kernel_0->dim[2] = 3;
  kernel_0->dim[3] = 3;
  kernel_0->dim_count = 4;
  kernel_0->qinfo = (struct csinn_quant_info *)(params_base + 48);
  kernel_0->quant_channel = 1;
  struct csinn_tensor *bias_0 = csinn_alloc_tensor(sess);
  bias_0->name = "bias_0";
  bias_0->data = params_base + 123;
  bias_0->is_const = 1;
  bias_0->dtype = CSINN_DTYPE_INT32;
  bias_0->layout = CSINN_LAYOUT_O;
  bias_0->dim[0] = 1;
  bias_0->dim_count = 1;
  bias_0->qinfo = (struct csinn_quant_info *)(params_base + 99);
  bias_0->quant_channel = 1;
  struct csinn_conv2d_params *params_0 = csinn_alloc_params(sizeof(struct csinn_conv2d_params), sess);
  params_0->group = 1;
  params_0->stride_height = 2;
  params_0->stride_width = 2;
  params_0->dilation_height = 1;
  params_0->dilation_width = 1;
  params_0->conv_extra.kernel_tm = NULL;
  params_0->conv_extra.conv_mode = CSINN_DIRECT;
  params_0->pad_top = 0;
  params_0->pad_left = 0;
  params_0->pad_down = 0;
  params_0->pad_right = 0;
  params_0->base.name = "conv2d_/conv1/Conv_PART_0_1_fuse_bias_add_/conv1/Conv_2";
  params_0->base.quant_type = CSINN_QUANT_INT8_ASYM;
  csinn_conv2d_init(input, output_0, kernel_0, bias_0, params_0);
  csinn_set_tensor_entry(input, sess);
  csinn_set_input(0, input, sess);

  csinn_conv2d(input, output_0, kernel_0, bias_0, params_0);
  csinn_set_output(0, output_0, sess);

  csinn_session_setup(sess);
  return sess;

}

void *csinn_cpu(char *params_base) {
  struct csinn_session *sess = csinn_alloc_session();
  sess->base_run_mode = CSINN_RM_CPU_GRAPH;
  sess->base_quant_type = CSINN_QUANT_INT8_ASYM;
  sess->model.save_mode = CSINN_RUN_ONLY;
  sess->base_api = CSINN_C920;
  sess->base_dtype = CSINN_DTYPE_INT8;
  sess->dynamic_shape = CSINN_FALSE;
  csinn_session_init(sess);
  csinn_set_input_number(1, sess);
  csinn_set_output_number(1, sess);

  struct csinn_tensor *input = csinn_alloc_tensor(sess);
  input->name = "input@@conv2d_/conv1/Conv_PART_0_1_fuse_bias_add_/conv1/Conv_2_0";
  input->dtype = CSINN_DTYPE_INT8;
  input->layout = CSINN_LAYOUT_NCHW;
  input->dim[0] = 1;
  input->dim[1] = 1;
  input->dim[2] = 111;
  input->dim[3] = 111;
  input->dim_count = 4;
  input->qinfo = (struct csinn_quant_info *)(params_base + 24);
  input->quant_channel = 1;
  struct csinn_tensor *output_2 = csinn_alloc_tensor(sess);
  output_2->name = "output_2";
  output_2->dtype = CSINN_DTYPE_INT8;
  output_2->layout = CSINN_LAYOUT_NCHW;
  output_2->dim[0] = 1;
  output_2->dim[1] = 1;
  output_2->dim[2] = 111;
  output_2->dim[3] = 111;
  output_2->dim_count = 4;
  output_2->qinfo = (struct csinn_quant_info *)(params_base + 128);
  output_2->quant_channel = 1;
  struct csinn_tensor *rhs_2 = csinn_alloc_tensor(sess);
  rhs_2->name = "rhs_2";
  rhs_2->data = params_base + 176;
  rhs_2->is_const = 1;
  rhs_2->dtype = CSINN_DTYPE_INT8;
  rhs_2->layout = CSINN_LAYOUT_O;
  rhs_2->dim[0] = 1;
  rhs_2->dim_count = 1;
  rhs_2->qinfo = (struct csinn_quant_info *)(params_base + 152);
  rhs_2->quant_channel = 1;
  struct csinn_diso_params *params_2 = csinn_alloc_params(sizeof(struct csinn_diso_params), sess);
  params_2->base.name = "power_/Pow_3";
  params_2->base.quant_type = CSINN_QUANT_INT8_ASYM;
  csinn_power_init(input, rhs_2, output_2, params_2);
  struct csinn_tensor *output_4 = csinn_alloc_tensor(sess);
  output_4->name = "output_4";
  output_4->dtype = CSINN_DTYPE_INT8;
  output_4->layout = CSINN_LAYOUT_NCHW;
  output_4->dim[0] = 1;
  output_4->dim[1] = 1;
  output_4->dim[2] = 55;
  output_4->dim[3] = 55;
  output_4->dim_count = 4;
  output_4->qinfo = (struct csinn_quant_info *)(params_base + 180);
  output_4->quant_channel = 1;
  struct csinn_tensor *kernel_4 = csinn_alloc_tensor(sess);
  kernel_4->name = "kernel_4";
  kernel_4->data = params_base + 228;
  kernel_4->is_const = 1;
  kernel_4->dtype = CSINN_DTYPE_INT8;
  kernel_4->layout = CSINN_LAYOUT_O1HW;
  kernel_4->dim[0] = 1;
  kernel_4->dim[1] = 1;
  kernel_4->dim[2] = 3;
  kernel_4->dim[3] = 3;
  kernel_4->dim_count = 4;
  kernel_4->qinfo = (struct csinn_quant_info *)(params_base + 204);
  kernel_4->quant_channel = 1;
  struct csinn_tensor *bias_4 = csinn_alloc_tensor(sess);
  bias_4->name = "bias_4";
  bias_4->data = params_base + 261;
  bias_4->is_const = 1;
  bias_4->dtype = CSINN_DTYPE_INT32;
  bias_4->layout = CSINN_LAYOUT_O;
  bias_4->dim[0] = 1;
  bias_4->dim_count = 1;
  bias_4->qinfo = (struct csinn_quant_info *)(params_base + 237);
  bias_4->quant_channel = 1;
  struct csinn_conv2d_params *params_4 = csinn_alloc_params(sizeof(struct csinn_conv2d_params), sess);
  params_4->group = 1;
  params_4->stride_height = 2;
  params_4->stride_width = 2;
  params_4->dilation_height = 1;
  params_4->dilation_width = 1;
  params_4->conv_extra.kernel_tm = NULL;
  params_4->conv_extra.conv_mode = CSINN_DIRECT;
  params_4->pad_top = 0;
  params_4->pad_left = 0;
  params_4->pad_down = 0;
  params_4->pad_right = 0;
  params_4->base.name = "conv2d_/conv2/Conv_PART_0_4_fuse_bias_add_/conv2/Conv_5";
  params_4->base.quant_type = CSINN_QUANT_INT8_ASYM;
  csinn_conv2d_init(output_2, output_4, kernel_4, bias_4, params_4);

  csinn_set_tensor_entry(input, sess);
  csinn_set_input(0, input, sess);

  csinn_power(input, rhs_2, output_2, params_2);
  csinn_conv2d(output_2, output_4, kernel_4, bias_4, params_4);
  csinn_set_output(0, output_4, sess);

  csinn_session_setup(sess);
  return sess;

}

void *csinn_npu2(char *params_base) {
  struct csinn_session *sess = csinn_alloc_session();
  sess->base_run_mode = CSINN_RM_NPU_GRAPH;
  sess->base_quant_type = CSINN_QUANT_INT8_ASYM;
  sess->model.priority = 0;
  sess->base_api = CSINN_TH1520;
  sess->base_dtype = CSINN_DTYPE_INT8;
  csinn_session_init(sess);
  csinn_set_input_number(1, sess);
  csinn_set_output_number(1, sess);


  struct csinn_tensor *input = csinn_alloc_tensor(sess);
  input->name = "input";
  input->dtype = CSINN_DTYPE_INT8;
  input->layout = CSINN_LAYOUT_NCHW;
  input->dim[0] = 1;
  input->dim[1] = 1;
  input->dim[2] = 55;
  input->dim[3] = 55;
  input->dim_count = 4;
  input->qinfo = (struct csinn_quant_info *)(params_base + 180);
  input->quant_channel = 1;
  int32_t *shape_5 = malloc(2 * 4);
  shape_5[0] = 1;
  shape_5[1] = -1;
  struct csinn_tensor *output_5 = csinn_alloc_tensor(sess);
  output_5->name = "output_5";
  output_5->dtype = CSINN_DTYPE_INT8;
  output_5->layout = CSINN_LAYOUT_NC;
  output_5->dim[0] = 1;
  output_5->dim[1] = 3025;
  output_5->dim_count = 2;
  output_5->qinfo = (struct csinn_quant_info *)(params_base + 268);
  output_5->quant_channel = 1;
  struct csinn_reshape_params *params_5 = csinn_alloc_params(sizeof(struct csinn_reshape_params), sess);
  params_5->shape = shape_5;
  params_5->shape_num = 2;
  params_5->base.name = "reshape_/Reshape_6";
  params_5->base.quant_type = CSINN_QUANT_INT8_ASYM;
  csinn_reshape_init(input, output_5, params_5);
  struct csinn_tensor *output_6 = csinn_alloc_tensor(sess);
  output_6->name = "dense_/linear/Gemm_PART_0_7_fuse_add_output@@/linear/Gemm_8_6";
  output_6->dtype = CSINN_DTYPE_INT8;
  output_6->layout = CSINN_LAYOUT_NC;
  output_6->dim[0] = 1;
  output_6->dim[1] = 32;
  output_6->dim_count = 2;
  output_6->qinfo = (struct csinn_quant_info *)(params_base + 292);
  output_6->quant_channel = 1;
  struct csinn_tensor *kernel_6 = csinn_alloc_tensor(sess);
  kernel_6->name = "kernel_6";
  kernel_6->data = params_base + 340;
  kernel_6->is_const = 1;
  kernel_6->dtype = CSINN_DTYPE_INT8;
  kernel_6->layout = CSINN_LAYOUT_OI;
  kernel_6->dim[0] = 32;
  kernel_6->dim[1] = 3025;
  kernel_6->dim_count = 2;
  kernel_6->qinfo = (struct csinn_quant_info *)(params_base + 316);
  kernel_6->quant_channel = 1;
  struct csinn_tensor *bias_6 = csinn_alloc_tensor(sess);
  bias_6->name = "bias_6";
  bias_6->data = params_base + 97164;
  bias_6->is_const = 1;
  bias_6->dtype = CSINN_DTYPE_INT32;
  bias_6->layout = CSINN_LAYOUT_O;
  bias_6->dim[0] = 32;
  bias_6->dim_count = 1;
  bias_6->qinfo = (struct csinn_quant_info *)(params_base + 97140);
  bias_6->quant_channel = 1;
  struct csinn_fc_params *params_6 = csinn_alloc_params(sizeof(struct csinn_fc_params), sess);
  params_6->units = 32;
  params_6->base.name = "dense_/linear/Gemm_PART_0_7_fuse_add_output@@/linear/Gemm_8";
  params_6->base.quant_type = CSINN_QUANT_INT8_ASYM;
  csinn_fullyconnected_init(output_5, output_6, kernel_6, bias_6, params_6);
  csinn_set_tensor_entry(input, sess);
  csinn_set_input(0, input, sess);

  csinn_reshape(input, output_5, params_5);
  csinn_fullyconnected(output_5, output_6, kernel_6, bias_6, params_6);
  csinn_set_output(0, output_6, sess);

  csinn_session_setup(sess);
  return sess;
}

void csinn_update_input_and_run(struct csinn_tensor **input_tensors , void *sess) {
  csinn_update_input(0, input_tensors[0], sess);
  csinn_session_run(sess);
}
