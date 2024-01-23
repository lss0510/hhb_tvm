/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/* auto generate by HHB_VERSION "2.9.5" */

#include "process.h"

#include "io.h"
#include "jpeglib.h"
#include "png.h"
#include "zlib.h"

#define LINEAR_INTERPOLATION(l_value, r_value, coff) \
  ({ (1 - (coff)) * (l_value) + (coff) * (r_value); })

/******************************************************************************
 *                                                                            *
 *                      Static Functions                                      *
 *                                                                            *
 * ***************************************************************************/

/*!
 * \brief Clip data to range: [v_min, v_max]
 *
 * \param data The value will be clip
 * \param v_min The left boundary
 * \param v_max The right boundary
 * \return The clipped value
 *
 */
static float _clip(float data, float v_min, float v_max) {
  data = data >= v_min ? data : v_min;
  data = data <= v_max ? data : v_max;
  return data;
}

/*!
 * \brief Read JPEG image, and convert to image_data with float data.
 *
 * \param filename The file name of JPEG image.
 * \param img The pointer of struct image_data, which will hold the decompressed
 *                  image data.
 */
static void _read_jpeg(const char* filename, struct image_data* img) {
  FILE* file = NULL;
  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;
  JSAMPARRAY buffer;
  uint32_t width, height, depth;
  uint8_t* point = NULL;
  uint8_t* origin_data = NULL;
  int i;

  file = fopen(filename, "rb");
  if (file == NULL) {
    printf("Fail to open %s\n", filename);
  }

  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_decompress(&cinfo);
  jpeg_stdio_src(&cinfo, file);
  jpeg_read_header(&cinfo, TRUE);

  // cinfo.dct_method = JDCT_IFAST;

  jpeg_start_decompress(&cinfo);
  width = cinfo.output_width;
  height = cinfo.output_height;
  depth = cinfo.output_components;

  origin_data = (uint8_t*)malloc(sizeof(uint8_t) * (width * height * depth));  // NOLINT
  img->data = (float*)malloc(sizeof(float) * (width * height * depth));        // NOLINT
  img->shape = (uint32_t*)malloc(sizeof(uint32_t) * 3);                        // NOLINT
  img->shape[0] = height;
  img->shape[1] = width;
  img->shape[2] = depth;
  img->dim = 3;

  buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr)&cinfo, JPOOL_IMAGE, width * depth, 1);
  point = origin_data;
  while (cinfo.output_scanline < height) {
    jpeg_read_scanlines(&cinfo, buffer, 1);
    memcpy(point, *buffer, width * depth);
    point += width * depth;
  }
  jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);
  for (i = 0; i < width * height * depth; i++) {
    img->data[i] = (float)origin_data[i];  // NOLINT
  }
  if (file) fclose(file);
  if (origin_data) free(origin_data);
}

/*!
 * \brief Read PNG image, and convert to image_data with float data.
 *
 * \param filename The file name of JPEG image.
 * \param img The pointer of struct image_data, which will hold the decompressed
 *                  image data.
 */
static void _read_png(const char* filename, struct image_data* img) {
  png_image image;  /* The control structure used by libpng */
  png_bytep buffer; /* Pixel value buffer */
  uint32_t i;
  /* Initialize the 'png_image' structure. */
  memset(&image, 0, (sizeof image));
  image.version = PNG_IMAGE_VERSION;
  /* The first argument is the file to read: */
  if (png_image_begin_read_from_file(&image, filename) != 0) {
    //  image.format = PNG_FORMAT_RGBA;
    buffer = malloc(PNG_IMAGE_SIZE(image));
    img->dim = 3;
    img->shape = (uint32_t*)malloc(sizeof(uint32_t) * 3);  // NOLINT
    img->shape[0] = image.height;
    img->shape[1] = image.width;
    img->shape[2] = PNG_IMAGE_PIXEL_CHANNELS(image.format);
    img->data = (float*)malloc(PNG_IMAGE_SIZE(image) * sizeof(float));  // NOLINT
    if (buffer != NULL && png_image_finish_read(&image, NULL /*background*/, buffer,
                                                0 /*row_stride*/, NULL /*colormap*/) != 0) {
      for (i = 0; i < PNG_IMAGE_SIZE(image); i++) {
        img->data[i] = buffer[i];
      }
    } else {
      if (buffer == NULL)
        png_image_free(&image);
      else
        free(buffer);
    }
  } else {
    printf("pngtopng: error: %s\n", image.message);
    exit(1);
  }
}

/*!
 * \brief Get data from tensor file or text file.
 * Note that: Only One data in a line in the file.
 *
 * \param filename The file path, the suffix is .tensor or .txt
 * \param size The number of data items
 *
 */
static float* _get_data_from_file(const char* filename, uint32_t size) {
  uint32_t j;
  float fval = 0.0;
  float* buffer = NULL;
  FILE* fp = fopen(filename, "rb");
  if (fp == NULL) {
    printf("Invalid input file: %s\n", filename);
    return NULL;
  }

  buffer = malloc(size * sizeof(float));
  if (buffer == NULL) {
    printf("Malloc fail\n");
    return NULL;
  }
  for (j = 0; j < size; j++) {
    if (fscanf(fp, "%f ", &fval) != 1) {
      printf("Invalid input file\n");
      return NULL;
    } else {
      buffer[j] = fval;
    }
  }

  fclose(fp);
  return buffer;
}

/*!
 * \brief Obain the number of pixels in given image.
 *
 *  \param img The object of struct image_data
 *  \return The number of pixels
 */
uint32_t get_size(struct image_data img) {
  uint32_t i;
  uint32_t sz = 1;
  for (i = 0; i < img.dim; i++) {
    sz *= img.shape[i];
  }
  return sz;
}

/*!
 * \brief Get Value of image at (h_idx, w_idx, c_idx)
 *
 * \param img The pointer of struct image_data
 * \param h_idx The index value of the point's height
 * \param w_idx The index value of the point's width
 * \param c_idx The index value of the point's channel
 * \return The pixel value of image at (h_idx, w_idx, c_idx)
 *
 */
float get_value(struct image_data img, uint32_t h_idx, uint32_t w_idx, uint32_t c_idx) {
  int32_t height = img.shape[0];
  int32_t width = img.shape[1];
  int32_t channel = img.shape[2];
  if (h_idx < 0 || h_idx >= height || w_idx < 0 || w_idx >= width || c_idx < 0 ||
      c_idx >= channel) {
    printf("Invalid shape index! (%d, %d, %d)\n", h_idx, w_idx, c_idx);
    exit(1);
  }
  uint32_t idx = h_idx * (width * channel) + w_idx * channel + c_idx;
  return img.data[idx];
}

/*!
 * \brief Get the data of the specified file
 * Generally, the data obtained from tensor file can be directly used for model
 * inference while the data obtained from image file needs further preprocessing.
 *
 * \param filename The path of data file
 * \param size The expected number of data. If the file is image, this param will
 *  be ignored
 * \return The object struct image_data that contain the loaded image data
 */
struct image_data* get_input_data(const char* filename, uint32_t size) {
  enum file_type type;
  struct image_data* img = calloc(1, sizeof(struct image_data));
  type = get_file_type(filename);
  if (type == FILE_JPEG || type == FILE_PNG) {
    // read data from image
    imread(filename, img);
  } else if (type == FILE_TENSOR) {
    // read data from tensor or txt file.
    img->data = _get_data_from_file(filename, size);
  } else if (type == FILE_BIN) {
    img->data = (float*)get_binary_from_file(filename, NULL);
  } else {
    free(img);
    return NULL;
  }
  return img;
}

void free_image_data(struct image_data* img) {
  if (img->shape) {
    free(img->shape);
  }
  free(img);
}

/*!
 * \brief Substract mean values(RGB). If the channel of data is 1, then use
 * r_mean only.
 *
 * \param img The pointer of struct image_data
 * \param r_mean The mean value of r-channel in img->data
 * \param g_mean The mean value of g-channel in img->data that will be ignored
 *               if the dim of original image's channel is 1
 * \param b_mean The mean value of b-channel in img->data that will be ignored
 *               if the dim of original image's channel is 1
 */
void sub_mean(struct image_data* img, float r_mean, float g_mean, float b_mean) {
  uint32_t sz, channel;
  uint32_t idx;

  channel = img->shape[2];
  if (channel != 1 && channel != 3) {
    printf("Don't know how to sub mean with channel=%d\n", channel);
    exit(1);
  }
  sz = get_size(*img);
  for (idx = 0; idx < sz; idx += channel) {
    if (channel == 1) {
      img->data[idx] -= r_mean;
    } else {
      img->data[idx + 0] -= r_mean;
      img->data[idx + 1] -= g_mean;
      img->data[idx + 2] -= b_mean;
    }
  }
}

/*!
 * \brief Scale the image data with specified value.
 *
 * \param img The pointer of struct image_data
 * \param scale All the data in image will be multiplied by this value
 */
void data_scale(struct image_data* img, float scale) {
  uint32_t idx;
  for (idx = 0; idx < get_size(*img); idx++) {
    img->data[idx] *= scale;
  }
}

/**
 * \brief Crop the image data with specified shape, using central crop method.
 *
 * \param img The pointer of struct image_data
 * \param height crop the height of data by height value
 * \param width crop the width of data by width value
 *
 */
void data_crop(struct image_data* img, uint32_t height, uint32_t width) {
  uint32_t ori_width, ori_height, ori_channel;
  uint32_t row, col, c;
  uint32_t start_row, start_col;

  if (img->shape[0] == height && img->shape[1] == width) {
    return;
  }

  ori_height = img->shape[0];
  ori_width = img->shape[1];
  ori_channel = img->shape[2];

  if (width > ori_width || height > ori_height) {
    printf("Can not crop data by (%d, %d)\n", height, width);
    exit(1);
  }
  float* new_data = (float*)malloc(sizeof(float) * (height * width * ori_channel));  // NOLINT

  start_row = ori_height / 2 - height / 2;
  start_col = ori_width / 2 - width / 2;
  for (row = 0; row < height; row++) {
    for (col = 0; col < width; col++) {
      for (c = 0; c < ori_channel; c++) {
        new_data[row * (width * ori_channel) + col * ori_channel + c] =
            get_value(*img, start_row + row, start_col + col, c);
      }
    }
  }
  free(img->data);
  img->data = new_data;
  img->shape[0] = height;
  img->shape[1] = width;
}

/******************************************************************************
 *                                                                            *
 *                      Main image processing ops                             *
 *                                                                            *
 * ***************************************************************************/

/*!
 * \brief Save image data into file as JPEG format.
 * Reference to JPEG-9d: example.c
 *
 * \param img The object of struct image_data
 * \param filename The destination filename
 */
void imsave(struct image_data img, const char* filename) {
  if (img.shape[2] != 3) {
    printf("Error image data with channel %d\n", img.shape[2]);
    exit(1);
  }
  uint8_t* img_buffer = (uint8_t*)malloc(sizeof(uint8_t) * get_size(img));  // NOLINT
  struct jpeg_compress_struct cinfo;
  // This struct represents a JPEG error handler.
  struct jpeg_error_mgr jerr;
  /* More stuff */
  FILE* outfile;           /* target file */
  JSAMPROW row_pointer[1]; /* pointer to JSAMPLE row[s] */
  int row_stride;          /* physical row width in image buffer */

  /*Get image buffer*/
  uint32_t i;
  for (i = 0; i < get_size(img); i++) {
    img_buffer[i] = (uint8_t)img.data[i];
  }

  /* Step 1: allocate and initialize JPEG compression object */
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_compress(&cinfo);

  /* Step 2: specify data destination (eg, a file) */
  if ((outfile = fopen(filename, "wb")) == NULL) {
    printf("can't open %s\n", filename);
    exit(1);
  }
  jpeg_stdio_dest(&cinfo, outfile);

  /* Step 3: set parameters for compression */
  cinfo.image_width = img.shape[1]; /* image width and height, in pixels */
  cinfo.image_height = img.shape[0];
  cinfo.input_components = 3;     /* # of color components per pixel */
  cinfo.in_color_space = JCS_RGB; /* colorspace of input image */

  jpeg_set_defaults(&cinfo);
  jpeg_set_quality(&cinfo, 100, TRUE);

  /* Step 4: Start compressor */
  jpeg_start_compress(&cinfo, TRUE);

  /* Step 5: while (scan lines remain to be written) */
  /*           jpeg_write_scanlines(...); */
  row_stride = img.shape[1] * 3; /* JSAMPLEs per row in image_buffer */
  while (cinfo.next_scanline < cinfo.image_height) {
    row_pointer[0] = &img_buffer[cinfo.next_scanline * row_stride];
    (void)jpeg_write_scanlines(&cinfo, row_pointer, 1);
  }

  /* Step 6: Finish compression */
  jpeg_finish_compress(&cinfo);
  /* After finish_compress, we can close the output file. */
  fclose(outfile);

  /* Step 7: release JPEG compression object */
  jpeg_destroy_compress(&cinfo);
}

/*!
 * \brief Decompress JPEG or PNG image, and convert to image_data with float data.
 *
 * \param filename The file name of JPEG image.
 * \param img The pointer of struct image_data, which will hold the decompressed
 *                  image data.
 */
void imread(const char* filename, struct image_data* img) {
  enum file_type type;
  type = get_file_type(filename);
  if (type == FILE_JPEG) {
    _read_jpeg(filename, img);
  } else if (type == FILE_PNG) {
    _read_png(filename, img);
  }
}

/*!
 * \brief Resize the image into target image size with bilinear interpolation method.
 *
 *            |                   |                  |
 *            |                   |                  |
 *  ---p00(srcY_i, srcX_i)--------f1------p01(srcY_i, srcX_i+1)-----
 *            |                   |                  |
 *            |   p(srcY_i+h_coff, srcX_i+w_coff)    |
 *            |                   |                  |
 *  ---p10(srcY_i+1, srcX_i)------f2-------p11(srcY_i+1, srcY_i)-----
 *            |                   |                  |
 *            |                   |                  |
 *
 * srcX(or srcY) can be got by:
 *      src = (dst + 0.5) * scale - 0.5
 * and
 *      coff = src - floor(src) which denotes the weight in single Linear interpolation.
 * Finaly, we can get the value as following:
 *      f1 = p00 * (1-coff1) + coff1 * p01
 *      f2 = p10 * (1-coff1) + coff1 * p11
 *      p = f1 * (1-coff2) + coff2 * f2
 *
 * \param img The pointer of struct image_data, which denote the image data that will
 *              be resized.
 * \param dst_height The height of image after resizing it.
 * \param dst_widht The width of image after resize it.
 *
 */
void imresize(struct image_data* img, uint32_t dst_height, uint32_t dst_width) {
  uint32_t srcX, srcY, dstX, dstY;
  float srcX_f, srcY_f;  // float index
  int srcX_i, srcY_i;    // integer index
  float w_coff, h_coff;
  float scaleX = (float)img->shape[1] / (float)dst_width;   // NOLINT
  float scaleY = (float)img->shape[0] / (float)dst_height;  // NOLINT

  float up_left, bottom_left, up_right, bottom_right;
  uint32_t c;  // index of channel
  float f1, f2;
  float* resized_data;

  if (img->shape[0] == dst_height && img->shape[1] == dst_width) {
    return;
  }
  resized_data =
      (float*)malloc(sizeof(float) * (dst_height * dst_width * img->shape[2]));  // NOLINT
  for (dstY = 0; dstY < dst_height; dstY++) {
    for (dstX = 0; dstX < dst_width; dstX++) {
      // Get the mapping position of the current point in the original image
      srcX_f = ((float)dstX + 0.5) * scaleX - 0.5;  // NOLINT
      srcY_f = ((float)dstY + 0.5) * scaleY - 0.5;  // NOLINT
      // Get weight in interpolation
      w_coff = srcX_f - floor(srcX_f);
      h_coff = srcY_f - floor(srcY_f);

      srcX_i = floor(srcX_f);
      srcY_i = floor(srcY_f);

      for (c = 0; c < img->shape[2]; c++) {
        // Get the pixel values of four points around
        up_left = get_value(*img, _clip(srcY_i, 0, img->shape[0] - 1),
                            _clip(srcX_i, 0, img->shape[1] - 1), c);
        up_right = get_value(*img, _clip(srcY_i, 0, img->shape[0] - 1),
                             _clip(srcX_i + 1, 0, img->shape[1] - 1), c);
        bottom_left = get_value(*img, _clip(srcY_i + 1, 0, img->shape[0] - 1),
                                _clip(srcX_i, 0, img->shape[1] - 1), c);
        bottom_right = get_value(*img, _clip(srcY_i + 1, 0, img->shape[0] - 1),
                                 _clip(srcX_i + 1, 0, img->shape[1] - 1), c);

        // Horizontal linear interpolation
        f1 = LINEAR_INTERPOLATION(up_left, up_right, w_coff);
        f2 = LINEAR_INTERPOLATION(bottom_left, bottom_right, w_coff);
        // Vertical linear interpolation
        resized_data[dstY * (dst_width * img->shape[2]) + dstX * img->shape[2] + c] =
            LINEAR_INTERPOLATION(f1, f2, h_coff);
      }
    }
  }
  // Updata data in place
  free(img->data);
  img->data = NULL;
  img->shape[0] = dst_height;
  img->shape[1] = dst_width;
  img->data = resized_data;
}

/*!
 * \brief Convert image from RGB to BGR.
 *
 * \param img The pointer of struct image_data
 */
void imrgb2bgr(struct image_data* img) {
  uint32_t idx;
  float tmp;
  if (img->dim != 3) {
    printf("Invalid dim: %d\n", img->dim);
    return;
  }
  if (img->shape[2] == 1) {
    return;
  } else if (img->shape[2] != 3) {
    printf("Invalid channel: %d\n", img->shape[2]);
    return;
  } else {
    for (idx = 0; idx < get_size(*img); idx += 3) {
      tmp = img->data[idx];
      img->data[idx] = img->data[idx + 2];
      img->data[idx + 2] = tmp;
    }
  }
}

/*!
 * \brief Convert image data from HWC to CHW.
 *
 * \param img The pointer of struct image_data
 *
 */
void imhwc2chw(struct image_data* img) {
  uint32_t row, col, channel;
  float* transposed_data = NULL;
  uint32_t H, W, C;
  if (img->dim != 3) {
    printf("Invalid dim: %d\n", img->dim);
    return;
  }
  H = img->shape[0];
  W = img->shape[1];
  C = img->shape[2];
  transposed_data = (float*)malloc(sizeof(float) * get_size(*img));  // NOLINT
  for (channel = 0; channel < C; channel++) {
    for (row = 0; row < H; row++) {
      for (col = 0; col < W; col++) {
        transposed_data[channel * (H * W) + row * W + col] = get_value(*img, row, col, channel);
      }
    }
  }
  // Updata image data
  free(img->data);
  img->data = transposed_data;
  img->shape[0] = C;
  img->shape[1] = H;
  img->shape[2] = W;
}

/*!
 * \brief Convert non-RGB data to rgb data.
 * For example, the shape of gray image data is (h ,w, 1) and the shape of
 * RGBA image data is (h, w, 4), all of these image data should be convert
 * to (h, w, 3) if neccesary.
 *
 * \param img The pointer of struct image_data
 *
 */
void im2rgb(struct image_data* img) {
  uint32_t idx, cnt = 0;
  float* new_data = NULL;
  uint32_t new_size, ori_size;
  uint32_t ori_channel;

  ori_channel = img->shape[2];
  if (ori_channel == 3) {
    return;
  }
  if (ori_channel == 2 || ori_channel > 4) {
    printf("Invalid dim: %d\n", ori_channel);
    exit(1);
  }
  ori_size = get_size(*img);
  new_size = img->shape[0] * img->shape[1] * 3;
  new_data = (float*)malloc(sizeof(float) * new_size);  // NOLINT

  for (idx = 0; idx < ori_size; idx++) {
    if (ori_channel == 1) {
      new_data[idx * 3 + 0] = img->data[idx];
      new_data[idx * 3 + 1] = img->data[idx];
      new_data[idx * 3 + 2] = img->data[idx];
    } else if (ori_channel == 4) {
      if ((idx + 1) % 4 == 0) continue;
      new_data[cnt] = img->data[idx];
      cnt++;
    }
  }

  free(img->data);
  img->data = new_data;
  img->shape[2] = 3;
}
