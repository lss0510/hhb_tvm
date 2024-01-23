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

#ifndef PROCESS_H_
#define PROCESS_H_

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "io.h"

struct image_data {
  float* data;      // the data of image
  uint32_t* shape;  // the shape of image, default to HWC
  uint32_t dim;     // the number of shape
};

/* Utils to process image data*/
uint32_t get_size(struct image_data img);
float get_value(struct image_data img, uint32_t h_idx, uint32_t w_idx, uint32_t c_idx);
struct image_data* get_input_data(const char* filename, uint32_t size);
void free_image_data(struct image_data* img);

void sub_mean(struct image_data* img, float b_mean, float g_mean, float r_mean);
void data_scale(struct image_data* img, float scale);
void data_crop(struct image_data* img, uint32_t height, uint32_t width);

/* Main image processing operators */
void imread(const char* filename, struct image_data* img);
void imresize(struct image_data* img, uint32_t dst_height, uint32_t dst_width);
void imsave(struct image_data img, const char* filename);
void imrgb2bgr(struct image_data* img);
void imhwc2chw(struct image_data* img);
void im2rgb(struct image_data* img);

#endif  // PROCESS_H_
