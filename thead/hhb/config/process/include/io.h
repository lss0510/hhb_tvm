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

/* auto generate by HHB_VERSION "2.8.1" */

#ifndef HHB_IO_H_
#define HHB_IO_H_

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_FILE_LINE 50001
#define MAX_INPUT_NUMBER 4
#define MAX_FILENAME_LEN 1280

enum file_type { FILE_PNG, FILE_JPEG, FILE_TENSOR, FILE_TXT, FILE_BIN };

/* Utils to process image data*/
enum file_type get_file_type(const char* filename);
void save_data_to_file(const char* filename, float* data, uint32_t size);
char* get_binary_from_file(const char* filename, int* size);
char** read_string_from_file(const char* filename, int* len);
uint32_t shape2string(uint32_t* shape, uint32_t dim_num, char* buf, uint32_t buf_sz);

#endif  // HHB_IO_H_
