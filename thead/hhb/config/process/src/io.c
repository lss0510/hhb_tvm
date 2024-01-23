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

#include "io.h"

/******************************************************************************
 *                                                                            *
 *                      Utils for process data                                *
 *                                                                            *
 * ***************************************************************************/

/*!
 * \brief Get the type of file (JPEG PNG or txt/tensor by suffix
 *
 * \param filename Image file name.
 * \return FILE_PNG FILE_JPEG or FILE_TENSOR of enum file_type
 *
 */
enum file_type get_file_type(const char* filename) {
  enum file_type type = 0;
  const char* ptr;
  char sep = '.';
  uint32_t pos, n;
  char buff[32] = {0};

  ptr = strrchr(filename, sep);
  pos = ptr - filename;
  n = strlen(filename) - (pos + 1);
  strncpy(buff, filename + (pos + 1), n);

  if (strcmp(buff, "jpg") == 0 || strcmp(buff, "jpeg") == 0 || strcmp(buff, "JPG") == 0 ||
      strcmp(buff, "JPEG") == 0) {
    type = FILE_JPEG;
  } else if (strcmp(buff, "png") == 0 || strcmp(buff, "PNG") == 0) {
    type = FILE_PNG;
  } else if (strcmp(buff, "tensor") == 0) {
    type = FILE_TENSOR;
  } else if (strcmp(buff, "txt") == 0) {
    type = FILE_TXT;
  } else if (strcmp(buff, "bin") == 0) {
    type = FILE_BIN;
  } else {
    printf("Unsupport for .%s file\n", buff);
    exit(1);
  }
  return type;
}

/*!
 * \brief Save float data into file.
 *
 * \param filename The file that you will put the data into.
 * \param data The float data that you will put into file.
 * \param size The size of data.
 */
void save_data_to_file(const char* filename, float* data, uint32_t size) {
  int i = 0;
  FILE* fp = fopen(filename, "w+");
  for (i = 0; i < size; i++) {
    if (i == size - 1) {
      fprintf(fp, "%f", data[i]);
    } else {
      fprintf(fp, "%f\n", data[i]);
    }
  }
  fclose(fp);
}

/*!
 * \brief Get the binary params char from model.params
 *
 * \param filename It is generally model.params
 * \return The char data of params
 *
 */
char* get_binary_from_file(const char* filename, int* size) {
  char* buffer = NULL;
  int file_size;
  int ret;
  FILE* fp = fopen(filename, "rb");
  if (fp == NULL) {
    printf("Invalid input file: %s\n", filename);
    return NULL;
  }
  fseek(fp, 0, SEEK_END);
  file_size = ftell(fp);
  rewind(fp);

  buffer = (char*)malloc(file_size);  // NOLINT
  if (buffer == NULL) {
    printf("Malloc fail\n");
    return NULL;
  }

  ret = fread(buffer, 1, file_size, fp);
  if (ret != file_size) {
    printf("Read input file error\n");
    return NULL;
  }

  fclose(fp);
  if (size) {
    *size = file_size;
  }
  return buffer;
}

char** read_string_from_file(const char* filename, int* len) {
  char buff[MAX_FILENAME_LEN];
  char** result = (char**)malloc(sizeof(char*) * (MAX_FILE_LINE * MAX_INPUT_NUMBER));  // NOLINT
  char *find, *sep, *inter;
  FILE* fp = fopen(filename, "r");
  if (fp == NULL) {
    return NULL;
  }
  int cnt = 0;
  while (fgets(buff, sizeof(buff), fp)) {
    if (strcmp(buff, "\n") == 0) continue;
    find = strchr(buff, '\n');
    if (find) {
      *find = '\0';
    }
    sep = strtok(buff, " ");                                  // NOLINT
    inter = (char*)malloc((strlen(sep) + 2) * sizeof(char));  // NOLINT
    memcpy(inter, sep, strlen(sep) + 1);
    result[cnt++] = inter;
    while (sep != NULL) {
      sep = strtok(NULL, " ");  // NOLINT
      if (sep) {
        inter = (char*)malloc((strlen(sep) + 2) * sizeof(char));  // NOLINT
        memcpy(inter, sep, strlen(sep) + 1);
        result[cnt++] = inter;
      }
    }
  }
  *len = cnt;
  fclose(fp);
  return result;
}

uint32_t shape2string(uint32_t* shape, uint32_t dim_num, char* buf, uint32_t buf_sz) {
  uint32_t s;
  uint32_t count;
  if (NULL == shape || NULL == buf || dim_num == 0 || buf_sz == 0) {
    return 0;
  }
  count = 0;
  for (s = 0; s < dim_num; s++) {
    if (count >= buf_sz) {
      break;
    }
    count += snprintf(&buf[count], buf_sz - count, "%d_", shape[s]);
  }
  buf[count - 1] = 0;
  return count;
}