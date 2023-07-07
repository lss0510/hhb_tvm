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

if(USE_CSINN)
  find_package(ZLIB REQUIRED)

  add_definitions(-DUSE_JSON_RUNTIME=1)
  file(GLOB CSINN_RELAY_CONTRIB_SRC src/relay/backend/contrib/csinn/*.cc)

  list(APPEND COMPILER_SRCS ${CSINN_RELAY_CONTRIB_SRC})
  file(GLOB SHL_RUNTIME_SRC src/runtime/contrib/shl/shl_json_runtime.cc)
  list(APPEND RUNTIME_SRCS ${SHL_RUNTIME_SRC})
  file(GLOB CSINN_CONTRIB_SRC src/runtime/contrib/csinn/*.cc)
  #list(APPEND RUNTIME_SRCS ${CSINN_CONTRIB_SRC})
  if(USE_CSINN STREQUAL "ON")
    set(CSINN_PATH "${tvm_SOURCE_DIR}/install_nn2/x86/")
    include_directories(${CSINN_PATH}/include ${CSINN_PATH}/include/csinn)
    set(CSINN_CONTRIB_LIB "${CSINN_PATH}/lib/libshl.a")
    message("-- CSI-NN path: ${CSINN_PATH}")
  elseif(USE_CSINN STREQUAL "C906")
    set(CSINN_PATH "${tvm_SOURCE_DIR}/install_nn2/c906/")
    include_directories(${CSINN_PATH}/include ${CSINN_PATH}/include/csinn)
    set(CSINN_CONTRIB_LIB "${CSINN_PATH}/lib/libshl.so")
    message("-- CSI-NN path: ${CSINN_PATH}")
  else()
    include_directories(${CSINN_PATH}/include ${CSINN_PATH}/include/csinn)
    find_library(CSINN_CONTRIB_LIB shl_ref_x86 ${USE_CSINN}/lib/)
    message("-- CSI-NN path: ${USE_CSINN}")
  endif()
  if(NOT EXISTS ${CSINN_CONTRIB_LIB})
    message(FATAL_ERROR "can not find CSI-NN lib at path: ${USE_CSINN}/x86/lib/libshl_ref_x86.a")
  endif()
  message("-- CSI-NN path: ${CSINN_CONTRIB_LIB}")
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${CSINN_CONTRIB_LIB})
endif(USE_CSINN)
