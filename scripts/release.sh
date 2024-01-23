#!/bin/sh -x
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


rm build -rf
mkdir build
cd build
cp ../cmake/config.cmake .
cmake ..
make -j16
cd -

# pyinstaller package
cd thead
rm dist -rf
rm build -rf
pyinstaller ../scripts/hhb.spec
rm /tools/hhb -rf
cp dist/hhb /tools -r
rm /tools/hhb/libtorch_cpu.so
rm /tools/hhb/torch/lib/libtorch_cpu.so
hhb --version
