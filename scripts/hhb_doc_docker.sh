#!/bin/bash
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

set -e
set -u
set -o pipefail

script_dir="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"

docker run --rm -v $script_dir/../../hhb:/mnt hhb_build:0.5 sh -c "cd mnt/ && rm build -rf  &&  mkdir build  &&  cd build  &&  cp ../cmake/config.cmake . && echo 'set(USE_MICRO ON)' >> config.cmake &&  cmake ..  &&  make -j32"

# docker run --rm -v $script_dir/../../hhb:/mnt yl-hub.eng.t-head.cn/flow-design/hhb_build:0.9 sh -c "cd mnt/ && ./scripts/hhb_doc.sh"
docker run --rm -v $script_dir/../../hhb:/mnt hhb_build:0.5 sh -c "cd mnt/ && ./scripts/hhb_doc.sh"

cd docs/_build/html
python3 -m http.server

echo "Now you can browse the html pages as these two steps:"
echo "1. execute the following commands to start server:"
echo "    cd docs/_build/html"
echo "    python3 -m http.server"
echo "2. switch to your browser and type this url:"
echo "    server_ip:8000"