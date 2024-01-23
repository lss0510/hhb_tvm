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

"""
This is the global script that set the version information of TVM.
This script runs and update all the locations that related to versions

List of affected files:
- tvm-root/thead/hhb/utils.py
"""
import os
import re
import time

# current version
# We use the version of the incoming release for code
# that is under development
__version__ = "2.9.5"
__build_time__ = time.strftime("%Y%m%d", time.localtime())

# Implementations
def update(file_name, pattern, repl):
    update = []
    hit_counter = 0
    need_update = False
    for l in open(file_name):
        result = re.findall(pattern, l)
        if result:
            assert len(result) == 1
            hit_counter += 1
            if result[0] != repl:
                l = re.sub(pattern, repl, l)
                need_update = True
                print("%s: %s->%s" % (file_name, result[0], repl))
            else:
                print("%s: version is already %s" % (file_name, repl))

        update.append(l)
    if hit_counter != 1:
        raise RuntimeError("Cannot find version in %s" % file_name)

    if need_update:
        with open(file_name, "w") as output_file:
            for l in update:
                output_file.write(l)


def update_version(file_name):
    update(file_name, r"(?<=HHB_VERSION \")[.0-9a-z]+", __version__)


def main():
    proj_root = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    # python path
    update(
        os.path.join(proj_root, "thead", "hhb", "core", "common.py"),
        r"(?<=__version__ = \")[.0-9a-z]+",
        __version__,
    )
    update(
        os.path.join(proj_root, "thead", "setup.py"),
        r"(?<=__version__ = \")[.0-9a-z]+",
        __version__,
    )
    update(
        os.path.join(proj_root, "python", "setup.py"),
        r"(?<=__version__ = \")[.0-9a-z]+",
        __version__,
    )
    update(
        os.path.join(proj_root, "thead", "hhb", "core", "common.py"),
        r"(?<=__build_time__ = \")[.0-9a-z]+",
        __build_time__,
    )
    update(
        os.path.join(proj_root, "thead", "hhb", "__init__.py"),
        r"(?<=__version__ = \")[.0-9a-z]+",
        __version__,
    )

    update_version(os.path.join(proj_root, "thead", "hhb", "config", "c906.tp"))
    update_version(os.path.join(proj_root, "thead", "hhb", "config", "c920.tp"))
    update_version(os.path.join(proj_root, "thead", "hhb", "config", "th1520.tp"))
    update_version(os.path.join(proj_root, "thead", "hhb", "config", "th1520_jit.tp"))
    update_version(os.path.join(proj_root, "thead", "hhb", "config", "process", "include", "io.h"))
    update_version(
        os.path.join(proj_root, "thead", "hhb", "config", "process", "include", "process.h")
    )
    update_version(os.path.join(proj_root, "thead", "hhb", "config", "process", "src", "io.c"))
    update_version(os.path.join(proj_root, "thead", "hhb", "config", "process", "src", "process.c"))
    update_version(
        os.path.join(proj_root, "src", "relay", "backend", "contrib", "csinn", "csinn.h")
    )


if __name__ == "__main__":
    main()
