# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import logging
from pathlib import Path


logger = logging.getLogger(__name__)

def import_module_from_path(file_path):
    file_path = Path(file_path).resolve()
    # Generate a unique module name based on the file path
    module_name = file_path.as_posix().replace("/", "_").replace(".", "_")

    # Load the module specification
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    module = importlib.util.module_from_spec(spec)

    # Execute the module (loads its contents)
    spec.loader.exec_module(module)

    return module


def load_modules(paths: list[str]):
    for path in paths:
        logger.warning(f"Loading module and factories from {path}")
        import_module_from_path(path)
