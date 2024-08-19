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

import os

import jinja2


def fill_template(template_name: str, variables: dict) -> str:
    """Create a file from a Jinja template and return the filename."""
    assert template_name.endswith(".j2"), template_name
    root_dir = os.path.dirname(__file__)
    template_path = os.path.join(root_dir, "templates", template_name)
    if not os.path.exists(template_path):
        raise FileNotFoundError(f'Template "{template_name}" does not exist.')
    with open(template_path, "r", encoding="utf-8") as fin:
        template = fin.read()

    j2_template = jinja2.Environment(
        loader=jinja2.FileSystemLoader(os.path.join(os.path.dirname(__file__), "templates"))
    ).from_string(template)
    content = j2_template.render(**variables)
    return content
