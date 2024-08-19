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

from typing import Any

from typer import rich_utils

TABLE_STYLES: dict[str, Any] = {
    "show_lines": rich_utils.STYLE_COMMANDS_TABLE_SHOW_LINES,
    "leading": rich_utils.STYLE_COMMANDS_TABLE_LEADING,
    "border_style": rich_utils.STYLE_COMMANDS_TABLE_BORDER_STYLE,
    "row_styles": rich_utils.STYLE_COMMANDS_TABLE_ROW_STYLES,
    "pad_edge": rich_utils.STYLE_COMMANDS_TABLE_PAD_EDGE,
    "padding": rich_utils.STYLE_COMMANDS_TABLE_PADDING,
}
BOX_STYLE = rich_utils.STYLE_OPTIONS_TABLE_BOX
