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

import logging

from rich.console import Console, _is_jupyter
from rich.logging import RichHandler

CONSOLE = Console()


class CustomConfigRepr:
    def __init__(self, obj):
        self.obj = obj

    def __repr__(self):
        original_repr = repr(self.obj)
        # Remove the specific patterns from the representation
        cleaned_repr = original_repr.replace("<Config[", "").replace("]>", "")
        return cleaned_repr


def configure_logging(level: str):
    handlers = [RichHandler(console=CONSOLE)]
    if _is_jupyter():
        handlers = None
    logging.basicConfig(
        level=logging.getLevelName(level.upper()),
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers,
        force=True,
    )


def deconfigure_logging():
    handlers = logging.getLogger().handlers
    if len(handlers) > 0:
        logging.getLogger().removeHandler(logging.getLogger().handlers[0])
