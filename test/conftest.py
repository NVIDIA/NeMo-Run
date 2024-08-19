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

from typing import Any, Optional

from invoke.config import Config
from invoke.context import Context


class MockContext(Context):
    def __init__(self, config: Optional[Config] = None) -> None:
        defaults = Config.global_defaults()
        defaults["run"]["pty"] = True
        defaults["run"]["in_stream"] = False
        super().__init__(config=config)

    def run(self, command: str, **kwargs: Any):
        kwargs["in_stream"] = False
        runner = self.config.runners.local(self)
        return self._run(runner, command, **kwargs)
