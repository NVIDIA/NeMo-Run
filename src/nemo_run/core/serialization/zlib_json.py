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

import base64
import zlib
from typing import Optional

from fiddle._src import config
from fiddle._src.experimental import serialization


class ZlibJSONSerializer:
    """Serializer that uses JSON, zlib, and base64 encoding."""

    def serialize(
        self,
        cfg: config.Buildable,
        pyref_policy: Optional[serialization.PyrefPolicy] = None,
    ) -> str:
        return base64.urlsafe_b64encode(
            zlib.compress(serialization.dump_json(cfg, pyref_policy).encode())
        ).decode("ascii")

    def deserialize(
        self,
        serialized: str,
        pyref_policy: Optional[serialization.PyrefPolicy] = None,
    ) -> config.Buildable:
        return serialization.load_json(
            zlib.decompress(base64.urlsafe_b64decode(serialized)).decode(),
            pyref_policy=pyref_policy,
        )
