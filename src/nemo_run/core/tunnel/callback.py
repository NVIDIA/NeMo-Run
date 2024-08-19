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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nemo_run.core.tunnel.client import Tunnel


class Callback:
    def setup(self, tunnel: "Tunnel"):
        """Called when the tunnel is setup."""
        self.tunnel = tunnel

    def on_start(self):
        """Called when the keep_alive loop starts."""
        pass

    def on_interval(self):
        """Called at each interval during the keep_alive loop."""
        pass

    def on_stop(self):
        """Called when the keep_alive loop stops."""
        pass

    def on_error(self, error: Exception):
        """Called when an error occurs during the keep_alive loop.

        Args:
            error (Exception): The exception that was raised.
        """
        pass
