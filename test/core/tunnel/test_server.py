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
from pathlib import Path
from unittest.mock import MagicMock, patch

from nemo_run.core.tunnel import server


def test_server_dir():
    job_dir = "/tmp/nemo_run_tests"
    name = "test_tunnel"
    expected_path = Path(job_dir) / ".nemo_run" / ".tunnels" / name
    assert server.server_dir(job_dir, name) == expected_path


@patch("socket.socket")
def test_launch(mock_socket, tmpdir):
    path = Path(tmpdir)
    workspace_name = "test_workspace"
    hostname = "test_hostname"

    os.environ["USER"] = "dummy"
    mock_socket_obj = MagicMock()
    mock_socket.return_value = mock_socket_obj
    mock_socket_obj.getsockname.return_value = ("localhost", 1234)
    mock_context = MagicMock()
    with patch("nemo_run.core.tunnel.server.Context", return_value=mock_context):
        server.launch(path, workspace_name, hostname=hostname)

    mock_context.run.assert_any_call(
        'echo "PubkeyAuthentication yes" >> /etc/ssh/sshd_config.d/custom.conf'
    )
    mock_context.run.assert_any_call("/usr/sbin/sshd -D -p 1234", pty=True, hide=True)

    metadata = server.TunnelMetadata.restore(path)
    assert metadata.user == os.environ["USER"]
    assert metadata.hostname == hostname
    assert metadata.workspace_name == workspace_name


def test_tunnel_metadata_save_restore(tmpdir):
    path = Path(tmpdir)
    metadata = server.TunnelMetadata(
        user="test_user",
        hostname="test_hostname",
        port=1234,
        workspace_name="test_workspace",
    )
    metadata.save(path)

    restored_metadata = server.TunnelMetadata.restore(path)
    assert restored_metadata == metadata
