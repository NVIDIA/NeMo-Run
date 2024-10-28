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
from typing import Iterable

from fabric import Connection

logger = logging.getLogger(__name__)


# This is adapted from https://github.com/fabric/patchwork/blob/master/patchwork/transfers.py
# Cannot use the original library because of ImportError: cannot import name 'six' from 'invoke.vendor'
def rsync(
    c: Connection,
    source: str,
    target: str,
    exclude: str | Iterable[str] = (),
    delete: bool = False,
    strict_host_keys: bool = True,
    rsync_opts: str = "",
    ssh_opts: str = "",
    hide_output: bool = True,
):
    logger.info(f"rsyncing {source} to {target} ...")
    # Turn single-string exclude into a one-item list for consistency
    if isinstance(exclude, str):
        exclude = [exclude]
    # Create --exclude options from exclude list
    exclude_opts = ' --exclude "{}"' * len(exclude)
    # Double-backslash-escape
    exclusions = tuple([str(s).replace('"', '\\\\"') for s in exclude])
    # Honor SSH key(s)
    key_string = ""
    # TODO: seems plausible we need to look in multiple places if there's too
    # much deferred evaluation going on in how we eg source SSH config files
    # and so forth, re: connect_kwargs
    # TODO: we could get VERY fancy here by eg generating a tempfile from any
    # in-memory-only keys...but that's also arguably a security risk, so...
    keys = c.connect_kwargs.get("key_filename", [])
    # TODO: would definitely be nice for Connection/FabricConfig to expose an
    # always-a-list, always-up-to-date-from-all-sources attribute to save us
    # from having to do this sort of thing. (may want to wait for Paramiko auth
    # overhaul tho!)
    if isinstance(keys, str):
        keys = [keys]
    if keys:
        key_string = "-i " + " -i ".join(keys)
    # Get base cxn params
    user, host, port = c.user, c.host, c.port
    port_string = "-p {}".format(port)
    # Remote shell (SSH) options
    rsh_string = ""
    # Strict host key checking
    disable_keys = "-o StrictHostKeyChecking=no"
    if not strict_host_keys and disable_keys not in ssh_opts:
        ssh_opts += " {}".format(disable_keys)
    rsh_parts = [key_string, port_string, ssh_opts]
    if any(rsh_parts):
        rsh_string = "--rsh='ssh {}'".format(" ".join(rsh_parts))
    # Set up options part of string
    options_map = {
        "delete": "--delete" if delete else "",
        "exclude": exclude_opts.format(*exclusions),
        "rsh": rsh_string,
        "extra": rsync_opts,
    }
    options = "{delete}{exclude} -pthrvz {extra} {rsh}".format(**options_map)
    # Create and run final command string
    # TODO: richer host object exposing stuff like .address_is_ipv6 or whatever
    if host.count(":") > 1:
        # Square brackets are mandatory for IPv6 rsync address,
        # even if port number is not specified
        cmd = "rsync {} {} [{}@{}]:{}"
    else:
        cmd = "rsync {} {} {}@{}:{}"
    cmd = cmd.format(options, source, user, host, target)
    c.run(f"mkdir -p {target}", hide=hide_output)
    result = c.local(cmd, hide=hide_output)
    if result:
        logger.info(f"Successfully ran `{result.command}`")
    else:
        raise RuntimeError("rsync failed")
