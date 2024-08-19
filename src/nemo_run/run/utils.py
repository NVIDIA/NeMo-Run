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

import contextlib
import sys
from contextlib import redirect_stderr, redirect_stdout
from typing import Optional, TextIO


class _Tee:
    def __init__(self, filename: str, sysout: TextIO) -> None:
        self.filename: str = filename
        self.file: Optional[TextIO] = None
        self.sysout = sysout

    def write(self, data: str) -> int:
        if self.file is None:
            raise ValueError("File is not open")
        self.file.write(data)
        self.sysout.write(data)
        self.sysout.flush()
        return len(data)

    def flush(self) -> None:
        if self.file is None:
            raise ValueError("File is not open")
        self.file.flush()


class TeeStdoutStderr(contextlib.ExitStack):
    def __init__(self, filename: str) -> None:
        super().__init__()
        self.tee_out: _Tee = _Tee(filename, sysout=sys.stdout)
        self.tee_err: _Tee = _Tee(filename, sysout=sys.stderr)
        self.file: Optional[TextIO] = None

    def __enter__(self) -> "TeeStdoutStderr":
        self.file = open(self.tee_out.filename, "a")
        self.tee_out.file = self.file
        self.tee_err.file = self.file
        self.enter_context(redirect_stdout(self.tee_out))
        self.enter_context(redirect_stderr(self.tee_err))
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> Optional[bool]:
        if self.file:
            self.file.close()
        return super().__exit__(exc_type, exc_value, traceback)
