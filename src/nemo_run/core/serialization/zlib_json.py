# # SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# # SPDX-License-Identifier: Apache-2.0
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# # http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# import base64
# import inspect
# import json
# import os
# import sys
# import types
# import zlib
# from typing import Optional

# from fiddle._src import config
# from fiddle._src.experimental import serialization


# class ZlibJSONSerializer:
#     """Serializer that uses JSON, zlib, and base64 encoding."""

#     def serialize(
#         self,
#         cfg: config.Buildable,
#         pyref_policy: Optional[serialization.PyrefPolicy] = None,
#     ) -> str:
#         _json = serialization.dump_json(cfg, pyref_policy)

#         if "__main__" in sys.modules:
#             main_module = sys.modules['__main__']
#             # Put source-code in the _json under __main__ key
#             main_source = inspect.getsource(main_module)
#             _json_dict = json.loads(_json)
#             _json_dict["__main__"] = main_source
#             _json = json.dumps(_json_dict)

#         return base64.urlsafe_b64encode(zlib.compress(_json.encode())).decode("ascii")

#     def deserialize(
#         self,
#         serialized: str,
#         pyref_policy: Optional[serialization.PyrefPolicy] = None,
#     ) -> config.Buildable:
#         _json = zlib.decompress(base64.urlsafe_b64decode(serialized)).decode()
#         _json_dict = json.loads(_json)

#         # Check if __main__ key is in json
#         if "__main__" in _json_dict:
#             # Load the source-code from the key and put it in sys.modules['__main__']
#             main_source = _json_dict.pop("__main__")
#             main_module = types.ModuleType("__main__")
#             exec(main_source, main_module.__dict__)
#             sys.modules["__main__"] = main_module

#         # Remove the __main__ key from the json
#         _json = json.dumps(_json_dict)

#         return serialization.load_json(_json, pyref_policy=pyref_policy)



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
