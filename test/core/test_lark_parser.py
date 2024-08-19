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

from nemo_run.core.lark_parser import parse_args


class TestIDParsing:
    def test_single_dollar_id(self):
        args = ["some_path=$some_id"]
        parsed_args, parsed_overrides = parse_args(args)
        assert parsed_args == {"some_path": "some_id"}
        assert parsed_overrides == {}

    def test_braced_dollar_id(self):
        args = ["some_path=${some_id}"]
        parsed_args, parsed_overrides = parse_args(args)
        assert parsed_args == {"some_path": "some_id"}
        assert parsed_overrides == {}

    def test_mixed_ids(self):
        args = ["path1=$id1", "path.to.id2=${id2}"]
        parsed_args, parsed_overrides = parse_args(args)
        assert parsed_args == {"path1": "id1"}
        assert parsed_overrides == {"path.to.id2": "id2"}


class TestPathParsing:
    def test_simple_path(self):
        args = ["arg=to/some/path"]
        parsed_args, parsed_overrides = parse_args(args)
        assert parsed_args == {"arg": "to/some/path"}
        assert parsed_overrides == {}

    def test_path_with_extension(self):
        args = ["arg=hf://to/some/path.txt"]
        parsed_args, parsed_overrides = parse_args(args)
        assert parsed_args == {"arg": "hf://to/some/path.txt"}
        assert parsed_overrides == {}

    def test_path_with_overwrite(self):
        args = ["arg=to/some/path", "overwrite.path=to/another/path"]
        parsed_args, parsed_overrides = parse_args(args)
        assert parsed_args == {"arg": "to/some/path"}
        assert parsed_overrides == {"overwrite.path": "to/another/path"}

    def test_path_with_mixed_keys(self):
        args = [
            "arg1=to/some/path1",
            "arg2=to/some/path2",
            "overwrite.arg3=to/some/path3",
        ]
        parsed_args, parsed_overrides = parse_args(args)
        assert parsed_args == {"arg1": "to/some/path1", "arg2": "to/some/path2"}
        assert parsed_overrides == {"overwrite.arg3": "to/some/path3"}

    def test_path_with_empty_string(self):
        args = ["arg=''"]
        parsed_args, parsed_overrides = parse_args(args)
        assert parsed_args == {"arg": ""}
        assert parsed_overrides == {}

    def test_path_with_quoted_string(self):
        args = ['arg="to/some/path"']
        parsed_args, parsed_overrides = parse_args(args)
        assert parsed_args == {"arg": "to/some/path"}
        assert parsed_overrides == {}

    def test_path_with_multiple_slashes(self):
        args = ["arg=to/some//path"]
        parsed_args, parsed_overrides = parse_args(args)
        assert parsed_args == {"arg": "to/some//path"}
        assert parsed_overrides == {}


class TestIntParsing:
    def test_positive_int(self):
        parsed_args, parsed_overrides = parse_args(["key=123", "key.subkey=456"])
        assert parsed_args == {"key": 123}
        assert parsed_overrides == {"key.subkey": 456}

    def test_negative_int(self):
        parsed_args, parsed_overrides = parse_args(["key=-456", "key.subkey=-789"])
        assert parsed_args == {"key": -456}
        assert parsed_overrides == {"key.subkey": -789}

    def test_zero_int(self):
        parsed_args, parsed_overrides = parse_args(["key=0", "key.subkey=1"])
        assert parsed_args == {"key": 0}
        assert parsed_overrides == {"key.subkey": 1}

    def test_int_with_underscore(self):
        parsed_args, parsed_overrides = parse_args(["key=1_000", "key.subkey=2_000"])
        assert parsed_args == {"key": 1000}
        assert parsed_overrides == {"key.subkey": 2000}


class TestFloatParsing:
    def test_positive_float(self):
        parsed_args, parsed_overrides = parse_args(["key=123.456", "key.subkey=456.789"])
        assert parsed_args == {"key": 123.456}
        assert parsed_overrides == {"key.subkey": 456.789}

    def test_negative_float(self):
        parsed_args, parsed_overrides = parse_args(["key=-123.456", "key.subkey=-456.789"])
        assert parsed_args == {"key": -123.456}
        assert parsed_overrides == {"key.subkey": -456.789}

    def test_zero_float(self):
        parsed_args, parsed_overrides = parse_args(["key=0.0", "key.subkey=1.0"])
        assert parsed_args == {"key": 0.0}
        assert parsed_overrides == {"key.subkey": 1.0}

    def test_float_with_exponent(self):
        parsed_args, parsed_overrides = parse_args(["key=1.23e4", "key.subkey=5.67e-8"])
        assert parsed_args == {"key": 12300.0}
        assert parsed_overrides == {"key.subkey": 5.67e-08}

    def test_float_with_underscore(self):
        parsed_args, parsed_overrides = parse_args(["key=1_234.567", "key.subkey=8_910.1112"])
        assert parsed_args == {"key": 1234.567}
        assert parsed_overrides == {"key.subkey": 8910.1112}


class TestListValueParsing:
    def test_simple_list(self):
        parsed_args, parsed_overrides = parse_args(["key=[1, 2, 3]"])
        assert parsed_args == {"key": [1, 2, 3]}
        assert parsed_overrides == {}

    def test_nested_list(self):
        parsed_args, parsed_overrides = parse_args(["key=[[1, 2], [3, 4]]"])
        assert parsed_args == {"key": [[1, 2], [3, 4]]}
        assert parsed_overrides == {}

    def test_list_with_mixed_types(self):
        parsed_args, parsed_overrides = parse_args(["key=[1, 'two', 3.0, true]"])
        assert parsed_args == {"key": [1, "two", 3.0, True]}
        assert parsed_overrides == {}

    def test_list_with_overrides(self):
        parsed_args, parsed_overrides = parse_args(["key=[1, 2, 3]", "key.subkey=[4, 5, 6]"])
        assert parsed_args == {"key": [1, 2, 3]}
        assert parsed_overrides == {"key.subkey": [4, 5, 6]}

    def test_empty_list(self):
        parsed_args, parsed_overrides = parse_args(["key=[]", "key.subkey=[1]"])
        assert parsed_args == {"key": []}
        assert parsed_overrides == {"key.subkey": [1]}


class TestDictValueParsing:
    def test_simple_dict(self):
        parsed_args, parsed_overrides = parse_args(["key={a: 1, b: 2}"])
        assert parsed_args == {"key": {"a": 1, "b": 2}}
        assert parsed_overrides == {}

    def test_nested_dict(self):
        parsed_args, parsed_overrides = parse_args(["key={a: {b: 2}}"])
        assert parsed_args == {"key": {"a": {"b": 2}}}
        assert parsed_overrides == {}

    def test_dict_with_mixed_types(self):
        parsed_args, parsed_overrides = parse_args(["key={a: 1, b: 'two', c: 3.0, d: true}"])
        assert parsed_args == {"key": {"a": 1, "b": "two", "c": 3.0, "d": True}}
        assert parsed_overrides == {}

    def test_dict_with_overrides(self):
        parsed_args, parsed_overrides = parse_args(["key={a: 1, b: 2}", "key.subkey={c: 3, d: 4}"])
        assert parsed_args == {"key": {"a": 1, "b": 2}}
        assert parsed_overrides == {"key.subkey": {"c": 3, "d": 4}}

    def test_empty_dict(self):
        parsed_args, parsed_overrides = parse_args(["key={}", "key.subkey={a: 1}"])
        assert parsed_args == {"key": {}}
        assert parsed_overrides == {"key.subkey": {"a": 1}}


def test_time_parsing():
    args = ["time=00:30:00", "sub.key.time=00:30:00"]
    parsed_args, parsed_overrides = parse_args(args)
    assert parsed_args == {"time": "00:30:00"}
    assert parsed_overrides == {"sub.key.time": "00:30:00"}


def test_null_parsing():
    args = ["value=null", "value=NULL", "value=NoNE", "value=None"]
    parsed_args, _ = parse_args(args)
    assert all(map(lambda x: x is None, parsed_args.values()))
