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

import sys
from pathlib import Path
from test.dummy_factory import DummyModel
from typing import Any, Dict, List, Literal, Optional, Type, Union

import pytest

from nemo_run.cli.cli_parser import (
    ArgumentParsingError,
    ArgumentValueError,
    DictParseError,
    ListParseError,
    LiteralParseError,
    OperationError,
    ParseError,
    PythonicParser,
    TypeParser,
    UndefinedVariableError,
    UnknownTypeError,
    parse_cli_args,
    parse_value,
)
from nemo_run.config import Config, Partial


class TestSimpleValueParsing:
    def test_int_parsing(self):
        def func(a: int):
            pass

        assert parse_cli_args(func, ["a=123"]).a == 123
        assert parse_cli_args(func, ["a=-456"]).a == -456
        assert parse_cli_args(func, ["a=0"]).a == 0

    def test_float_parsing(self):
        def func(a: float):
            pass

        assert parse_cli_args(func, ["a=123.45"]).a == 123.45
        assert parse_cli_args(func, ["a=-67.89"]).a == -67.89
        assert parse_cli_args(func, ["a=1e-3"]).a == 0.001

    def test_string_parsing(self):
        def func(a: str):
            pass

        assert parse_cli_args(func, ["a='hello'"]).a == "hello"
        assert parse_cli_args(func, ['a="world"']).a == "world"
        assert parse_cli_args(func, ["a=unquoted"]).a == "unquoted"

    def test_bool_parsing(self):
        def func(a: bool):
            pass

        assert parse_cli_args(func, ["a=true"]).a is True
        assert parse_cli_args(func, ["a=false"]).a is False
        assert parse_cli_args(func, ["a=True"]).a is True
        assert parse_cli_args(func, ["a=False"]).a is False
        assert parse_cli_args(func, ["a=1"]).a is True
        assert parse_cli_args(func, ["a=0"]).a is False

    def test_none_parsing(self):
        def func(a: Any):
            pass

        assert parse_cli_args(func, ["a=None"]).a is None
        assert parse_cli_args(func, ["a=null"]).a is None

    def test_path_parsing(self):
        def func(a: Path):
            pass

        assert parse_cli_args(func, ["a=/home/user/file.txt"]).a == Path("/home/user/file.txt")
        assert parse_cli_args(func, ["a=./relative/path"]).a == Path("./relative/path")
        assert parse_cli_args(func, ["a=C:\\Windows\\System32"]).a == Path("C:\\Windows\\System32")

        # Test with a path containing spaces
        assert parse_cli_args(func, ["a=path with spaces"]).a == Path("path with spaces")

        # Test with a path containing special characters
        assert parse_cli_args(func, ["a=path/with/!@#$%^&*()"]).a == Path("path/with/!@#$%^&*()")


class TestComplexTypeParsing:
    def test_list_parsing(self):
        def func(a: List[int]):
            pass

        assert parse_cli_args(func, ["a=[1, 2, 3]"]).a == [1, 2, 3]
        assert parse_cli_args(func, ["a=[]"]).a == []

    def test_nested_list_parsing(self):
        def func(a: List[List[int]]):
            pass

        assert parse_cli_args(func, ["a=[[1, 2], [3, 4]]"]).a == [[1, 2], [3, 4]]

    def test_dict_parsing(self):
        def func(a: Dict[str, int]):
            pass

        assert parse_cli_args(func, ["a={'x': 1, 'y': 2}"]).a == {"x": 1, "y": 2}
        assert parse_cli_args(func, ["a={}"]).a == {}

    def test_nested_dict_parsing(self):
        def func(a: Dict[str, Dict[str, int]]):
            pass

        assert parse_cli_args(func, ["a={'outer': {'inner': 42}}"]).a == {"outer": {"inner": 42}}

    def test_union_type_parsing(self):
        def func(a: Union[int, str]):
            pass

        assert parse_cli_args(func, ["a=123"]).a == 123
        assert parse_cli_args(func, ["a='string'"]).a == "string"

    def test_literal_type_parsing(self):
        def func(a: Literal["red", "green", "blue"]):
            pass

        assert parse_cli_args(func, ["a=red"]).a == "red"
        assert parse_cli_args(func, ["a='green'"]).a == "green"
        assert parse_cli_args(func, ['a="blue"']).a == "blue"

        with pytest.raises(LiteralParseError) as exc_info:
            parse_cli_args(func, ["a=yellow"])
        assert "Error parsing argument" in str(exc_info.value)
        assert "Expected one of ('red', 'green', 'blue'), got 'yellow'" in str(exc_info.value)

        with pytest.raises(LiteralParseError) as exc_info:
            parse_cli_args(func, ["a='yellow'"])
        assert "Error parsing argument" in str(exc_info.value)
        assert "Expected one of ('red', 'green', 'blue'), got 'yellow'" in str(exc_info.value)

        with pytest.raises(LiteralParseError) as exc_info:
            parse_cli_args(func, ['a="yellow"'])
        assert "Error parsing argument" in str(exc_info.value)
        assert "Expected one of ('red', 'green', 'blue'), got 'yellow'" in str(exc_info.value)


class TestFactoryFunctionParsing:
    def test_simple_factory_function(self):
        def func(model: DummyModel):
            pass

        result = parse_cli_args(func, ["model=dummy_model_config"])
        assert isinstance(result.model, Config)
        assert result.model.hidden == 2000
        assert result.model.activation == "tanh"

    def test_factory_function_with_args(self):
        def func(model: DummyModel):
            pass

        result = parse_cli_args(func, ["model=my_dummy_model(1000)"])
        assert isinstance(result.model, Config)
        assert result.model.hidden == 1000
        assert result.model.activation == "tanh"

    def test_factory_function_with_list(self):
        def func(model: List[DummyModel]):
            pass

        result = parse_cli_args(
            func,
            [
                "model=[my_dummy_model(1000), my_dummy_model(2000)]",
                "model[0].hidden=5000",
            ],
        )
        assert isinstance(result.model, list)
        assert len(result.model) == 2
        assert result.model[0].hidden == 5000
        assert result.model[1].hidden == 2000
        assert result.model[0].activation == "tanh"
        assert result.model[1].activation == "tanh"

    def test_factory_function_with_kwargs(self):
        def func(model: DummyModel):
            pass

        result = parse_cli_args(func, ["model=my_dummy_model(hidden=3000)"])
        assert isinstance(result.model, Config)
        assert result.model.hidden == 3000
        assert result.model.activation == "tanh"

    def test_with_overwrites(self):
        def func(model: DummyModel):
            pass

        result = parse_cli_args(func, ["model=dummy_model_config", "model.hidden=3"])
        assert isinstance(result.model, Config)
        assert result.model.hidden == 3
        assert result.model.activation == "tanh"


class TestFactoryLoading:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        # Setup: Add test functions to __main__
        def test_function():
            return Config(DummyModel, hidden=1200, activation="tanh")

        def test_function_with_args(hidden=None, activation=None):
            return Config(DummyModel, hidden=hidden, activation=activation)

        sys.modules["__main__"].test_function = test_function
        sys.modules["__main__"].test_function_with_args = test_function_with_args

        yield

        # Teardown: Remove test functions from __main__
        del sys.modules["__main__"].test_function
        del sys.modules["__main__"].test_function_with_args

    def test_simple_factory_loading(self):
        def func(model: DummyModel):
            pass

        result = parse_cli_args(func, ["model=dummy_model_config"])
        assert isinstance(result.model, Config)
        assert result.model.hidden == 2000
        assert result.model.activation == "tanh"

    def test_factory_with_args(self):
        def func(model: DummyModel):
            pass

        result = parse_cli_args(func, ["model=my_dummy_model(hidden=3000)"])
        assert isinstance(result.model, Config)
        assert result.model.hidden == 3000
        assert result.model.activation == "tanh"

    def test_from_main_module(self, setup_and_teardown):
        def func(model: DummyModel):
            pass

        result = parse_cli_args(func, ["model=test_function"])
        assert isinstance(result.model, Config)
        assert result.model.hidden == 1200
        assert result.model.activation == "tanh"

    def test_args_from_main_module(self, setup_and_teardown):
        def func(model: DummyModel):
            pass

        result = parse_cli_args(
            func, ["model=test_function_with_args(hidden=10, activation='relu')"]
        )
        assert isinstance(result.model, Config)
        assert result.model.hidden == 10
        assert result.model.activation == "relu"

    def test_dotted_import_factory(self):
        def func(model: DummyModel):
            pass

        result = parse_cli_args(func, ["model=test.dummy_factory.my_dummy_model"])
        assert isinstance(result.model, Config)
        assert result.model.hidden == 2000
        assert result.model.activation == "tanh"


class TestOperations:
    def test_addition(self):
        def func(a: int, b: List[int], c: Dict[str, int]):
            pass

        result = parse_cli_args(
            func, ["a=5", "a+=3", "b=[1, 2]", "b+=[3, 4]", "c={'x': 1}", "c|={'y': 2}"]
        )
        assert result.a == 8
        assert result.b == [1, 2, 3, 4]
        assert result.c == {"x": 1, "y": 2}

    def test_subtraction(self):
        def func(a: int):
            pass

        assert parse_cli_args(func, ["a=10", "a-=4"]).a == 6

    def test_multiplication(self):
        def func(a: int):
            pass

        assert parse_cli_args(func, ["a=3", "a*=2"]).a == 6

    def test_division(self):
        def func(a: float):
            pass

        assert parse_cli_args(func, ["a=10", "a/=2"]).a == 5.0

    def test_string_concatenation(self):
        def func(a: str):
            pass

        assert parse_cli_args(func, ["a='hello'", "a+=' world'"]).a == "hello world"


class TestExceptions:
    def test_undefined_variable_operation(self):
        def func(a: int):
            pass

        with pytest.raises(UndefinedVariableError, match="Cannot use '\\+=' on undefined variable"):
            parse_cli_args(func, ["a+=3"])

    def test_type_mismatch_addition(self):
        def func(a: List[int]):
            pass

        with pytest.raises(ListParseError, match="Failed to parse '3' as typing.List"):
            parse_cli_args(func, ["a=[1, 2]", "a+=3"])

    def test_type_mismatch_subtraction(self):
        def func(a: int):
            pass

        with pytest.raises(ParseError, match="Failed to parse ''2'' as <class 'int'>"):
            parse_cli_args(func, ["a=5", "a-='2'"])

    def test_division_by_zero(self):
        def func(a: float):
            pass

        with pytest.raises(OperationError, match="Operation '/=' failed: float division by zero "):
            parse_cli_args(func, ["a=10", "a/=0"])

    def test_invalid_key(self):
        def func(a: int):
            pass

        with pytest.raises(
            ArgumentValueError, match="Invalid argument: No parameter named 'b' exists for"
        ):
            parse_cli_args(func, ["b=5"])

    def test_invalid_operation(self):
        def func(a: int):
            pass

        with pytest.raises(ArgumentParsingError, match="Invalid argument format "):
            parse_cli_args(func, ["a=5", "a%=2"])

    def test_type_conversion_error(self):
        def func(a: int):
            pass

        with pytest.raises(ParseError, match="Invalid integer literal"):
            parse_cli_args(func, ["a=3.14"])

    def test_invalid_list_format(self):
        def func(a: List[int]):
            pass

        with pytest.raises(ListParseError, match="Invalid list: .*"):
            parse_cli_args(func, ["a=[1, 2, 3,.]"])

    def test_invalid_dict_format(self):
        def func(a: Dict[str, int]):
            pass

        with pytest.raises(DictParseError, match="Invalid dict: .*"):
            parse_cli_args(func, ["a={'key': 1, 'key2': 2,.}"])

    def test_invalid_literal(self):
        def func(a: Literal["red", "green", "blue"]):
            pass

        with pytest.raises(
            LiteralParseError,
            match="Invalid value for Literal type. Expected one of \\('red', 'green', 'blue'\\), got 'yellow'",
        ):
            parse_cli_args(func, ["a='yellow'"])


class TestParseValue:
    def test_parse_int(self):
        assert parse_value("123", int) == 123
        assert parse_value("-456", int) == -456
        assert parse_value("0", int) == 0
        assert parse_value("+789", int) == 789
        with pytest.raises(
            ParseError, match="Failed to parse '3.14' as <class 'int'>: Invalid integer literal"
        ):
            parse_value("3.14", int)
        with pytest.raises(
            ParseError,
            match="Failed to parse 'not_an_int' as <class 'int'>: Invalid integer literal",
        ):
            parse_value("not_an_int", int)

    def test_parse_float(self):
        assert parse_value("3.14", float) == 3.14
        assert parse_value("-2.5", float) == -2.5
        assert parse_value("1e-3", float) == 0.001
        assert parse_value("0.0", float) == 0.0
        assert parse_value("-0.0", float) == -0.0
        assert parse_value("inf", float) == float("inf")
        assert parse_value("-inf", float) == float("-inf")
        with pytest.raises(
            ParseError,
            match="Failed to parse 'not_a_float' as <class 'float'>: Could not convert string to float",
        ):
            parse_value("not_a_float", float)

    def test_parse_str(self):
        assert parse_value("hello", str) == "hello"
        assert parse_value("123", str) == "123"
        assert parse_value("", str) == ""
        assert parse_value(" ", str) == " "
        assert parse_value("True", str) == "True"

    def test_parse_bool(self):
        assert parse_value("true", bool) is True
        assert parse_value("True", bool) is True
        assert parse_value("TRUE", bool) is True
        assert parse_value("false", bool) is False
        assert parse_value("False", bool) is False
        assert parse_value("FALSE", bool) is False
        assert parse_value("1", bool) is True
        assert parse_value("0", bool) is False
        assert parse_value("yes", bool) is True
        assert parse_value("no", bool) is False
        assert parse_value("on", bool) is True
        assert parse_value("off", bool) is False
        with pytest.raises(
            ParseError,
            match="Failed to parse 'not_a_bool' as <class 'bool'>: Cannot convert .* to bool",
        ):
            parse_value("not_a_bool", bool)
        with pytest.raises(
            ParseError, match="Failed to parse '2' as <class 'bool'>: Cannot convert .* to bool"
        ):
            parse_value("2", bool)

    def test_parse_list(self):
        assert parse_value("[1, 2, 3]", List[int]) == [1, 2, 3]
        assert parse_value('["a", "b", "c"]', List[str]) == ["a", "b", "c"]
        assert parse_value("[1.1, 2.2, 3.3]", List[float]) == [1.1, 2.2, 3.3]
        assert parse_value("[]", List[Any]) == []
        with pytest.raises(ParseError, match="Failed to parse 'not_a_list' as typing.List"):
            parse_value("not_a_list", List[int])
        with pytest.raises(
            ParseError, match="Failed to parse '\\[1, 2, 'three'\\]' as typing.List"
        ):
            parse_value("[1, 2, 'three']", List[int])

    def test_parse_dict(self):
        assert parse_value('{"a": 1, "b": 2}', Dict[str, int]) == {"a": 1, "b": 2}
        assert parse_value('{"x": "foo", "y": "bar"}', Dict[str, str]) == {"x": "foo", "y": "bar"}
        assert parse_value("{}", Dict[str, Any]) == {}
        with pytest.raises(ParseError, match="Failed to parse 'not_a_dict' as typing.Dict"):
            parse_value("not_a_dict", Dict[str, int])
        with pytest.raises(ParseError, match="Failed to parse"):
            parse_value('{"a": 1, "b": "two"}', Dict[str, int])

    def test_parse_union(self):
        assert parse_value("123", Union[int, str]) == 123
        assert parse_value("hello", Union[int, str]) == "hello"
        assert parse_value("3.14", Union[int, float]) == 3.14
        with pytest.raises(
            ParseError,
            match="Failed to parse 'true' as typing.Union\\[int, float\\]: No matching type in Union.",
        ):
            parse_value("true", Union[int, float])

    def test_parse_optional(self):
        assert parse_value("123", Optional[int]) == 123
        assert parse_value("None", Optional[int]) is None
        assert parse_value("null", Optional[int]) is None
        assert parse_value("hello", Optional[str]) == "hello"
        assert parse_value("None", Optional[str]) is None
        assert parse_value("null", Optional[str]) is None
        with pytest.raises(
            ParseError,
            match="Failed to parse 'not_an_int' as typing.Optional\\[int\\]: No matching type in Union. Errors: Failed to parse 'not_an_int' as <class 'int'>: Invalid integer literal",
        ):
            parse_value("not_an_int", Optional[int])

    def test_parse_literal(self):
        Color = Literal["red", "green", "blue"]
        assert parse_value("red", Color) == "red"
        assert parse_value("green", Color) == "green"
        assert parse_value("blue", Color) == "blue"
        with pytest.raises(
            LiteralParseError,
            match="Failed to parse 'yellow' as typing.Literal: Invalid value for Literal type. Expected one of \\('red', 'green', 'blue'\\), got 'yellow'",
        ):
            parse_value("yellow", Color)

    def test_parse_nested_types(self):
        assert parse_value("[[1, 2], [3, 4]]", List[List[int]]) == [[1, 2], [3, 4]]
        assert parse_value('{"a": [1, 2], "b": [3, 4]}', Dict[str, List[int]]) == {
            "a": [1, 2],
            "b": [3, 4],
        }
        with pytest.raises(
            ListParseError,
            match="Failed to parse '\\[1, 2, 3\\]' as typing.List: Invalid list: Failed to parse '1' as typing.List: Invalid list: Not a list",
        ):
            parse_value("[1, 2, 3]", List[List[int]])

    def test_parse_unknown_type(self):
        class UnknownType:
            pass

        with pytest.raises(
            UnknownTypeError,
            match="Failed to parse 'value' as <class '.*UnknownType'>: Unsupported type",
        ):
            parse_value("value", UnknownType)

    def test_type_inference(self):
        assert parse_value("123") == 123
        assert parse_value("3.14") == 3.14
        assert parse_value("true") == "true"  # Note: inferred as str, not bool

    def test_custom_parser(self):
        type_parser = TypeParser()

        @type_parser.register_parser(complex)
        def parse_complex(value: str, _: Type) -> complex:
            try:
                return complex(value)
            except ValueError:
                raise ParseError(value, complex, "Invalid complex number")

        assert type_parser.parse("1+2j", complex) == 1 + 2j
        assert type_parser.parse("-3-4j", complex) == -3 - 4j
        with pytest.raises(
            ParseError,
            match="Failed to parse 'not_a_complex' as <class 'complex'>: Invalid complex number",
        ):
            type_parser.parse("not_a_complex", complex)

    def test_strict_mode(self):
        strict_parser = TypeParser(strict_mode=True)
        lenient_parser = TypeParser(strict_mode=False)

        class CustomType:
            pass

        with pytest.raises(
            ParseError, match="Failed to parse 'value' as <class '.*CustomType'>: Unsupported type"
        ):
            strict_parser.parse("value", CustomType)

        assert lenient_parser.parse("value", CustomType) == "value"

    def test_caching(self):
        # This test is a bit tricky to write because caching is an implementation detail.
        # We can test that repeated calls with the same arguments return the same result quickly.
        import time

        start = time.time()
        for _ in range(1000):
            parse_value("123", int)
        end = time.time()
        assert end - start < 0.1  # This should be very fast due to caching


class TestPythonicParser:
    @pytest.fixture
    def parser(self):
        return PythonicParser()

    def test_parse_value(self, parser):
        assert parser.parse_value("42") == 42
        assert parser.parse_value("3.14") == 3.14
        assert parser.parse_value("true") is True
        assert parser.parse_value("false") is False
        assert parser.parse_value("None") is None
        assert parser.parse_value("[1, 2, 3]") == [1, 2, 3]
        assert parser.parse_value("{'a': 1, 'b': 2}") == {"a": 1, "b": 2}

    def test_parse_constructor(self, parser):
        assert parser.parse_constructor("dict(x=1, y=2)") == {"x": 1, "y": 2}
        assert parser.parse_constructor("list(1, 2, 3)") == [1, 2, 3]
        assert parser.parse_constructor("tuple(1, 2, 3)") == (1, 2, 3)
        assert parser.parse_constructor("set(1, 2, 3)") == {1, 2, 3}

    def test_parse_comprehension(self, parser):
        assert parser.parse_comprehension("[x for x in range(3)]") == [0, 1, 2]
        assert parser.parse_comprehension("{x: x**2 for x in range(3)}") == {0: 0, 1: 1, 2: 4}

    def test_parse_lambda(self, parser):
        # Test safe lambdas
        lambda_func = parser.parse_lambda("lambda x: x * 2")
        assert lambda_func(5) == 10

        lambda_func = parser.parse_lambda("lambda x, y: x + y")
        assert lambda_func(2, 3) == 5

        lambda_func = parser.parse_lambda("lambda x: x ** 2 + 2*x - 1")
        assert lambda_func(3) == 14

        # Test lambdas with allowed built-ins
        lambda_func = parser.parse_lambda("lambda x: abs(x)")
        assert lambda_func(-5) == 5

        lambda_func = parser.parse_lambda("lambda x, y: max(x, y)")
        assert lambda_func(3, 7) == 7

        # Test potentially unsafe lambdas
        with pytest.raises(ArgumentValueError):
            parser.parse_lambda("lambda: __import__('os').system('echo hacked')")

        with pytest.raises(ArgumentValueError):
            parser.parse_lambda(
                "lambda x: globals()['__builtins__']['eval']('__import__(\"os\").system(\"echo hacked\")')"
            )

        with pytest.raises(ArgumentValueError):
            parser.parse_lambda("lambda: open('/etc/passwd').read()")

        with pytest.raises(ArgumentValueError):
            parser.parse_lambda("lambda x: getattr(x, 'dangerous_method')()")

        # Test lambda with disallowed built-in functions
        with pytest.raises(ArgumentValueError):
            parser.parse_lambda("lambda x: __import__('os')")

        # Test lambda with disallowed attribute access
        with pytest.raises(ArgumentValueError):
            parser.parse_lambda("lambda x: x.__dict__")


class TestEdgeCases:
    def test_empty_input(self):
        def func():
            pass

        assert parse_cli_args(func, []) == Partial(func)

    def test_multiple_assignments(self):
        def func(a: int, b: int):
            pass

        result = parse_cli_args(func, ["a=1", "b=2", "a=3"])
        assert result.a == 3
        assert result.b == 2

    def test_complex_nested_structures(self):
        def func(a: List[Dict[str, Union[int, List[str]]]]):
            pass

        result = parse_cli_args(func, ["a=[{'x': 1, 'y': ['a', 'b']}, {'z': 2}]"])
        assert result.a == [{"x": 1, "y": ["a", "b"]}, {"z": 2}]
