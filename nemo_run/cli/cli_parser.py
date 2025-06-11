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

import ast
import functools
import importlib
import inspect
import logging
import operator
import re
import sys
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    ForwardRef,
    FrozenSet,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

import fiddle as fdl

from nemo_run.config import Config, Partial


logger = logging.getLogger(__name__)


BUILTIN_TO_TYPING = {
    list: List,
    dict: Dict,
    tuple: tuple,
    set: set,
    type: Type,
}

# Add the reverse mapping for normalization
TYPING_TO_BUILTIN = {v: k for k, v in BUILTIN_TO_TYPING.items()}


class Operation(Enum):
    ASSIGN = "="
    ADD = "+="
    SUBTRACT = "-="
    MULTIPLY = "*="
    DIVIDE = "/="
    OR = "|="
    AND = "&="
    UNION = "|="


class CLIException(Exception):
    def __init__(self, message: str, arg: str, context: Dict[str, Any]):
        self.arg = arg
        self.context = context
        super().__init__(f"{message} (Argument: {arg}, Context: {context})")

    def user_friendly_message(self) -> str:
        """Return a user-friendly error message."""
        return f"Error processing argument '{self.arg}': {self.args[0]}"


class ArgumentParsingError(CLIException):
    """Raised when there's an error parsing the initial argument structure."""


class TypeParsingError(CLIException):
    """Raised when there's an error parsing the type of an argument."""


class OperationError(CLIException):
    """Raised when there's an error performing an operation on an argument."""


class ArgumentValueError(CLIException):
    """Raised when the value of a CLI argument is invalid."""


class UndefinedVariableError(CLIException):
    """Raised when an operation is attempted on an undefined variable."""


class ParseError(CLIException):
    """Base exception class for parsing errors."""

    def __init__(self, value: str, expected_type: Type, reason: str):
        self.value = value
        self.expected_type = expected_type
        self.reason = reason
        super().__init__(
            f"Failed to parse '{value}' as {expected_type}: {reason}",
            value,
            {"expected_type": expected_type},
        )


class LiteralParseError(ParseError):
    """Raised when parsing a Literal type fails."""


class CollectionParseError(ParseError):
    """Base class for collection parsing errors."""


class ListParseError(CollectionParseError):
    """Raised when parsing a list fails."""


class DictParseError(CollectionParseError):
    """Raised when parsing a dict fails."""


class UnknownTypeError(ParseError):
    """Raised when attempting to parse an unknown or unsupported type."""


def cli_exception_handler(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except CLIException as e:
            logger.error(e.user_friendly_message())
            # You could add custom handling here, such as printing a user-friendly message
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise CLIException("An unexpected error occurred", "", {}) from e

    return wrapper


class PythonicParser:
    """
    A parser for handling Pythonic-style command-line arguments.

    This class provides methods to parse and evaluate various Python-like
    expressions and structures passed as command-line arguments.
    """

    def __init__(self):
        """
        Initialize the PythonicParser with supported operations and safe built-in functions.
        """
        self.operations = {
            Operation.ASSIGN: lambda old, new: new,
            Operation.ADD: operator.iadd,
            Operation.SUBTRACT: operator.isub,
            Operation.MULTIPLY: operator.imul,
            Operation.DIVIDE: operator.itruediv,
            Operation.OR: operator.ior,
            Operation.AND: operator.iand,
        }
        self.safe_builtins = {
            "abs": abs,
            "all": all,
            "any": any,
            "bin": bin,
            "bool": bool,
            "chr": chr,
            "divmod": divmod,
            "enumerate": enumerate,
            "filter": filter,
            "float": float,
            "hex": hex,
            "int": int,
            "isinstance": isinstance,
            "len": len,
            "list": list,
            "map": map,
            "max": max,
            "min": min,
            "oct": oct,
            "ord": ord,
            "pow": pow,
            "range": range,
            "reversed": reversed,
            "round": round,
            "set": set,
            "slice": slice,
            "sorted": sorted,
            "str": str,
            "sum": sum,
            "tuple": tuple,
            "zip": zip,
        }

    def parse(self, arg: str) -> Dict[str, Any]:
        """
        Parse a single command-line argument into a key-operation-value tuple.

        Args:
            arg (str): The command-line argument to parse.

        Returns:
            Dict[str, Any]: A dictionary with a single key-value pair, where the key is the
                            argument name and the value is a tuple of (Operation, value).

        Raises:
            ArgumentParsingError: If the argument format is invalid or the operation is not recognized.

        Example:
            >>> parser = PythonicParser()
            >>> parser.parse("x+=5")
            {'x': (Operation.ADD, '5')}
        """
        assignment_match = re.match(r"^([\w\[\]\.]+)\s*(=|\+=|-=|\*=|/=|\|=|&=)\s*(.+)$", arg)
        if assignment_match:
            key, op_str, value = assignment_match.groups()
            try:
                op = Operation(op_str)
            except ValueError:
                raise ArgumentParsingError(f"Invalid operation: {op_str}", arg, {"key": key, "value": value})
            return {key: (op, value)}
        raise ArgumentParsingError("Invalid argument format", arg, {})

    def parse_value(self, value: str) -> Any:
        """
        Parse a string value into its corresponding Python object.

        This method attempts to evaluate the string as a Python literal or expression.
        It handles various types including booleans, constructors, comprehensions,
        lambda functions, and ternary expressions.

        Args:
            value (str): The string value to parse.

        Returns:
            Any: The parsed Python object.

        Example:
            >>> parser = PythonicParser()
            >>> parser.parse_value("[1, 2, 3]")
            [1, 2, 3]
            >>> parser.parse_value("lambda x: x * 2")
            <function <lambda> at ...>
        """
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            if isinstance(value, str) and value.lower() in ("true", "false"):
                return value.lower() == "true"
            elif value.startswith(("dict(", "list(", "tuple(", "set(")):
                return self.parse_constructor(value)
            elif "[" in value and "]" in value and "for" in value:
                return self.parse_comprehension(value)
            elif value.startswith("lambda"):
                return self.parse_lambda(value)
            elif "if" in value and "else" in value:
                return self.parse_ternary(value)
            return value

    def parse_constructor(self, value: str) -> Any:
        """
        Parse a constructor-like string into its corresponding Python object.

        This method handles dict, list, tuple, and set constructors.

        Args:
            value (str): The constructor string to parse.

        Returns:
            Any: The constructed Python object.

        Raises:
            ArgumentValueError: If the constructor format is invalid.

        Example:
            >>> parser = PythonicParser()
            >>> parser.parse_constructor("dict(x=1, y=2)")
            {'x': 1, 'y': 2}
            >>> parser.parse_constructor("list(1, 2, 3)")
            [1, 2, 3]
        """
        constructor_match = re.match(r"(dict|list|tuple|set)\((.*)\)$", value)
        if constructor_match:
            constructor, args = constructor_match.groups()
            if constructor == "dict":
                pairs = re.findall(r"(\w+)\s*=\s*([^,]+)(?:,|$)", args)
                return {k: self.parse_value(v.strip()) for k, v in pairs}
            else:
                parsed_args = self.parse_constructor_args(args)
                if constructor == "list":
                    return list(parsed_args)
                elif constructor == "tuple":
                    return tuple(parsed_args)
                elif constructor == "set":
                    return set(parsed_args)
        raise ArgumentValueError(f"Invalid constructor: {value}", value, {})

    def parse_constructor_args(self, args: str) -> List[Any]:
        """
        Parse the arguments of a constructor into a list of Python objects.

        This method handles nested structures and ensures proper parsing of
        comma-separated arguments.

        Args:
            args (str): The string containing constructor arguments.

        Returns:
            List[Any]: A list of parsed argument values.

        Example:
            >>> parser = PythonicParser()
            >>> parser.parse_constructor_args("1, 'two', [3, 4]")
            [1, 'two', [3, 4]]
        """
        parsed_args = []
        current_arg = ""
        nesting_level = 0
        for char in args + ",":
            if char == "," and nesting_level == 0:
                if current_arg:
                    parsed_args.append(self.parse_value(current_arg.strip()))
                    current_arg = ""
            else:
                current_arg += char
                if char in "([{":
                    nesting_level += 1
                elif char in ")]}":
                    nesting_level -= 1
        return parsed_args

    def parse_comprehension(self, value: str) -> Any:
        """
        Parse a comprehension expression into its corresponding Python object.

        This method safely evaluates list, dict, and set comprehensions.

        Args:
            value (str): The comprehension string to parse.

        Returns:
            Any: The result of the comprehension.

        Raises:
            ArgumentValueError: If the comprehension is invalid or cannot be safely evaluated.

        Example:
            >>> parser = PythonicParser()
            >>> parser.parse_comprehension("[x for x in range(3)]")
            [0, 1, 2]
        """
        try:
            tree = ast.parse(value, mode="eval")
            if isinstance(tree.body, (ast.ListComp, ast.DictComp, ast.SetComp)):
                return self.eval_ast(tree.body)
            raise ValueError("Not a valid comprehension")
        except Exception as e:
            raise ArgumentValueError(f"Invalid comprehension: {str(e)}", value, {})

    def eval_ast(self, node: ast.AST, context: Dict[str, Any] = None) -> Any:
        """
        Safely evaluate an AST node.

        This method traverses the AST and evaluates it in a restricted environment,
        allowing only safe operations and built-in functions.

        Args:
            node (ast.AST): The AST node to evaluate.
            context (Dict[str, Any], optional): A dictionary of variables for evaluation context.

        Returns:
            Any: The result of evaluating the AST node.

        Raises:
            ValueError: If an unsupported or unsafe operation is encountered.

        Note:
            This method is recursive and handles various AST node types.
        """
        if context is None:
            context = {}

        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Name):
            return context.get(node.id, getattr(__builtins__, node.id, None))
        elif isinstance(node, ast.BinOp):
            left = self.eval_ast(node.left, context)
            right = self.eval_ast(node.right, context)
            op_func = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
                ast.FloorDiv: operator.floordiv,
                ast.Mod: operator.mod,
                ast.Pow: operator.pow,
            }.get(type(node.op))
            if op_func:
                return op_func(left, right)
        elif isinstance(node, ast.Compare):
            left = self.eval_ast(node.left, context)
            for op, right in zip(node.ops, node.comparators):
                right_val = self.eval_ast(right, context)
                op_func = {
                    ast.Eq: operator.eq,
                    ast.NotEq: operator.ne,
                    ast.Lt: operator.lt,
                    ast.LtE: operator.le,
                    ast.Gt: operator.gt,
                    ast.GtE: operator.ge,
                }.get(type(op))
                if op_func:
                    if not op_func(left, right_val):
                        return False
                    left = right_val
                else:
                    raise ValueError(f"Unsupported comparison operator: {type(op)}")
            return True
        elif isinstance(node, ast.Call):
            func = self.eval_ast(node.func, context)
            args = [self.eval_ast(arg, context) for arg in node.args]
            return func(*args)
        elif isinstance(node, (ast.ListComp, ast.DictComp, ast.SetComp)):
            # Implement safe evaluation of comprehensions
            # This is a simplified version and may need more robust implementation
            return eval(compile(ast.Expression(node), "<string>", "eval"), {}, {})
        raise ValueError(f"Unsupported AST node: {type(node)}")

    def parse_lambda(self, value: str) -> Callable:
        """
        Parse a lambda expression into a callable function.

        This method safely evaluates lambda expressions, checking for potentially
        unsafe operations.

        Args:
            value (str): The lambda expression string to parse.

        Returns:
            Callable: The parsed lambda function.

        Raises:
            ArgumentValueError: If the lambda contains unsafe operations or is invalid.

        Example:
            >>> parser = PythonicParser()
            >>> func = parser.parse_lambda("lambda x: x * 2")
            >>> func(5)
            10
        """
        try:
            tree = ast.parse(value, mode="eval")
            if isinstance(tree.body, ast.Lambda):
                if self._contains_unsafe_operations(tree.body):
                    raise ValueError("Unsafe operations detected in lambda")
                return eval(value, {"__builtins__": self.safe_builtins}, {})
            raise ArgumentValueError(f"Invalid lambda: {value}", value, {})
        except Exception as e:
            raise ArgumentValueError(f"Error parsing lambda '{value}': {str(e)}", value, {})

    def _contains_unsafe_operations(self, node: ast.AST) -> bool:
        """
        Check if an AST node contains any unsafe operations.

        This method recursively traverses the AST to detect potentially harmful
        operations like attribute access or unsafe function calls.

        Args:
            node (ast.AST): The AST node to check.

        Returns:
            bool: True if unsafe operations are detected, False otherwise.

        Note:
            This is a helper method used internally by parse_lambda.
        """
        if isinstance(node, ast.Call):
            # Allow calls to safe built-in functions
            if isinstance(node.func, ast.Name) and node.func.id in self.safe_builtins:
                return False
            return True
        elif isinstance(node, ast.Attribute):
            # Prevent attribute access
            return True
        elif isinstance(node, ast.Name):
            # Allow only certain built-in names, parameter names, and safe built-ins
            allowed_names = {"True", "False", "None"}.union(self.safe_builtins.keys())
            return node.id not in allowed_names and not node.id.isidentifier()
        elif isinstance(node, ast.Lambda):
            return self._contains_unsafe_operations(node.body)
        elif isinstance(node, ast.Expression):
            return self._contains_unsafe_operations(node.body)
        elif isinstance(node, ast.BinOp):
            # Allow basic arithmetic operations
            return self._contains_unsafe_operations(node.left) or self._contains_unsafe_operations(node.right)
        elif isinstance(node, ast.UnaryOp):
            return self._contains_unsafe_operations(node.operand)
        elif isinstance(node, (ast.List, ast.Tuple, ast.Set, ast.Dict)):
            return any(self._contains_unsafe_operations(elt) for elt in ast.iter_child_nodes(node))
        elif isinstance(node, (ast.Num, ast.Str, ast.Bytes, ast.NameConstant, ast.Ellipsis)):
            # Allow basic literals
            return False
        return True

    def parse_ternary(self, value: str) -> Any:
        """
        Parse a ternary expression into its corresponding Python object.

        This method safely evaluates ternary (conditional) expressions.

        Args:
            value (str): The ternary expression string to parse.

        Returns:
            Any: The result of evaluating the ternary expression.

        Raises:
            ArgumentValueError: If the ternary expression is invalid or cannot be safely evaluated.

        Example:
            >>> parser = PythonicParser()
            >>> parser.parse_ternary("'yes' if True else 'no'")
            'yes'
        """
        try:
            tree = ast.parse(value, mode="eval")
            if isinstance(tree.body, ast.IfExp):
                return eval(value)
            raise ArgumentValueError(f"Invalid ternary expression: {value}", value, {})
        except Exception as e:
            raise ArgumentValueError(f"Error parsing ternary expression '{value}': {str(e)}", value, {})

    def apply_operation(self, op: Operation, old: Any, new: Any) -> Any:
        """
        Apply the specified operation to the old and new values.

        Args:
            op (Operation): The operation to apply.
            old (Any): The existing value.
            new (Any): The new value to apply the operation with.

        Returns:
            Any: The result of applying the operation.

        Raises:
            OperationError: If the operation fails or is not supported for the given types.
        """
        try:
            if op == Operation.OR and isinstance(old, dict) and isinstance(new, dict):
                return {**old, **new}
            elif op == Operation.OR and hasattr(old, "__dict__") and hasattr(new, "__dict__"):
                return {**old.__dict__, **new.__dict__}
            elif op in self.operations:
                return self.operations[op](old, new)
            else:
                raise ValueError(f"Unsupported operation: {op}")
        except Exception as e:
            raise OperationError(
                f"Operation '{op.value}' failed: {str(e)}",
                f"{old} {op.value} {new}",
                {"old": old, "new": new},
            ) from e


class TypeParser:
    """A parser for converting string values to specified types.

    This class provides methods to parse string values into various Python types,
    including basic types (int, float, str, bool), container types (list, dict),
    and more complex types (Union, Optional, Literal, Path).

    Attributes:
        parsers (Dict[Type, Callable]): A dictionary mapping types to their parsing functions.
        custom_parsers (Dict[Type, Callable]): A dictionary of user-defined custom parsers.
        strict_mode (bool): If True, raises an error for unknown types. If False, returns the original value.

    Example:
        >>> parser = TypeParser()
        >>> parser.parse("123", int)
        123
        >>> parser.parse("[1, 2, 3]", List[int])
        [1, 2, 3]
    """

    __slots__ = ("parsers", "custom_parsers", "strict_mode")

    def __init__(self, strict_mode: bool = True):
        """Initialize the TypeParser.

        Args:
            strict_mode (bool, optional): If True, raises an error for unknown types.
                If False, returns the original value. Defaults to True.
        """
        self.parsers = {
            int: self.parse_int,
            float: self.parse_float,
            str: self.parse_str,
            bool: self.parse_bool,
            list: self.parse_list,
            # List: self.parse_list,
            dict: self.parse_dict,
            # Dict: self.parse_dict,
            Union: self.parse_union,
            Optional: self.parse_optional,
            Literal: self.parse_literal,
            Path: self.parse_path,
            ForwardRef: self.parse_forward_ref,
        }
        self.custom_parsers = {}
        self.strict_mode = strict_mode

    def register_parser(self, type_: Type):
        """Decorator to register a custom parser for a specific type.

        Args:
            type_ (Type): The type for which to register the parser.

        Returns:
            Callable: A decorator function that registers the parser.

        Example:
            @parser.register_parser(CustomType)
            def parse_custom_type(value: str, annotation: Type) -> CustomType:
                # Custom parsing logic here
                pass
        """

        def decorator(func: Callable[[str, Type], Any]):
            self.custom_parsers[type_] = func
            return func

        return decorator

    @lru_cache(maxsize=128)
    def get_parser(self, annotation: Type) -> Callable[[str, Type], Any]:
        """Get the appropriate parser function for the given type annotation.

        Args:
            annotation (Type): The type annotation to find a parser for.

        Returns:
            Callable[[str, Type], Any]: The parser function for the given type.
        """
        # Try to get the origin safely for both Python 3.8 and 3.9+
        try:
            origin = get_origin(annotation)
        except (TypeError, AttributeError):
            origin = None

        # Handle direct type references (int, str, etc.)
        if annotation in self.parsers:
            return self.parsers[annotation]

        # Handle custom parsers
        if annotation in self.custom_parsers:
            return self.custom_parsers[annotation]

        # If we have an origin, map it to the correct parser
        if origin is not None:
            # Map built-in container origins to their corresponding parser
            if origin in (list, List):
                return self.parse_list
            elif origin in (dict, Dict):
                return self.parse_dict
            elif origin is Union:
                return self.parse_union
            # Add other mappings as needed

            # Check for custom parsers for the origin
            if origin in self.custom_parsers:
                return self.custom_parsers[origin]
            if origin in self.parsers:
                return self.parsers[origin]

        # Handle older-style generic aliases
        if hasattr(annotation, "__origin__"):
            origin = annotation.__origin__

            # Map older-style typing module generics
            if origin is list or origin is List:
                return self.parse_list
            elif origin is dict or origin is Dict:
                return self.parse_dict
            elif origin is Union:
                return self.parse_union

            # Check for parsers registered for the origin
            if origin in self.custom_parsers:
                return self.custom_parsers[origin]
            if origin in self.parsers:
                return self.parsers[origin]

        # Fall back to the unknown type parser
        return self.parse_unknown

    def parse(self, value: str, annotation: Type) -> Any:
        """Parse a string value according to the given type annotation.

        Args:
            value (str): The string value to parse.
            annotation (Type): The type annotation to parse the value into.

        Returns:
            Any: The parsed value.

        Raises:
            TypeParsingError: If parsing fails.
        """
        if value.startswith("<Config") or value.startswith("<Partial"):
            value = value[1:-1].replace("()", "")
        if value.startswith("Config") or value.startswith("Partial"):
            return self.parse_buildable(value, annotation)

        parser = self.get_parser(annotation)
        try:
            return parser(value.strip('"'), annotation)
        except ParseError:
            raise
        except Exception as e:
            raise TypeParsingError(
                f"Failed to parse '{value}' as {annotation}: {str(e)}",
                value,
                {"expected_type": annotation},
            )

    def parse_buildable(self, value: str, annotation: Type[Config | Partial]) -> Config | Partial:
        """Parse a string value into a Buildable type (Config or Partial).

        Args:
            value (str): The string value to parse.
            annotation (Type[Config | Partial]): The Buildable type annotation.

        Returns:
            Config | Partial: The parsed Buildable type.
        """
        match = re.match(r"(Config|Partial)\[(.*)\]", value)
        if match:
            buildable_type, target = match.groups()

            # Import the target
            module_name, function_name = target.rsplit(".", 1)
            module = importlib.import_module(module_name)
            config_type = getattr(module, function_name)

            if buildable_type == "Config":
                return Config(config_type)
            elif buildable_type == "Partial":
                return Partial(config_type)

        if str(annotation).startswith("typing.Annotated"):
            args = get_args(annotation)
            if str(args[0]).startswith("typing.Optional") and len(args) > 1:
                cfg_type = get_args(args[0])[0]
                buildable = args[1].__origin__
                if issubclass(buildable, fdl.Buildable):
                    return buildable(cfg_type)

        return Config(annotation)

    def parse_int(self, value: str, _: Type) -> int:
        """Parse a string value into an integer.

        Args:
            value (str): The string value to parse.
            _ (Type): Unused type parameter.

        Returns:
            int: The parsed integer value.

        Raises:
            ParseError: If the value cannot be parsed as an integer.
        """
        try:
            return int(value)
        except ValueError:
            raise ParseError(value, int, "Invalid integer literal")

    def parse_float(self, value: str, _: Type) -> float:
        """Parse a string value into a float.

        Args:
            value (str): The string value to parse.
            _ (Type): Unused type parameter.

        Returns:
            float: The parsed float value.

        Raises:
            ParseError: If the value cannot be parsed as a float.
        """
        try:
            return float(value)
        except ValueError:
            raise ParseError(value, float, f"Could not convert string to float: '{value}'")

    def parse_str(self, value: str, _: Type) -> str:
        """Parse a string value, removing surrounding quotes if present.

        Args:
            value (str): The string value to parse.
            _ (Type): Unused type parameter.

        Returns:
            str: The parsed string value.
        """
        if len(value) >= 2:
            if (value[0] == "'" and value[-1] == "'") or (value[0] == '"' and value[-1] == '"'):
                return value[1:-1]
        return value

    def parse_bool(self, value: str, _: Type) -> bool:
        """Parse a string value into a boolean.

        Args:
            value (str): The string value to parse.
            _ (Type): Unused type parameter.

        Returns:
            bool: The parsed boolean value.

        Raises:
            ParseError: If the value cannot be parsed as a boolean.
        """
        lower_value = value.lower()
        if lower_value in ("true", "yes", "1", "on"):
            return True
        elif lower_value in ("false", "no", "0", "off"):
            return False
        raise ParseError(value, bool, f"Cannot convert '{value}' to bool")

    def parse_list(self, value: str, annotation: Type[List]) -> List:
        """Parse a string value into a list of the specified type.

        Args:
            value (str): The string value to parse.
            annotation (Type[List]): The list type annotation.

        Returns:
            List: The parsed list.

        Raises:
            ListParseError: If the value cannot be parsed as a list.
        """
        try:
            parsed = ast.literal_eval(value)
            if not isinstance(parsed, list):
                raise ValueError("Not a list")
            elem_type = get_args(annotation)[0]
            return [self.parse(str(item), elem_type) for item in parsed]
        except Exception as e:
            raise ListParseError(value, List, f"Invalid list: {str(e)}")

    def parse_dict(self, value: str, annotation: Type[Dict]) -> Dict:
        """Parse a string value into a dictionary of the specified key-value types.

        Args:
            value (str): The string value to parse.
            annotation (Type[Dict]): The dictionary type annotation.

        Returns:
            Dict: The parsed dictionary.

        Raises:
            DictParseError: If the value cannot be parsed as a dictionary.
        """
        try:
            parsed = ast.literal_eval(value)
            if not isinstance(parsed, dict):
                raise ValueError("Not a dict")
            key_type, val_type = get_args(annotation)
            return {self.parse(str(k), key_type): self.parse(str(v), val_type) for k, v in parsed.items()}
        except Exception as e:
            raise DictParseError(value, Dict, f"Invalid dict: {str(e)}")

    def parse_optional(self, value: str, annotation) -> Any:
        """Parse a string value as an Optional type.

        Args:
            value (str): The string value to parse.
            annotation: The Optional type annotation.

        Returns:
            Any: The parsed value or None.
        """
        if value.lower() in ("none", "null"):
            return None
        return self.parse_union(value, annotation)

    def parse_union(self, value: str, annotation) -> Any:
        """Parse a string value as a Union type.

        Args:
            value (str): The string value to parse.
            annotation: The Union type annotation.

        Returns:
            Any: The parsed value.

        Raises:
            ParseError: If the value cannot be parsed as any of the Union types.
        """
        args = get_args(annotation)
        if type(None) in args and value.lower() in ("none", "null"):
            return None
        errors = []
        for arg in args:
            if arg is not type(None):  # Skip NoneType in Union
                try:
                    return self.parse(value, arg)
                except ParseError as e:
                    errors.append(str(e))
        raise ParseError(value, annotation, f"No matching type in Union. Errors: {'; '.join(errors)}")

    def parse_unknown(self, value: str, annotation: Type) -> Any:
        """Parse a string value for an unknown or unsupported type.

        Args:
            value (str): The string value to parse.
            annotation (Type): The type annotation.

        Returns:
            Any: The parsed value or the original string if in non-strict mode.

        Raises:
            UnknownTypeError: If in strict mode and the type is unsupported.
        """
        if annotation == Any:
            return self.parse_any(value, annotation)
        if not self.strict_mode:
            return value
        raise UnknownTypeError(value, annotation, f"Unsupported type: {annotation}")

    def parse_any(self, value: str, _: Type) -> Any:
        """Parse a string value as Any type.

        Args:
            value (str): The string value to parse.
            _ (Type): Unused type parameter.

        Returns:
            Any: The parsed value, attempting to infer the correct type.
        """
        if value.lower() in ("none", "null"):
            return None
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value

    def parse_literal(self, value: str, annotation) -> Any:
        """Parse a string value as a Literal type.

        Args:
            value (str): The string value to parse.
            annotation: The Literal type annotation.

        Returns:
            Any: The parsed Literal value.

        Raises:
            LiteralParseError: If the value is not one of the allowed Literal values.
        """
        literal_values = get_args(annotation)
        if (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')):
            value = value[1:-1]
        if value in literal_values:
            return value
        raise LiteralParseError(
            value,
            Literal,
            f"Invalid value for Literal type. Expected one of {literal_values}, got '{value}'",
        )

    def parse_path(self, value: str, _: Type) -> Path:
        """Parse a string value into a Path object.

        Args:
            value (str): The string value to parse.
            _ (Type): Unused type parameter.

        Returns:
            Path: The parsed Path object.

        Raises:
            ParseError: If the path contains null characters.
        """
        if "\0" in value:
            raise ParseError(value, Path, "Invalid path: contains null character")
        return Path(value.strip("'\" "))

    def parse_forward_ref(self, value: str, annotation) -> Any:
        return value

    def infer_type(self, value: str) -> Type:
        """Infer the type of a string value.

        Args:
            value (str): The string value to infer the type from.

        Returns:
            Type: The inferred type of the value.
        """
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, bool):
                return bool
            return type(parsed)
        except Exception:
            return str


type_parser = TypeParser()


@lru_cache(maxsize=128)
def parse_value(value: str, annotation: Type = None) -> Any:
    if annotation is None:
        annotation = type_parser.infer_type(value)
    return type_parser.parse(value, annotation)


@cli_exception_handler
def parse_cli_args(
    fn: Callable,
    args: List[str],
    output_type: Type[TypeVar("OutputT", Partial, Config)] = Partial,
) -> TypeVar("OutputT", Partial, Config):
    """Parse command-line arguments and apply them to a function or class.

    This is the main public API for parsing command-line arguments in the NeMo Run framework.
    It takes a function or class, a list of command-line arguments, and an output type,
    and returns a configured instance of the output type.

    Args:
        fn (Callable): The function or class to which the arguments will be applied.
        args (List[str]): A list of command-line arguments to parse.
        output_type (Type[TypeVar('OutputT', Partial, Config)], optional): The type of object
            to return. Defaults to Partial.

    Returns:
        TypeVar('OutputT', Partial, Config): An instance of the output_type with the parsed
        arguments applied.

    Raises:
        CLIException: If there's an error during parsing or applying the arguments.

    Example:
        >>> def my_function(x: int, y: str, model: DummyModel):
        ...     pass
        >>> args = ["x=5", "y=hello", "model=dummy_model_config", "model.hidden=3000"]
        >>> result = parse_cli_args(my_function, args)
        >>> print(result)
        Partial(my_function, {'x': 5, 'y': 'hello', 'model': Config(DummyModel, hidden=3000, activation='tanh')})

    Notes:
        - This function supports various argument formats, including positional and keyword arguments.
        - It can handle nested attributes using dot notation (e.g., "config.learning_rate=0.01").
        - The function uses type annotations to correctly parse and cast argument values.
        - Custom parsing logic can be added by registering parsers with the TypeParser class.
        - Nested arguments are supported using dot notation. For example, "model.hidden=3000"
          will set the 'hidden' attribute of the 'model' object.
        - Factory functions can be used to create instances of arbitrary Python types. This is
          achieved using the cli.factory decorator. For example:

          @cli.factory
          def dummy_model_config():
              return Config(DummyModel, hidden=2000, activation="tanh")

          Then, in the command line: "model=dummy_model_config" will create a DummyModel instance.
        - Factory functions can also accept arguments: "model=my_dummy_model(hidden=3000)" will
          call the my_dummy_model factory function with the specified arguments.
        - Operations on nested attributes are supported, e.g., "model.hidden*=3" will multiply
          the 'hidden' attribute of the 'model' object by 3.
    """
    parser = PythonicParser()
    if isinstance(fn, (Config, Partial)):
        output = fn
    elif isinstance(fn, (list, tuple)) and all(isinstance(item, (Config, Partial)) for item in fn):
        output = fn
    else:
        if output_type in (Partial, Config):
            output = output_type(fn)
        else:
            output = output_type

    for arg in _args_to_kwargs(fn, args):
        logger.debug(f"Processing argument: {arg}")
        parsed = parser.parse(arg)
        key, (op, value) = next(iter(parsed.items()))
        logger.debug(f"Parsed key: {key}, op: {op}, value: {value}")

        # Skip the internal _target_ argument used for serialization
        if key == "_target_":
            logger.debug(f"Skipping internal '_target_' argument: {arg}")
            continue

        if "." not in key:
            if isinstance(fn, (Config, Partial)):
                signature = inspect.signature(fn.__fn_or_cls__)
            else:
                try:
                    signature = inspect.signature(fn)
                except Exception:
                    signature = inspect.signature(fn.__class__)
            arg_name, nested = key, output
        else:
            dot_split, nested = key.split("."), output
            for attr in dot_split[:-1]:
                try:
                    nested = parse_attribute(attr, nested)
                except AttributeError as e:
                    raise ArgumentValueError(f"Invalid attribute: {attr}", key, {"nested": nested}) from e

            signature = _signature(nested.__fn_or_cls__)
            # If nested.__fn_or_cls__ is a class and has just *args and **kwargs as parameters,
            # Get signature of the __init__ method
            if len(signature.parameters) == 2 and inspect.isclass(nested.__fn_or_cls__):
                signature = inspect.signature(nested.__fn_or_cls__.__init__)

            arg_name = dot_split[-1]

        param = signature.parameters.get(arg_name)
        if param is None:
            kwargs_param = next(
                (p for p in signature.parameters.values() if p.kind == inspect.Parameter.VAR_KEYWORD),
                None,
            )
            if kwargs_param:
                # Create a generic parameter for the kwargs argument
                param = inspect.Parameter(
                    arg_name,
                    inspect.Parameter.KEYWORD_ONLY,
                    annotation=kwargs_param.annotation if kwargs_param.annotation != inspect.Parameter.empty else Any,
                )
            else:
                raise ArgumentValueError(
                    f"Invalid argument: No parameter named '{arg_name}' exists for {fn}",
                    arg,
                    {"key": key, "value": value},
                )

        annotation, parsed_value = None, None
        if param.annotation != inspect.Parameter.empty:
            annotation = param.annotation
        logger.debug(f"Parsing value {value} as {annotation}")

        annotation = _maybe_resolve_annotation(getattr(nested, "__fn_or_cls__", nested), arg_name, annotation)

        if annotation:
            try:
                parsed_value = parse_factory(fn, arg_name, annotation, value)
            except Exception:
                pass
        if not parsed_value:
            try:
                parsed_value = parse_value(value, annotation)
                logger.debug(f"Parsed value: {parsed_value}")
            except ParseError as e:
                # Preserve the original ParseError subclass
                raise e.__class__(
                    f"Error parsing argument: {str(e)}",
                    arg,
                    {"key": key, "value": value, "expected_type": param.annotation},
                ) from e

        try:
            if op == Operation.ASSIGN:
                setattr(nested, arg_name, parsed_value)
            else:
                if not hasattr(nested, arg_name):
                    raise UndefinedVariableError(
                        f"Cannot use '{op.value}' on undefined variable",
                        arg,
                        {"key": key},
                    )
                setattr(
                    nested,
                    arg_name,
                    parser.apply_operation(op, getattr(nested, arg_name), parsed_value),
                )
        except AttributeError as e:
            raise ArgumentValueError(f"Invalid argument: {str(e)}", arg, {"key": key, "value": value})

    return output


def parse_partial(fn: Callable, *args: str) -> Partial:
    return parse_cli_args(fn, args, output_type=Partial)


def parse_config(fn: Callable, *args: str) -> Config:
    return parse_cli_args(fn, args, output_type=Config)


def parse_factory(parent: Type, arg_name: str, arg_type: Type, value: str) -> Any:
    """Parse a factory-style argument and instantiate the corresponding object(s).

    This function handles single factory calls, lists of factory calls, and dotted imports.
    Configuration files using the @ syntax are handled by load_config_from_path in lazy.py.

    Args:
        parent (Type): The parent class or function where the argument is defined.
        arg_name (str): The name of the argument.
        arg_type (Type): The expected type of the argument.
        value (str): The string value to parse, expected to be in factory format, a list of factory formats,
                     or a dotted import path.

    Returns:
        Any: The instantiated object(s) created by the factory function(s).

    Raises:
        ValueError: If the factory format is invalid or no matching factory is found.
    """
    import catalogue

    from nemo_run.cli.api import extract_constituent_types
    from nemo_run.config import Partial, get_type_namespace

    def _get_from_registry(val, annotation, name):
        if catalogue.check_exists(get_type_namespace(annotation), val):
            return catalogue._get((get_type_namespace(annotation), val))

        namespace = f"{get_type_namespace(annotation)}.{name}"
        if catalogue.check_exists(namespace, val):
            return catalogue._get((namespace, val))

        return catalogue._get((str(annotation), val))

    def parse_single_factory(factory_str):
        # Extract factory name and arguments
        match = re.match(r"^([\w\.]+)(?:\((.*)\))?$", factory_str.strip())
        if not match:
            raise ValueError(f"Invalid factory format: {factory_str}")

        factory_name, args_str = match.groups()
        args_str = args_str or ""

        # Find the factory function
        factory_fn = None

        # Check if it's a dotted import
        if "." in factory_name:
            try:
                module_name, function_name = factory_name.rsplit(".", 1)
                module = importlib.import_module(module_name)
                factory_fn = getattr(module, function_name)

                # Handle constants
                if not callable(factory_fn):
                    return factory_fn
            except (ImportError, AttributeError):
                pass  # If import fails, continue with other parsing methods

        # If not found as dotted import, try registry
        if not factory_fn:
            try:
                factory_fn = _get_from_registry(factory_name, parent, name=arg_name)
            except catalogue.RegistryError:
                types = extract_constituent_types(arg_type)
                for t in types:
                    try:
                        factory_fn = _get_from_registry(factory_name, t, name=factory_name)
                        break
                    except catalogue.RegistryError:
                        continue

        # If not found in registry, check sys.modules["__main__"]
        if not factory_fn:
            main_module = sys.modules.get("__main__")
            if main_module and hasattr(main_module, factory_name):
                factory_fn = getattr(main_module, factory_name)

        if not factory_fn:
            raise ValueError(f"No matching factory found for: {factory_str}")

        if args_str:
            cli_args = [arg.strip() for arg in args_str.split(",") if arg.strip()]
            partial_factory = parse_cli_args(factory_fn, cli_args, output_type=Partial)
            return fdl.build(partial_factory)()

        return factory_fn()

    # Check if the value is a list
    if value.startswith("[") and value.endswith("]"):
        # Check if arg_type is List[T], if so get T
        if get_origin(arg_type) is list:
            arg_type = get_args(arg_type)[0]

        # Parse list with nested structure handling
        inner = value.strip("[] ")
        elements = []
        current = ""
        nesting = 0

        # Parse character by character to handle nested structures
        for char in inner:
            if char == "," and nesting == 0:
                if current.strip():
                    elements.append(current.strip())
                current = ""
            else:
                if char in "([{":
                    nesting += 1
                elif char in ")]}":
                    nesting -= 1
                current += char

        # Add the last element if it exists
        if current.strip():
            elements.append(current.strip())

        return [parse_single_factory(item.strip()) for item in elements]

    return parse_single_factory(value)


def _signature(fn: Callable):
    if fn is dict:
        # Create a signature that accepts **kwargs for dict
        return inspect.Signature([inspect.Parameter("kwargs", inspect.Parameter.VAR_KEYWORD, annotation=Any)])
    return inspect.signature(fn)


def _args_to_kwargs(fn: Callable, args: List[str]) -> List[str]:
    if isinstance(fn, (Config, Partial)):
        signature = inspect.signature(fn.__fn_or_cls__)
    elif isinstance(fn, (list, tuple)):
        signature = None
    else:
        try:
            signature = inspect.signature(fn)
        except Exception:
            signature = inspect.signature(fn.__class__)
    if signature is None:
        for arg in args:
            if "=" not in arg:
                raise ArgumentParsingError(
                    "Positional argument found after keyword argument",
                    arg,
                    {"position": len(args)},
                )

        return args
    params = list(signature.parameters.values())

    updated_args = []
    positional_count = 0
    seen_kwarg = False

    for arg in args:
        if "=" in arg:
            seen_kwarg = True
            updated_args.append(arg)
        else:
            if seen_kwarg:
                raise ArgumentParsingError(
                    "Positional argument found after keyword argument",
                    arg,
                    {"position": len(updated_args)},
                )
            if positional_count < len(params):
                param_name = params[positional_count].name
                updated_args.append(f"{param_name}={arg}")
                positional_count += 1
            else:
                raise ArgumentParsingError(
                    "Too many positional arguments",
                    arg,
                    {"max_positional": len(params)},
                )

    return updated_args


def parse_attribute(attr, nested):
    """Parse and apply attribute access and indexing operations."""
    parts = re.split(r"(\[|\])", attr)
    result = nested

    for part in parts:
        if part == "[" or part == "]" or part == "":
            continue
        elif part.isdigit():
            try:
                result = result[int(part)]
            except (IndexError, KeyError, TypeError) as e:
                raise ArgumentValueError(f"Invalid index '{part}' for {attr}", attr, {"nested": nested}) from e
        else:
            try:
                result = getattr(result, part)
            except AttributeError as e:
                raise ArgumentValueError(f"Invalid attribute '{part}' for {attr}", attr, {"nested": nested}) from e

    return result


def _maybe_resolve_annotation(fn: Callable, arg_name: str, annotation: Any) -> Any:
    """Internal function to resolve an annotation to its actual type.

    This function handles string annotations, ForwardRef, and generic types (e.g., Optional, List)
    by resolving string annotations within them, using TYPE_CHECKING blocks and their imports.

    Args:
        fn (Callable): The function containing the annotation
        arg_name (str): The name of the parameter with the annotation
        annotation (Any): The annotation to resolve (string, ForwardRef, or type)

    Returns:
        Any: The resolved type, or the original annotation if resolution fails
    """
    # Case 1: Annotation is a string
    if isinstance(annotation, str):
        resolved = _resolve_type_checking_annotation(fn, annotation)
        return resolved if resolved != annotation else annotation

    # Case 2: Annotation is a ForwardRef
    elif isinstance(annotation, ForwardRef):
        return _resolve_type_checking_annotation(fn, annotation.__forward_arg__)

    # Case 3: Annotation is a generic type (e.g., Optional, List, Union)
    elif (origin := get_origin(annotation)) is not None:
        args = get_args(annotation)
        resolved_args = tuple(_maybe_resolve_annotation(fn, arg_name, arg) for arg in args)
        if origin is list:
            return List[resolved_args[0]]
        elif origin is dict:
            return Dict[resolved_args[0], resolved_args[1]]
        elif origin is tuple:
            return Tuple[resolved_args]
        elif origin is set:
            return Set[resolved_args[0]]
        elif origin is frozenset:
            return FrozenSet[resolved_args[0]]
        elif origin is Union:
            return Union[resolved_args]
        else:
            return annotation  # Unhandled generic types return as-is

    # Case 4: Annotation is a non-generic type (e.g., int, str)
    else:
        return annotation


def _resolve_type_checking_annotation(fn: Callable, annotation: str) -> Any:
    """Helper function to resolve a string annotation to its actual type using TYPE_CHECKING imports."""
    if hasattr(fn, "__fn_or_cls__"):
        fn = fn.__fn_or_cls__

    try:
        source_file = inspect.getsourcefile(fn)
        if not source_file:
            return annotation
        with open(source_file, "r") as f:
            source = f.read()
        tree = ast.parse(source)
        type_checking_imports = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING":
                for stmt in node.body:
                    if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                        if isinstance(stmt, ast.Import):
                            for name in stmt.names:
                                type_checking_imports[name.asname or name.name] = name.name
                        else:  # ImportFrom
                            module = stmt.module or ""
                            for name in stmt.names:
                                full_name = f"{module}.{name.name}" if module else name.name
                                type_checking_imports[name.asname or name.name] = full_name
        if annotation in type_checking_imports:
            try:
                full_path = type_checking_imports[annotation]
                module_name, type_name = full_path.rsplit(".", 1)
                module = importlib.import_module(module_name)
                return getattr(module, type_name)
            except (ImportError, AttributeError):
                pass
    except Exception:
        pass
    return annotation
