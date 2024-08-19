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

from typing import Any

from lark import Lark, Token, Transformer

_GRAMMAR = r"""
// High-level command-line override.
start: (key EQUAL value? | TILDE key (EQUAL value?)? | PLUS key EQUAL value?)

// Keys.
key: (OVERWRITE | ARG)

// Elements (that may be swept over).
value: element
element: primitive | list_value | dict_value | function

// Functions.
arg_name: ARG EQUAL
function: ARG "(" (arg_name? element (COMMA arg_name? element)*)? ")"

// Data structures.
list_value: "[" (element (COMMA element)*)? "]"
dict_value: "{" (dict_key_value_pair (COMMA dict_key_value_pair)*)? "}"
dict_key_value_pair: ARG COLON element

// Primitive types.
primitive: QUOTED_VALUE | TIME | PATH | (ID | NULL | INT | FLOAT | BOOL | COLON | ESC | WS)+

// Terminals (tokens).
OVERWRITE: /[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)+/
BOOL.100: "true"i | "false"i
ID: /\$[a-zA-Z_][a-zA-Z0-9_]*|\$\{[a-zA-Z_][a-zA-Z0-9_]*\}/
ARG: /[a-zA-Z_][a-zA-Z0-9_]*/
PATH.1: /[a-zA-Z_][a-zA-Z0-9_\/:.-]*/
NULL.50: "null"i | "none"i
EQUAL: "="
TILDE: "~"
PLUS: "+"
AT: "@"
SLASH: "/"
COMMA: ","
COLON: ":"
INT_UNSIGNED: "0" | /[1-9]\d*/ | /[1-9](_?\d)*/
QUOTED_VALUE: /'[^']*'|"[^"]*"/
ESC: /\\[nrt'"\[\]{}:=\\]/
INT: /[+-]?/ INT_UNSIGNED
DIGIT: /[0-9]/
POINT_FLOAT: INT_UNSIGNED "." | INT_UNSIGNED? "." DIGIT (("_")? DIGIT)*
EXPONENT_FLOAT: (INT_UNSIGNED | POINT_FLOAT) /[eE]/ /[+-]?/ DIGIT (("_")? DIGIT)*
FLOAT: /[+-]?/ (POINT_FLOAT | EXPONENT_FLOAT | /[Ii][Nn][Ff]/ | /[Nn][Aa][Nn]/)
TIME.1: /\d{1,2}:\d{2}(:\d{2})?/

%import common.WS
%ignore WS
"""


class ArgTransformer(Transformer):
    INT = int
    FLOAT = float

    def NULL(self, _):
        return None

    def BOOL(self, value):
        return value.lower() == "true"

    def QUOTED_VALUE(self, value):
        return value[1:-1]

    def ID(self, value):
        if value.startswith("${") and value.endswith("}"):
            return value[2:-1]
        return value[1:]

    def ARG(self, value):
        return str(value)

    def PATH(self, value):
        return str(value)

    def TIME(self, value):
        return str(value)

    def start(self, items):
        return (items[0], items[-1])

    def key(self, items):
        return items[0]

    def value(self, items):
        if not items or (
            len(items) == 1 and isinstance(items[0], Token) and items[0].type == "EQUAL"
        ):
            return None
        return items[0] if len(items) == 1 else items

    def element(self, items):
        return items[0]

    def primitive(self, items):
        return items[0]

    def list_value(self, items: list[Token]) -> list[Token]:
        return [item for item in items if item != ","]

    def dict_value(self, items: list[tuple[Token, Token]]) -> dict[Token, Token]:
        return {item[0]: item[1] for item in items if item != ","}

    def dict_key_value_pair(self, items: list[Token]) -> tuple[Token, Token]:
        return (items[0], items[-1])


def parse_args(args: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    parsed_args = {}
    parsed_overrides = {}
    parser = Lark(
        _GRAMMAR,
        start="start",
        parser="lalr",
        transformer=ArgTransformer(),
        cache=True,
    )
    for arg in args:
        key, value = parser.parse(arg)  # type: ignore

        if isinstance(key, Token):
            key = key.value

        if key.startswith("~") or key.startswith("+") or "." in key:
            parsed_overrides[key] = value
        else:
            parsed_args[key] = value
    return parsed_args, parsed_overrides
