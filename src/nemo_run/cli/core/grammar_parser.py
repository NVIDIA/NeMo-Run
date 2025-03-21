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
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from lark import Lark, UnexpectedInput, UnexpectedToken, GrammarError
from functools import lru_cache
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich import box
from rich.theme import Theme
from rich.syntax import Syntax
from rich.console import Group
from nemo_run.core.frontend.console.styles import BOX_STYLE, TABLE_STYLES
from typer import rich_utils
import difflib


class ArgumentParsingError(Exception):
    """Custom exception for argument parsing errors with additional context."""
    def __init__(self, message: str, args: List[str] = [], pos: int = 0):
        self.message = message
        self.args = args  # Original list of arguments for CLI rendering
        self.pos = pos    # Error position for highlighting
        super().__init__(message)


def extract_command_and_args(args: List[str], cache: Dict[str, Any]) -> Tuple[str, List[str]]:
    """
    Extract the command and arguments from the args list using the cache's group structure.

    Args:
        args (List[str]): The full list of command-line arguments, including the command.
        cache (Dict[str, Any]): Cache containing the app's group structure under "app" and "groups".

    Returns:
        Tuple[str, List[str]]: The extracted command (as a space-separated string) and the remaining arguments.

    Raises:
        ValueError: If the command is invalid or incomplete (e.g., points to a group instead of a command).
    """
    path = []
    current = cache["app"]["groups"]
    for i, arg in enumerate(args):
        if arg in current:
            path.append(arg)
            if "commands" not in current[arg] or not current[arg]["commands"]:
                # It's a command (leaf node, no further "commands" key or empty "commands")
                command = " ".join(path)
                arguments = args[i + 1:]
                return command, arguments
            else:
                # It's a group, continue traversing
                current = current[arg]["commands"]
        else:
            # Argument not found in current context, invalid command
            raise ValueError(f"Invalid command: {' '.join(path + [arg])}")
    # If loop exits without finding a command, the path is a group, not a complete command
    raise ValueError(f"Incomplete command: {' '.join(path)}")


def get_parser_for_command(command: str, cache: Dict[str, Any]) -> Lark:
    """
    Generate or retrieve a cached Lark parser for the given command using the cache.

    Args:
        command (str): The full command path, e.g., "nemo llm pretrain".
        cache (Dict[str, Any]): Cache containing command metadata under "commands".

    Returns:
        Lark: A Lark parser instance for the command's grammar.
    """
    # First check if we already have a parser cached for this command
    parser_cache_key = f"parser_{command}"
    if parser_cache_key in cache.get("parsers", {}):
        return cache["parsers"][parser_cache_key]
    
    # Generate the grammar for this command
    grammar = generate_command_grammar(command, cache)
    parser = Lark(grammar, start="start", parser="lalr")
    
    # Cache the parser for future use
    if "parsers" not in cache:
        cache["parsers"] = {}
    cache["parsers"][parser_cache_key] = parser
    
    return parser


def generate_command_grammar(command: str, cache: Dict[str, Any]) -> str:
    """
    Generate a Lark grammar string for the given command based on its metadata in the cache.

    Args:
        command (str): The full command path, e.g., "nemo llm pretrain".
        cache (Dict[str, Any]): Cache containing command metadata under "commands".

    Returns:
        str: The Lark grammar string for parsing the command's arguments.
    """
    # Find the command in the cache structure and get argument metadata
    cmd_parts = command.split()
    cmd_metadata = None
    current = cache["app"]["groups"]
    for part in cmd_parts:
        if part in current:
            if "commands" in current[part]:
                current = current[part]["commands"]
            else:
                cmd_metadata = current[part]
                break
        else:
            break
    
    # Extract arguments metadata
    arguments_metadata = []
    if cmd_metadata and "signature" in cmd_metadata:
        arguments_metadata = cmd_metadata["signature"]
    
    # Print debug information
    print(f"Command: {command}")
    print(f"Arguments: {[arg['name'] for arg in arguments_metadata]}")
    
    # Base grammar template with simplified structure
    base_grammar = """
    %import common (WS, DIGIT, LETTER, SIGNED_NUMBER, ESCAPED_STRING)
    %ignore WS

    start: argument+

    argument: named_argument | general_arg

    named_argument: {named_args}

    operation: "=" | "+=" | "-=" | "*=" | "/=" | "|=" | "&="

    value: literal | collection | expression | factory_call

    literal: number | string | boolean | "None" | "null"
    number: SIGNED_NUMBER
    string: ESCAPED_STRING
    boolean: "true" | "false" | "True" | "False" | "1" | "0"

    collection: list | dict
    list: "[" [value ("," value)* [","]] "]"
    dict: "{{" [key_value ("," key_value)* [","]] "}}"
    key_value: string ":" value

    expression: comprehension | lambda_expr | ternary_expr
    comprehension: "[" value "for" identifier "in" value "]"
    lambda_expr: "lambda" identifier ("," identifier)* ":" value
    ternary_expr: value "if" value "else" value

    factory_call: identifier "(" [factory_arg_list] ")"
    factory_arg_list: factory_argument ("," factory_argument)* [","]
    factory_argument: identifier "=" value

    identifier: (LETTER | "_") (LETTER | DIGIT | "_")*

    key: identifier (accessor)*
    accessor: "." identifier | "[" index "]"
    index: DIGIT+

    general_arg: key operation value
    """

    # Generate named argument rules - simpler approach without type validation in grammar
    named_args = []
    for arg in arguments_metadata:
        arg_name = arg["name"]
        named_args.append(f'"{arg_name}" operation value')

    # Combine named argument rules
    named_args_str = " | ".join(named_args) if named_args else "key operation value"
    
    # Format the grammar
    return base_grammar.format(named_args=named_args_str)


def get_close_matches(word: str, possibilities: List[str], n: int = 3, cutoff: float = 0.6) -> List[str]:
    """Get a list of close matches for a word from a list of possibilities.
    
    Args:
        word: The word to match
        possibilities: List of possible matches
        n: Maximum number of matches to return
        cutoff: Match threshold (0.0 - 1.0)
        
    Returns:
        List of strings that closely match the word
    """
    return difflib.get_close_matches(word, possibilities, n=n, cutoff=cutoff)


def format_factory_error(
    factory_name: str, 
    arg_name: str, 
    input_text: str, 
    valid_factories: List[str],
    suggestions: List[str],
    minimal_example: str,
    full_example: str,
    highlight_position: Optional[int] = None
) -> Tuple[str, str]:
    """
    Format a factory validation error with rich output.
    
    Args:
        factory_name: The invalid factory name
        arg_name: The argument name
        input_text: The user's input for context
        valid_factories: List of valid factories for this argument
        suggestions: List of suggested factories (close matches)
        minimal_example: Simple example without parameters
        full_example: Example with parameters
        highlight_position: Position to highlight in the context (for syntax errors)
    
    Returns:
        Tuple[str, str]: Rich-formatted error message and plain text version
    """
    console = Console(record=True)
    error_title = Text("Invalid Factory Error", style=rich_utils.STYLE_ERRORS_PANEL_BORDER)
    
    # Create context code with syntax highlighting
    if highlight_position is not None:
        # For UnexpectedInput errors
        code_context = Syntax(
            input_text,
            "python",
            theme="monokai",
            line_numbers=False,
            highlight_lines={1: [highlight_position, highlight_position + len(factory_name)]},
            word_wrap=True
        )
    else:
        # For pre-validation errors
        code_context = Syntax(
            input_text,
            "python",
            theme="monokai",
            line_numbers=False,
            highlight_lines={1: [input_text.find(factory_name), input_text.find(factory_name) + len(factory_name)]},
            word_wrap=True
        )
    
    # Format available factories as a table with consistent styling
    factories_table = Table(
        show_header=True,
        box=BOX_STYLE,
        title=f"Available Factories for '{arg_name}'",
        title_style="bold cyan",
        **TABLE_STYLES
    )
    factories_table.add_column("Factory Name", style="green")
    
    # Add factories in multiple columns if there are many
    num_cols = min(3, len(valid_factories))
    if num_cols > 0:
        factory_chunks = [
            valid_factories[i:i + len(valid_factories)//num_cols + 1] 
            for i in range(0, len(valid_factories), len(valid_factories)//num_cols + 1)
        ]
        
        # Ensure all columns have the same number of rows by padding with empty strings
        max_rows = max(len(chunk) for chunk in factory_chunks)
        for chunk in factory_chunks:
            chunk.extend([''] * (max_rows - len(chunk)))
        
        # Transpose the chunks to get rows with one item from each column
        for row in zip(*factory_chunks):
            factories_table.add_row(*row)
    
    # Create example syntax with both minimal and full examples
    example_text = minimal_example
    if full_example and full_example != minimal_example:
        example_text = f"{minimal_example}\n\n# With parameters:\n{full_example}"
    
    example_syntax = Syntax(
        example_text,
        "python",
        theme="monokai",
        line_numbers=False,
        word_wrap=True
    )
    
    # Create the base error components
    error_components = Group(
        Text(f"The factory '{factory_name}' is not valid for argument '{arg_name}'.\n\n"),
        Text("Context:", style="bold yellow"),
        code_context,
        Text("\n"),
        factories_table,
    )
    
    # Add suggestions if we have any
    if suggestions:
        suggestion_text = Text("\nDid you mean: ", style="bold green")
        for i, suggestion in enumerate(suggestions):
            suggestion_text.append(Text(suggestion, style="bold cyan"))
            if i < len(suggestions) - 1:
                suggestion_text.append(", ")
        error_components = Group(
            error_components,
            suggestion_text,
        )
    
    # Add examples
    error_components = Group(
        error_components,
        Text("\nExample Usage:", style="bold yellow"),
        example_syntax
    )
    
    # Render the panel with consistent styling
    console.print(Panel(
        error_components,
        title=error_title,
        border_style=rich_utils.STYLE_ERRORS_PANEL_BORDER,
        expand=False,
        title_align=rich_utils.ALIGN_ERRORS_PANEL,
    ))
    
    # Get the rendered output
    rich_error_msg = console.export_text()
    
    # Create plain text version for systems without rich support
    plain_error_msg = (
        f"Invalid factory '{factory_name}' for argument '{arg_name}':\n"
        f"{input_text}\n\n"
        f"Available factories for '{arg_name}': {', '.join(valid_factories)}\n"
    )
    
    if suggestions:
        plain_error_msg += f"Did you mean: {', '.join(suggestions)}\n\n"
        
    plain_error_msg += f"Example usage:\n{minimal_example}"
    if full_example != minimal_example:
        plain_error_msg += f"\n\nWith parameters:\n{full_example}"
    
    return rich_error_msg, plain_error_msg


def get_factory_examples(
    factory_name: str, 
    arg_name: str, 
    factory_params: Dict[str, Any]
) -> Tuple[str, str]:
    """
    Create examples for factory usage, both minimal and with parameters.
    
    Args:
        factory_name: The factory name to use in examples
        arg_name: The argument name
        factory_params: Dictionary of factory parameters and their types
        
    Returns:
        Tuple[str, str]: Minimal example (no params) and full example (with params)
    """
    # Create a minimal example (no params)
    minimal_example = f"{arg_name}={factory_name}()"
    
    # Build the parameterized example if we have parameters
    if factory_params:
        # Use actual parameters with example values based on types
        param_parts = []
        for param_name, param_info in factory_params.items():
            param_type = param_info.get("type", "str")
            if param_type == "int":
                example_value = "1"
            elif param_type == "float":
                example_value = "0.5"
            elif param_type == "bool":
                example_value = "True"
            elif param_type == "list":
                example_value = "[]"
            elif param_type == "dict":
                example_value = "{}"
            else:
                example_value = '"value"'
            param_parts.append(f"{param_name}={example_value}")
        
        param_example = ", ".join(param_parts)
        full_example = f"{arg_name}={factory_name}({param_example})"
    else:
        # Fallback to generic parameters if metadata isn't available
        full_example = f"{arg_name}={factory_name}(param1=value)"
    
    return minimal_example, full_example


def format_syntax_error(
    command: str,
    context: str,
    error: UnexpectedInput,
    position: int,
    token_value: Optional[str] = None,
    suggestions: List[str] = []
) -> Tuple[str, str]:
    """
    Format a syntax error with rich output.
    
    Args:
        command: The command being executed
        context: The context string around the error
        error: The UnexpectedInput error
        position: Position of error relative to context start
        token_value: Value of the token (for argument name suggestions)
        suggestions: List of suggested argument names
        
    Returns:
        Tuple[str, str]: Rich-formatted error message and plain text version
    """
    console = Console(record=True)
    error_title = Text("Syntax Error", style=rich_utils.STYLE_ERRORS_PANEL_BORDER)
    
    # Highlight the error position in the context
    code_syntax = Syntax(
        context,
        "python",
        theme="monokai",
        line_numbers=False,
        highlight_lines={1: [position, position+1]},
        word_wrap=True
    )
    
    # Create detailed error explanation
    error_details = str(error)
    if isinstance(error, UnexpectedToken):
        error_details = f"Unexpected token: '{error.token}'\nExpected one of: {', '.join(error.expected)}"
    
    # Build error message components
    error_components = Group(
        Text(f"Syntax error in command '{command}' at position {error.pos_in_stream}:\n\n"),
        Text("Context:", style="bold yellow"),
        code_syntax,
        Text("\nError Details:", style="bold yellow"),
        Text(error_details)
    )
    
    # Add suggestions if applicable
    if suggestions:
        suggestion_text = Text("\nDid you mean: ", style="bold green")
        for i, suggestion in enumerate(suggestions):
            suggestion_text.append(Text(suggestion, style="bold cyan"))
            if i < len(suggestions) - 1:
                suggestion_text.append(", ")
        
        # Create a new group with the suggestions
        error_components = Group(
            error_components,
            suggestion_text
        )
    
    # Print the panel
    console.print(Panel(
        error_components,
        title=error_title,
        border_style=rich_utils.STYLE_ERRORS_PANEL_BORDER,
        expand=False,
        title_align=rich_utils.ALIGN_ERRORS_PANEL,
    ))
    
    # Get rendered output
    rich_error = console.export_text()
    
    # Plain text version
    plain_error = (
        f"Syntax error in command '{command}' at position {error.pos_in_stream}:\n"
        f"{context}\n"
        f"{error_details}"
    )
    
    if suggestions:
        plain_error += f"\n\nDid you mean: {', '.join(suggestions)}"
    
    return rich_error, plain_error


def cached_parse(args: List[str], cache: Dict[str, Any]) -> 'Lark.Tree':
    """
    Parse command-line arguments for a command extracted from args using a cached Lark parser,
    with additional factory validation after parsing.

    Args:
        args (List[str]): The full list of command-line arguments, including the command.
        cache (Dict[str, Any]): Cache containing group structure under "app" and command metadata.

    Returns:
        Lark.Tree: The parse tree of the command's arguments.

    Raises:
        ValueError: If the command extracted from args is invalid or incomplete.
        ArgumentParsingError: If parsing the arguments fails, with detailed error information.
    """
    # Extract command and arguments from args
    command, arguments = extract_command_and_args(args, cache)
    user_input = " ".join(arguments)  # Join arguments into a single string for parsing
    
    # First, retrieve factory validation info
    cmd_parts = command.split()
    cmd_metadata = None
    current = cache["app"]["groups"]
    for part in cmd_parts:
        if part in current:
            if "commands" in current[part]:
                current = current[part]["commands"]
            else:
                cmd_metadata = current[part]
                break
        else:
            break
    
    # Build argument to valid factories mapping
    argument_factories = {}
    if cmd_metadata and "signature" in cmd_metadata:
        type_factories = cache.get("data", {}).get("factories", {})
        for arg in cmd_metadata["signature"]:
            arg_name = arg["name"]
            arg_type = arg.get("type", "")
            factories = []
            
            # Check type-based factories
            if arg_type and arg_type in type_factories:
                factories = [factory["name"] for factory in type_factories[arg_type]]
            
            # Check parameter-specific factories
            full_namespace = cmd_metadata.get("full_namespace", "")
            if full_namespace:
                param_namespace = f"{full_namespace}.{arg_name}"
                if param_namespace in type_factories:
                    factories.extend([factory["name"] for factory in type_factories[param_namespace]])
            
            # Store unique factories
            argument_factories[arg_name] = list(set(factories))

    print(f"Factories: {argument_factories}")
    
    # Pre-validate factory calls before parsing
    for arg in arguments:
        if "=" in arg and "(" in arg and ")" in arg:
            # This looks like a factory assignment
            parts = arg.split("=", 1)
            if len(parts) == 2:
                arg_name = parts[0].strip()
                factory_str = parts[1].strip()
                
                # Extract factory name
                factory_name = factory_str.split("(")[0].strip() if "(" in factory_str else factory_str
                
                # Validate if this factory is allowed for this argument
                if arg_name in argument_factories and argument_factories[arg_name]:
                    valid_factories = argument_factories[arg_name]
                    if factory_name not in valid_factories:
                        # Get suggestions for typos
                        suggestions = get_close_matches(factory_name, valid_factories)
                        
                        # Get factory parameters if available
                        factory_params = {}
                        arg_type = ""
                        
                        # Get the type from command metadata
                        if cmd_metadata and "signature" in cmd_metadata:
                            for param in cmd_metadata["signature"]:
                                if param["name"] == arg_name:
                                    arg_type = param.get("type", "")
                                    break
                        
                        # Get factory info using the correct type
                        if arg_type and arg_type in cache.get("data", {}).get("factories", {}):
                            for factory_info in cache["data"]["factories"][arg_type]:
                                if factory_info["name"] == valid_factories[0]:
                                    factory_params = factory_info.get("params", {})
                                    break
                        
                        # Create examples
                        minimal_example, full_example = get_factory_examples(
                            valid_factories[0], arg_name, factory_params
                        )
                        
                        # Format error message
                        error_msg, plain_error_msg = format_factory_error(
                            factory_name, arg_name, arg, valid_factories, 
                            suggestions, minimal_example, full_example
                        )
                        
                        raise ArgumentParsingError(
                            error_msg, args, 
                            len(" ".join(arguments[:arguments.index(arg)]))
                        )
    
    try:
        parser = get_parser_for_command(command, cache)
        # Parse the arguments using the command-specific parser
        tree = parser.parse(user_input)
        return tree
    except GrammarError as e:
        # Rich formatting for grammar errors
        console = Console(record=True)
        error_title = Text("Grammar Error", style=rich_utils.STYLE_ERRORS_PANEL_BORDER)
        error_panel = Panel(
            f"Error in command definition for '{command}':\n\n{str(e)}",
            title=error_title,
            border_style=rich_utils.STYLE_ERRORS_PANEL_BORDER,
            expand=False,
            title_align=rich_utils.ALIGN_ERRORS_PANEL,
        )
        
        console.print(error_panel)
        error_msg = console.export_text()
        plain_error_msg = f"Grammar error in command definition for '{command}': {str(e)}"
        
        raise ArgumentParsingError(error_msg, args) from e
    except UnexpectedInput as e:
        # Extract context from error
        if hasattr(e, 'pos_in_stream') and user_input and e.pos_in_stream < len(user_input):
            context_start = max(0, e.pos_in_stream - 20)
            context_end = min(len(user_input), e.pos_in_stream + 20)
            context = user_input[context_start:context_end]
            
            # Check if this could be related to an invalid factory
            for arg_name, factories in argument_factories.items():
                arg_prefix = f"{arg_name}="
                if arg_prefix in user_input[:e.pos_in_stream]:
                    idx = user_input[:e.pos_in_stream].rfind(arg_prefix)
                    if idx >= 0:
                        factory_start = idx + len(arg_prefix)
                        factory_part = user_input[factory_start:e.pos_in_stream].strip()
                        
                        if "(" in factory_part:
                            factory_name = factory_part.split("(")[0].strip()
                            
                            if factories and factory_name not in factories:
                                # Get suggestions for typos
                                suggestions = get_close_matches(factory_name, factories)
                                
                                # Get factory parameters if available
                                factory_params = {}
                                arg_type = ""
                                
                                # Get the type from command metadata
                                if cmd_metadata and "signature" in cmd_metadata:
                                    for param in cmd_metadata["signature"]:
                                        if param["name"] == arg_name:
                                            arg_type = param.get("type", "")
                                            break
                                
                                # Get factory info
                                if arg_type and arg_type in cache.get("data", {}).get("factories", {}):
                                    for factory_info in cache["data"]["factories"][arg_type]:
                                        if factory_info["name"] == factories[0]:
                                            factory_params = factory_info.get("params", {})
                                            break
                                
                                # Create examples
                                minimal_example, full_example = get_factory_examples(
                                    factories[0], arg_name, factory_params
                                )
                                
                                # Format error message
                                error_msg, plain_error_msg = format_factory_error(
                                    factory_name, arg_name, context, factories, 
                                    suggestions, minimal_example, full_example,
                                    e.pos_in_stream - context_start
                                )
                                
                                raise ArgumentParsingError(
                                    error_msg, args, e.pos_in_stream
                                ) from e

            # If not a factory error, provide a general syntax error
            
            # Check if this might be an invalid argument name
            arg_suggestions = []
            token_value = None
            if isinstance(e, UnexpectedToken) and e.token.type == 'IDENTIFIER':
                # Try to see if this is an invalid argument name
                token_value = e.token.value
                argument_names = []
                if cmd_metadata and "signature" in cmd_metadata:
                    argument_names = [param["name"] for param in cmd_metadata["signature"]]
                
                arg_suggestions = get_close_matches(token_value, argument_names)
            
            # Format the syntax error
            error_msg, plain_error_msg = format_syntax_error(
                command, context, e, 
                e.pos_in_stream - context_start,
                token_value, arg_suggestions
            )
            
            raise ArgumentParsingError(error_msg, args, getattr(e, 'pos_in_stream', 0)) from e
        else:
            # General error without position context
            console = Console(record=True)
            error_title = Text("Syntax Error", style=rich_utils.STYLE_ERRORS_PANEL_BORDER)
            
            console.print(Panel(
                Text(f"Syntax error in command '{command}':\n\n{str(e)}"),
                title=error_title,
                border_style=rich_utils.STYLE_ERRORS_PANEL_BORDER,
                expand=False,
                title_align=rich_utils.ALIGN_ERRORS_PANEL,
            ))
            
            error_msg = console.export_text()
            plain_error_msg = f"Syntax error in command '{command}': {e}"
        
        raise ArgumentParsingError(error_msg, args, getattr(e, 'pos_in_stream', 0)) from e


if __name__ == "__main__":
    # Load the real cache first
    cached = cache.load_cache()
    
    # Create a test cache with known validation data
    test_cache = {
        "app": {
            "groups": {
                "llm": {
                    "commands": {
                        "pretrain": {
                            "signature": [
                                {"name": "model", "type": "model.type"},
                                {"name": "dataset", "type": "dataset.type"}
                            ],
                            "full_namespace": "nemo.collections.nlp.models.language_modeling.megatron_gpt_model"
                        }
                    }
                }
            }
        },
        "data": {
            "factories": {
                "model.type": [
                    {"name": "valid_factory", "params": {
                        "size": {"type": "int"},
                        "name": {"type": "str"},
                        "use_cache": {"type": "bool"}
                    }},
                    {"name": "gpt_model"}
                ],
                "dataset.type": [
                    {"name": "dataset_factory"}
                ]
            }
        }
    }
    
    print("\n--- Testing with built-in test cache ---")
    try:
        # This should fail since invalid_factory isn't in the allowed factories
        tree = cached_parse(["llm", "pretrain", "model=invalid_factory()", "dataset=dataset_factory()"], test_cache)
        print("ERROR: Test did not fail as expected! The grammar accepted an invalid factory.")
    except ArgumentParsingError as e:
        print("Successfully caught parsing error as expected:")
        print(e.message)
    except ValueError as e:
        print(f"Command validation error: {e}")
    except Exception as e:
        print(f"Unexpected error: {type(e).__name__}: {e}")
    
    print("\n--- Testing with real cache ---")
    try:
        # Try with the real cache too - this will help validate with real data
        tree = cached_parse(["llm", "pretrain", "model=invalid_factory()", "dataset=dataset_factory()"], cached)
        print("WARNING: Real cache test did not fail. This might indicate missing validation info in the cache.")
    except ArgumentParsingError as e:
        print("Successfully caught parsing error with real cache:")
        print(e.message)
    except ValueError as e:
        print(f"Command validation error with real cache: {e}")
    except Exception as e:
        print(f"Unexpected error with real cache: {type(e).__name__}: {e}")