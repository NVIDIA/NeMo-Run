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
from typing import Any, Dict, List
import typer
from typer import Typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box
from typer import rich_utils

# Import necessary functions from workspace
from nemo_run.cli.core.workspace import load_cache, cache_executor_classes, sync


def render_help(argv: List[str]) -> int:
    """
    Renders help information from the cache based on provided arguments.
    
    Args:
        argv (List[str]): Command-line arguments (e.g., ["nemo", "llm", "finetune", "--help"]).
        
    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    try:
        sync(verbose=False)
        
        # Load the cache data
        cache_data = load_cache()
        
        # Extract the command/group path from arguments
        help_path = [arg for arg in argv[1:] if arg not in ("--help", "-h")]
        
        if not help_path:
            # If no specific path is requested, render the root help
            render_root_help(cache_data)
        else:
            # Find the requested group or command
            render_path_help(cache_data, help_path)
        
        # Exit with success since we've rendered the help information
        sys.exit(0)
        
    except Exception as e:
        from rich.console import Console
        console = Console()
        console.print(f"[bold red]Error displaying help from cache:[/bold red] {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

def render_root_help(cache_data: Dict[str, Any]) -> None:
    """Render the root help information from the cache."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    
    console = Console()
    
    # Display the app name and help
    app_data = cache_data["app"]
    title = Text("NeMo Run CLI", style="bold cyan")
    description = Text(app_data.get("help", "Command-line interface for NeMo Run."))
    
    console.print(title)
    console.print(description)
    console.print()
    
    # Display available groups
    console.print(Text("Available command groups:", style="bold yellow"))
    
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Name", style="green")
    table.add_column("Description", style="white")
    
    for group_name, group_data in app_data.get("groups", {}).items():
        help_text = group_data.get("help", f"Commands in the {group_name} module")
        table.add_row(group_name, help_text)
    
    console.print(table)
    console.print()
    
    # Show usage examples
    console.print(Text("Usage examples:", style="bold yellow"))
    console.print("  nemo <command-group> --help     Show help for a command group")
    console.print("  nemo <command-group> <command>  Run a specific command")
    console.print()
    
    # Show common options
    common_options = cache_data.get("data", {}).get("common_options", {})
    if common_options:
        console.print(Text("Common options:", style="bold yellow"))
        options_table = Table(show_header=False, box=None, padding=(0, 2))
        options_table.add_column("Option", style="green")
        options_table.add_column("Description", style="white")
        
        for option_name, option_data in common_options.items():
            options_table.add_row(option_name, option_data.get("help", ""))
        
        console.print(options_table)
        console.print()

def render_path_help(cache_data: Dict[str, Any], path: list[str]) -> None:
    """
    Render help for a specific command or group path.
    
    Args:
        cache_data (Dict[str, Any]): The cache data
        path (list[str]): The command/group path, e.g. ["llm"] or ["llm", "finetune"]
    """
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from nemo_run.help import render_arguments, render_entrypoint_factories, render_parameter_factories, render_executors
    
    console = Console()
    
    # Navigate the group structure to find the target
    current = cache_data["app"]["groups"]
    target = None
    found = False
    
    # Try direct path lookup
    current_level = current
    for i, part in enumerate(path):
        if part in current_level:
            if i == len(path) - 1:
                # This is the target
                target = current_level[part]
                found = True
                break
            elif "commands" in current_level[part]:
                current_level = current_level[part]["commands"]
            else:
                # Can't go deeper
                break
        else:
            # Path part not found at this level
            break
    
    # If we found the target, render it
    if found:
        if "signature" in target:
            # This is a command
            render_command_help(path[-1], target, path, cache_data)
        else:
            # This is a group
            render_group_help(path[-1], target, path, cache_data)
    else:
        # Show error with available options
        console.print(f"[bold red]Error:[/bold red] Command or group '[bold]{' '.join(path)}[/bold]' not found.")
        
        # Show what's available at the current level or root if we didn't match any part
        available_level = current
        available_path = []
        
        for i, part in enumerate(path):
            if part in available_level:
                available_path.append(part)
                if "commands" in available_level[part]:
                    available_level = available_level[part]["commands"]
                else:
                    break
            else:
                break
        
        console.print(f"\nAvailable options at [bold]nemo {' '.join(available_path)}[/bold]:")
        
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Name", style="green")
        table.add_column("Description", style="white")
        
        for name, data in available_level.items():
            help_text = data.get("help", "")
            table.add_row(name, help_text)
        
        console.print(table)

def render_command_help(cmd_name: str, cmd_data: Dict[str, Any], path_items: list[str], cache_data: Dict[str, Any]) -> None:
    """Render help for a command with styling that matches Typer exactly."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich import box
    from typer import rich_utils
    from nemo_run.help import render_arguments, render_entrypoint_factories, render_parameter_factories, render_executors
    
    # Create console with proper styling matching typer
    console = Console(
        highlighter=rich_utils.highlighter,
        color_system=rich_utils.COLOR_SYSTEM,
        force_terminal=rich_utils.FORCE_TERMINAL,
        width=rich_utils.MAX_WIDTH,
    )
    
    # Clean up path for display - remove duplicates and normalize
    clean_path = []
    for item in path_items:
        if item != "nemo" or not clean_path:  # Keep only first "nemo"
            clean_path.append(item)
    
    # Command header
    full_command = " ".join(clean_path)
    console.print(f"Usage: {full_command} [OPTIONS] [ARGUMENTS]", style=rich_utils.STYLE_USAGE)
    console.print()
    console.print(Text(f"[Entrypoint] {cmd_data.get('help', '')}", style="cyan"))
    console.print()
    
    # Get all available factories from the cache
    factories = cache_data["data"].get("factories", {})
    
    # Step 1: Render Options section first (to match the expected order)
    render_options_table(console)
    
    # Step 2: Arguments table using the existing render_arguments function
    if cmd_data.get("signature"):
        # Add the required line_no field if missing
        for param in cmd_data.get("signature", []):
            if "line_no" not in param:
                param["line_no"] = ""
                
        console.print(render_arguments(cmd_data["signature"]))
    
    # Step 3: Pre-loaded Entrypoint Factories
    full_namespace = cmd_data.get("full_namespace", "")
    if full_namespace and full_namespace in factories:
        # Ensure each factory has line_no data before rendering
        entrypoint_factories = factories[full_namespace]
        for factory in entrypoint_factories:
            if "line_no" not in factory:
                factory["line_no"] = ""
        
        panel = render_entrypoint_factories(
            entrypoint_factories,
            title="Pre-loaded entrypoint factories, run with --factory"
        )
        console.print(panel)
    
    # Step 4: For each parameter, render both parameter-specific and type-based factories
    # Track which types we've already rendered to avoid duplicates
    rendered_types = set()
    
    if cmd_data.get("signature"):
        for param in cmd_data.get("signature", []):
            param_name = param.get("name", "")
            param_type = param.get("type", "")  # Full type namespace for matching
            
            # First check for parameter-specific factories
            if full_namespace:
                param_namespace = f"{full_namespace}.{param_name}"
                if param_namespace in factories:
                    param_factories = factories[param_namespace]
                    
                    # Ensure each factory has line_no data
                    for factory in param_factories:
                        if "line_no" not in factory:
                            factory["line_no"] = ""
                    
                    panel = render_parameter_factories(param_name, param_type, param_factories)
                    console.print(panel)
            
            # Then check for type-based factories using the full type namespace
            if param_type and param_type != "Any" and param_type in factories and param_type not in rendered_types:
                type_factories = factories[param_type]
                
                # Ensure each factory has line_no data
                for factory in type_factories:
                    if "line_no" not in factory:
                        factory["line_no"] = ""
                
                panel = render_parameter_factories(param_name, param_type, type_factories)
                console.print(panel)
                rendered_types.add(param_type)
    
    # Step 5: Look for executor factories based on their namespace
    # Get executor namespaces from the metadata instead of hardcoding them
    executor_namespaces = cache_data["metadata"].get("executor_namespaces", [])
    
    # If we don't have executor namespaces in the cache (for backward compatibility),
    # populate them now
    if not executor_namespaces:
        cache_executor_classes()
        # Reload the cache to get the executor namespaces
        updated_cache = load_cache()
        executor_namespaces = updated_cache["metadata"].get("executor_namespaces", [])
    
    # Collect all executor factories, using a dictionary to deduplicate
    executor_factories_dict = {}
    for namespace in executor_namespaces:
        if namespace in factories:
            for factory in factories[namespace]:
                # Use factory name and function as the unique key
                key = (factory["name"], factory["fn"])
                if key not in executor_factories_dict:
                    if "line_no" not in factory:
                        factory["line_no"] = ""
                    executor_factories_dict[key] = factory

    # Convert to list for rendering
    executor_factories = list(executor_factories_dict.values())

    # Render the executors panel if we found any executor factories
    if executor_factories:
        panel = render_executors(executor_factories)
        console.print(panel)


def render_options_table(console: Console) -> None:
    """Render the options table using the standard entrypoint options with proper formatting."""
    from rich.panel import Panel
    from rich.table import Table
    from rich import box
    from typer import rich_utils
    
    # Create the options table with proper styling
    options_table = Table(
        show_header=False,
        box=None,  # Key difference - typer uses no internal boxes
        expand=True,
        show_lines=rich_utils.STYLE_OPTIONS_TABLE_SHOW_LINES,
        padding=rich_utils.STYLE_OPTIONS_TABLE_PADDING,
        pad_edge=rich_utils.STYLE_OPTIONS_TABLE_PAD_EDGE,
    )
    
    options_table.add_column("Option", style=rich_utils.STYLE_OPTION, no_wrap=True)
    options_table.add_column("Description", style="white")
    
    # Add all standard options from entrypoint_options with proper formatting
    from nemo_run.cli.core.entrypoint import entrypoint_options
    
    for option_name, option in entrypoint_options.items():
        # Format the option flags
        flag_str = ""
        if option.flags:
            flag_str = ", ".join(option.flags)
        else:
            flag_str = f"--{option_name}"
            
        # Special handling for toggle options that show as --opt/--no-opt
        if isinstance(option.default, bool):
            positive = f"--{option_name}"
            negative = f"--no-{option_name}"
            if option.flags:
                flags = [f for f in option.flags if f.startswith("--")]
                if len(flags) >= 2:
                    positive = flags[0]
                    negative = flags[1]
            flag_str = f"{positive}{' '*max(1, 20-len(positive))}{negative}"
        
        # Format the help text
        help_text = option.help or ""
        
        # Format default differently based on type
        if option.default is not None:
            # Format default value
            if isinstance(option.default, bool):
                default = "no-" + option_name if not option.default else option_name
                default_str = f"[default: {default}]"
            else:
                default_str = f"[default: {option.default}]"
            
            # Position the default value properly
            help_text = f"{help_text}\n{' '*8}{default_str}"
        
        options_table.add_row(flag_str, help_text)
    
    # Add standard Typer options
    options_table.add_row("--help", "Show this message and exit.")
    options_table.add_row("--version", "Show the version and exit.")
    
    console.print(
        Panel(
            options_table,
            title=rich_utils.OPTIONS_PANEL_TITLE,
            border_style=rich_utils.STYLE_OPTIONS_PANEL_BORDER,
            title_align=rich_utils.ALIGN_OPTIONS_PANEL,
            box=box.ROUNDED,
        )
    )

def render_group_help(group_name: str, group_data: Dict[str, Any], path_items: list[str], cache_data: Dict[str, Any]) -> None:
    """Render help for a group, matching Typer's styling exactly."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich import box
    from typer import rich_utils
    
    # Create console with proper styling matching typer
    console = Console(
        highlighter=rich_utils.highlighter,
        color_system=rich_utils.COLOR_SYSTEM,
        force_terminal=rich_utils.FORCE_TERMINAL,
        width=rich_utils.MAX_WIDTH,
    )
    
    # Clean up path for display - remove duplicates and normalize
    clean_path = []
    for item in path_items:
        if item != "nemo" or not clean_path:  # Keep only first "nemo"
            clean_path.append(item)
    
    # Format the help header similar to Typer
    full_path = " ".join(clean_path)
    console.print(f"Usage: {full_path} [OPTIONS] COMMAND [ARGS]...", style=rich_utils.STYLE_USAGE)
    console.print()
    console.print(Text(group_data.get("help", f"[Module] {group_name}"), style="cyan"))
    console.print()
    
    # Options section with proper box styling (exactly matching typer)
    options_table = Table(
        show_header=False,
        box=None,  # Key difference - typer uses no internal boxes
        expand=True,
        show_lines=rich_utils.STYLE_OPTIONS_TABLE_SHOW_LINES,
        padding=rich_utils.STYLE_OPTIONS_TABLE_PADDING,
        pad_edge=rich_utils.STYLE_OPTIONS_TABLE_PAD_EDGE,
    )
    options_table.add_column("Option", style=rich_utils.STYLE_OPTION, no_wrap=True)
    options_table.add_column("Description", style="white")
    options_table.add_row("--help", "Show this message and exit.")
    
    console.print(
        Panel(
            options_table,
            title=rich_utils.OPTIONS_PANEL_TITLE,
            border_style=rich_utils.STYLE_OPTIONS_PANEL_BORDER,
            title_align=rich_utils.ALIGN_OPTIONS_PANEL,
            box=box.ROUNDED,
        )
    )
    
    # Commands section with proper box styling
    commands = []
    for cmd_name, cmd_data in group_data.get("commands", {}).items():
        if isinstance(cmd_data, dict) and "signature" in cmd_data:
            # Extract just the first line of help for the command list
            help_text = cmd_data.get("help", "")
            first_line = help_text.split('\n')[0] if help_text else ""
            commands.append((cmd_name, f"[Entrypoint] {first_line}"))
    
    if commands:
        commands_table = Table(
            show_header=False,
            box=None,  # Key difference - typer uses no internal boxes
            expand=True,
            show_lines=rich_utils.STYLE_COMMANDS_TABLE_SHOW_LINES,
            padding=rich_utils.STYLE_COMMANDS_TABLE_PADDING,
            pad_edge=rich_utils.STYLE_COMMANDS_TABLE_PAD_EDGE,
        )
        commands_table.add_column("Command", style=rich_utils.STYLE_COMMANDS_TABLE_FIRST_COLUMN, no_wrap=True)
        commands_table.add_column("Description", style="white")
        
        for cmd_name, help_text in commands:
            commands_table.add_row(cmd_name, help_text)
        
        console.print(
            Panel(
                commands_table,
                title=rich_utils.COMMANDS_PANEL_TITLE,
                border_style=rich_utils.STYLE_COMMANDS_PANEL_BORDER, 
                title_align=rich_utils.ALIGN_COMMANDS_PANEL,
                box=box.ROUNDED,
            )
        )
    
    # Nested groups (if any)
    nested_groups = []
    for nested_name, nested_data in group_data.get("commands", {}).items():
        if isinstance(nested_data, dict) and "commands" in nested_data and "signature" not in nested_data:
            help_text = nested_data.get("help", f"Commands in the {nested_name} group")
            nested_groups.append((nested_name, help_text))
    
    if nested_groups:
        nested_table = Table(
            show_header=False,
            box=None,  # Key difference - typer uses no internal boxes
            expand=True,
            show_lines=rich_utils.STYLE_COMMANDS_TABLE_SHOW_LINES,
            padding=rich_utils.STYLE_COMMANDS_TABLE_PADDING,
            pad_edge=rich_utils.STYLE_COMMANDS_TABLE_PAD_EDGE,
        )
        nested_table.add_column("Group", style=rich_utils.STYLE_COMMANDS_TABLE_FIRST_COLUMN, no_wrap=True)
        nested_table.add_column("Description", style="white")
        
        for nested_name, help_text in nested_groups:
            nested_table.add_row(nested_name, help_text)
        
        console.print(
            Panel(
                nested_table,
                title="Nested Groups",
                border_style=rich_utils.STYLE_COMMANDS_PANEL_BORDER,
                title_align=rich_utils.ALIGN_COMMANDS_PANEL,
                box=box.ROUNDED,
            )
        )