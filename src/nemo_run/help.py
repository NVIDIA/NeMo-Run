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

import inspect
import os
import sys
import typing

import catalogue
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from typer import rich_utils

from nemo_run.cli.core.workspace import load_entrypoints, load_workspace
from nemo_run.config import get_type_namespace, get_underlying_types
from nemo_run.core.frontend.console.api import CONSOLE
from nemo_run.core.frontend.console.styles import BOX_STYLE, TABLE_STYLES

RECURSIVE_TYPES = (typing.Union, typing.Optional)

is_help_request = "--help" in sys.argv or "-h" in sys.argv
is_no_cache = "--no-cache" in sys.argv
IN_HELP = is_help_request and not is_no_cache


def help(
    entity: typing.Callable,
    with_docs: bool = True,
    console=None,
    namespace: typing.Optional[str] = None,
) -> None:
    """
    Outputs help for the passed Callable
    along with all factories registered for the Callable's args.
    Optionally outputs docstrings as well.
    """
    return help_for_callable(entity, with_docs=with_docs, namespace=namespace)


def help_for_callable(
    entity: typing.Callable,
    with_docs: bool = True,
    namespace: typing.Optional[str] = None,
    display_executors: bool = True,
) -> None:
    if not callable(entity):
        CONSOLE.print(
            f"[bold cyan]Help unavailable for {entity}. Entity is not callable.[/bold cyan]"
        )
        return

    try:
        sig = inspect.signature(entity)
    except Exception:
        CONSOLE.print(
            f"[bold cyan]Help unavailable for {entity}. Failed getting signature.[/bold cyan]"
        )
        return

    params = sig.parameters

    # Render Arguments
    signature_data = [
        {
            "name": param_name,
            "type": class_to_str(param.annotation) if param.annotation != inspect.Parameter.empty else "Any",
            "default": param.default if param.default != inspect.Parameter.empty else None,
        }
        for param_name, param in params.items()
    ]
    CONSOLE.print(render_arguments(signature_data))

    # Render Entrypoint Factories
    entrypoint_factories_panel = help_for_type(
        entity,
        console=CONSOLE,
        with_docs=with_docs,
        title="Pre-loaded entrypoint factories, run with --factory",
    )
    if entrypoint_factories_panel:
        CONSOLE.print(entrypoint_factories_panel)

    # Render Parameter Factories
    _factories = {arg_name: get_underlying_types(param.annotation) for arg_name, param in params.items()}
    for arg_name, typ in _factories.items():
        if typ == inspect.Parameter.empty:
            continue
        param_factories_panel = help_for_type(
            list(typ)[0],  # TODO: Fix this properly
            with_docs=with_docs,
            console=CONSOLE,
            arg_name=arg_name,
        )
        if param_factories_panel:
            CONSOLE.print(param_factories_panel)

    # Render Executors
    if display_executors:
        from nemo_run.core.execution import LocalExecutor, SkypilotExecutor, SlurmExecutor
        from nemo_run.core.execution.base import Executor
        executors_panel = help_for_type(
            typing.Union[Executor, LocalExecutor, SlurmExecutor, SkypilotExecutor],
            CONSOLE,
            title="Registered executors",
            with_docs=with_docs,
        )
        if executors_panel:
            CONSOLE.print(executors_panel)


def help_for_type(
    entity: typing.Type,
    console: Console,
    with_docs: bool = True,
    arg_name: typing.Optional[str] = None,
    title: typing.Optional[str] = None,
):
    load_entrypoints()
    load_workspace()

    registry_details = {}
    for t in get_underlying_types(entity):
        namespace = get_type_namespace(t)
        registry_details.update(catalogue._get_all((namespace,)))

    if not registry_details:
        return None

    # Prepare factory data
    factories = [
        {
            "name": func_namespace[-1],
            "fn": f"{func.__module__}.{func.__name__}",
            "line_no": inspect.getsourcelines(func)[1],
        }
        for func_namespace, func in registry_details.items()
    ]

    # Render based on context
    if arg_name:
        return render_parameter_factories(arg_name, class_to_str(entity), factories)
    else:
        return render_entrypoint_factories(factories, title or f"Factory for {class_to_str(entity)}")


def render_entrypoint_factories(factories: list[dict[str, typing.Any]], title: str) -> Panel:
    """Render a panel with a table of entrypoint factories."""
    table = Table(show_header=False, expand=True, box=BOX_STYLE, **TABLE_STYLES)
    table.add_column("Name", style="bold magenta", ratio=1)
    table.add_column("Fn", style="bold cyan", ratio=1)
    table.add_column("Line No", style="bold green", ratio=1)
    for factory in factories:
        table.add_row(
            Text(factory["name"], style="magenta"),
            Text(factory["fn"], style="cyan"),
            Text(f"line {factory['line_no']}", style="green")
        )
    return Panel(
        table,
        title=title,
        border_style=rich_utils.STYLE_OPTIONS_PANEL_BORDER,
        title_align=rich_utils.ALIGN_OPTIONS_PANEL,
    )


def render_arguments(signature: list[dict[str, typing.Any]]) -> Panel:
    """Render a panel with a table of arguments from the function signature."""
    table = Table(show_header=False, expand=True, box=BOX_STYLE, **TABLE_STYLES)
    table.add_column("Argument", style="bold magenta", ratio=1)
    table.add_column("Type", style="bold cyan", ratio=1)
    table.add_column("Default", style="magenta", ratio=1)
    for param in signature:
        name = Text(param["name"], style="magenta")
        type_str = Text(param["type"], style="cyan")
        default = Text(str(param["default"]), style="magenta") if param["default"] is not None else Text("")
        table.add_row(name, type_str, default)
    return Panel(
        table,
        title="Arguments",
        border_style=rich_utils.STYLE_OPTIONS_PANEL_BORDER,
        title_align=rich_utils.ALIGN_OPTIONS_PANEL,
    )


def render_parameter_factories(param_name: str, param_type: str, factories: list[dict[str, typing.Any]]) -> Panel:
    """Render a panel with a table of factories for a specific parameter."""
    table = Table(show_header=False, expand=True, box=BOX_STYLE, **TABLE_STYLES)
    table.add_column("Name", style="bold magenta", ratio=1)
    table.add_column("Fn", style="bold cyan", ratio=1)
    table.add_column("Line No", style="bold green", ratio=1)
    for factory in factories:
        table.add_row(
            Text(factory["name"], style="magenta"),
            Text(factory["fn"], style="cyan"),
            Text(f"line {factory['line_no']}", style="green")
        )
    title = f"Factory for {param_name}: {param_type}"
    return Panel(
        table,
        title=title,
        border_style=rich_utils.STYLE_OPTIONS_PANEL_BORDER,
        title_align=rich_utils.ALIGN_OPTIONS_PANEL,
    )


def render_executors(executors: list[dict[str, typing.Any]]) -> Panel:
    """Render a panel with a table of registered executors."""
    table = Table(show_header=False, expand=True, box=BOX_STYLE, **TABLE_STYLES)
    table.add_column("Name", style="bold magenta", ratio=1)
    table.add_column("Fn", style="bold cyan", ratio=1)
    table.add_column("Line No", style="bold green", ratio=1)
    for executor in executors:
        table.add_row(
            Text(executor["name"], style="magenta"),
            Text(executor["fn"], style="cyan"),
            Text(f"line {executor['line_no']}", style="green")
        )
    return Panel(
        table,
        title="Registered executors",
        border_style=rich_utils.STYLE_OPTIONS_PANEL_BORDER,
        title_align=rich_utils.ALIGN_OPTIONS_PANEL,
    )


def class_to_str(class_obj):
    if hasattr(class_obj, "__or__") and hasattr(class_obj, "__args__"):
        # This is likely a types.UnionType (created with | operator)
        args = ", ".join(class_to_str(arg) for arg in class_obj.__args__)
        return f"Union[{args}]"
    elif hasattr(class_obj, "__origin__"):
        # Special handling for Optional types which are represented as Union[X, NoneType]
        if class_obj._name == "Optional":
            args = class_to_str(typing.get_args(class_obj)[0])
            return f"Optional[{args}]"
        else:
            # Get the base type
            base = class_obj.__origin__.__name__
            # Get the arguments to the type if any (e.g., the 'str' in Optional[str])
            args = ", ".join(class_to_str(arg) for arg in typing.get_args(class_obj))
            return f"{base}[{args}]"
    elif class_obj.__module__ == "builtins":
        return class_obj.__name__
    else:
        module = _get_module(class_obj)

        full_class_name = f"{module}.{class_obj.__name__}"

        if full_class_name in (
            "lightning.pytorch.core.module.LightningModule",
            "pytorch_lightning.core.module.LightningModule",
        ):
            return "[link=https://lightning.ai/docs/pytorch/latest/common/lightning_module.html]L.LightningModule[/link]"
        if full_class_name in (
            "lightning.pytorch.core.datamodule.LightningDataModule",
            "pytorch_lightning.core.datamodule.LightningDataModule",
        ):
            return "[link=https://lightning.ai/docs/pytorch/latest/api/lightning.pytorch.core.LightningDataModule.html#lightning.pytorch.core.LightningDataModule]L.LightningDataModule[/link]"
        if full_class_name == "nemo.lightning.pytorch.trainer.Trainer":
            # TODO: Add link to docs when we publish it
            return "nm.Trainer"
        if full_class_name == "nemo.lightning.pytorch.opt.base.OptimizerModule":
            return "nm.OptimizerModule"

        return full_class_name


def _get_module(class_obj) -> str:
    module = class_obj.__module__
    if module == "__main__":
        # Get the filename without extension
        main_module = sys.modules["__main__"]
        filename = os.path.basename(main_module.__file__)
        module = os.path.splitext(filename)[0]

    return module


def _get_rows_for_factories(
    factories: dict[tuple[str, ...], typing.Any], with_docs: bool = False
) -> list[list[Text | Syntax]]:
    rows = []
    if "NEMO_EDITOR" in os.environ:
        editor = os.environ["NEMO_EDITOR"]
    elif os.getenv("TERM_PROGRAM") == "vscode":
        editor = "vscode"
    else:
        editor = "file"

    for func_namespace, func in factories.items():
        module = _get_module(func)
        line_no = inspect.getsourcelines(func)[1]
        docstring = func.__doc__

        # Get the file path of the module
        try:
            file_path = inspect.getfile(func)
        except TypeError:
            file_path = None

        if file_path:
            if editor == "file":
                link = f"file://{file_path}#L{line_no}"
            else:
                link = f"{editor}://file/{file_path}:{line_no}"
            func_text = Text.from_markup(
                f"[link={link}]{module}.{func.__name__}[/link]", style="bold cyan"
            )
        else:
            func_text = Text(f"{module}.{func.__name__}" if module else "N/A", style="bold cyan")

        row: list[Text | Syntax] = [
            Text(func_namespace[-1], style="bold magenta"),
            func_text,
            Text(f"line {line_no}" if line_no else "N/A"),
        ]

        if with_docs:
            row.append(
                Syntax(
                    docstring if docstring else "No docs",
                    "python",
                )
            )

        rows.append(row)

    return rows