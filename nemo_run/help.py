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

from nemo_run.cli.api import _load_entrypoints, _load_workspace
from nemo_run.config import get_type_namespace, get_underlying_types
from nemo_run.core.frontend.console.api import CONSOLE
from nemo_run.core.frontend.console.styles import BOX_STYLE, TABLE_STYLES

RECURSIVE_TYPES = (typing.Union, typing.Optional)


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

    box_style = getattr(box, BOX_STYLE, None)

    help_for_type(
        entity,
        CONSOLE,
        title="Pre-loaded entrypoint factories, run with --factory",
        with_docs=with_docs,
    )

    table = Table(
        highlight=True,
        show_header=False,
        expand=True,
        box=box_style,
        **TABLE_STYLES,
    )
    table.add_column("argument", style=None, ratio=1)
    table.add_column("type", style=None, ratio=1)
    table.add_column("default", style=None, ratio=1)

    try:
        sig = inspect.signature(entity)
    except Exception:
        CONSOLE.print(
            f"[bold cyan]Help unavailable for {entity}. Failed getting signature.[/bold cyan]"
        )
        return

    params = sig.parameters

    for arg_name, param in params.items():
        arg_text = Text(arg_name, style="bold magenta")
        type_text = Text.from_markup(class_to_str(param.annotation), style="bold cyan")

        default_value_text = Text("")
        default_value = param.default
        if default_value != inspect._empty:
            default_value_text = Text(str(default_value), style="magenta")

        table.add_row(arg_text, type_text, default_value_text)

    CONSOLE.print(
        Panel(
            table,
            title="Arguments",
            border_style=rich_utils.STYLE_OPTIONS_PANEL_BORDER,
            title_align=rich_utils.ALIGN_OPTIONS_PANEL,
        )
    )

    _factories = {}
    for arg_name, param in params.items():
        _factories[arg_name] = get_underlying_types(param.annotation)

    for arg_name, typ in _factories.items():
        if typ == inspect._empty:
            continue

        help_for_type(
            list(typ)[0],  # TODO: Fix this properly
            with_docs=with_docs,
            console=CONSOLE,
            arg_name=arg_name,
        )

    if display_executors:
        from nemo_run.core.execution import LocalExecutor, SkypilotExecutor, SlurmExecutor
        from nemo_run.core.execution.base import Executor

        help_for_type(
            typing.Union[Executor, LocalExecutor, SlurmExecutor, SkypilotExecutor],
            CONSOLE,
            title="Registered executors",
            with_docs=with_docs,
        )


def help_for_type(
    entity: typing.Type,
    console: Console,
    with_docs: bool = True,
    arg_name: typing.Optional[str] = None,
    title: typing.Optional[str] = None,
):
    _load_entrypoints()
    _load_workspace()

    registry_details = {}
    for t in get_underlying_types(entity):
        namespace = get_type_namespace(t)
        registry_details.update(catalogue._get_all((namespace,)))

    if not registry_details:
        return

    box_style = getattr(box, BOX_STYLE, None)

    table_registry = Table(
        highlight=False,
        show_header=False,
        expand=True,
        box=box_style,
        **TABLE_STYLES,
    )
    table_registry.add_column("Name", style="cyan", ratio=1)
    table_registry.add_column("Fn", style="bold cyan", ratio=1)
    table_registry.add_column("Line No", style="bold green", ratio=1)
    if with_docs:
        table_registry.add_column("Docstring", style="bold cyan", ratio=2)

    rows = _get_rows_for_factories(factories=registry_details, with_docs=with_docs)
    for row in rows:
        table_registry.add_row(*row)

    factory_name = class_to_str(entity)
    if arg_name:
        factory_name = f"{arg_name}: {factory_name}"

    console.print(
        Panel(
            table_registry,
            title=title or f"Factory for {factory_name}",
            border_style=rich_utils.STYLE_OPTIONS_PANEL_BORDER,
            title_align=rich_utils.ALIGN_OPTIONS_PANEL,
        )
    )


def class_to_str(class_obj):
    if hasattr(class_obj, "__origin__"):
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


def _get_module(class_obj) -> str:
    module = class_obj.__module__
    if module == "__main__":
        # Get the filename without extension
        main_module = sys.modules["__main__"]
        filename = os.path.basename(main_module.__file__)
        module = os.path.splitext(filename)[0]

    return module
