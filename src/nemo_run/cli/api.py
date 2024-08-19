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

import logging
import sys
from typing import TypeVar

from rich.logging import RichHandler
from typer import Option, Typer
from typing_extensions import ParamSpec
import importlib_metadata as metadata

from nemo_run.api import list_tasks
from nemo_run.cli import devspace as devspace_cli
from nemo_run.cli import experiment as experiment_cli
from nemo_run.cli.dynamic_cli import GeneralCommand

Params = ParamSpec("Params")
ReturnType = TypeVar("ReturnType")


def _add_typer_nested(typer: Typer, to_add: dict):
    for key, value in to_add.items():
        if isinstance(value, dict):
            nested = _get_or_add_typer(typer, name=key)
            _add_typer_nested(nested, value)  # type: ignore
        elif hasattr(value, "cli"):
            value.cli(typer)
        else:
            raise ValueError(f"Cannot add {value} to typer")


def _get_or_add_typer(typer: Typer, name: str, help=None, **kwargs):
    for r in typer.registered_groups:
        if name == r.name:
            return r.typer_instance

    help = help or name
    help = f"[Module] {help}"

    output = Typer()
    typer.add_typer(
        output,
        name=name,
        help=help,
        context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
        **kwargs,
    )

    return output


def create_cli(
    add_verbose_callback: bool = True,
) -> Typer:
    app: Typer = Typer()
    entrypoints = metadata.entry_points().select(group="run.factories")
    metadata.entry_points().select(group="run.factories")
    for ep in entrypoints:
        _get_or_add_typer(app, name=ep.name)

    if len(sys.argv) > 1 and sys.argv[1] in entrypoints.names:
        _add_typer_nested(app, list_tasks())

    app.add_typer(
        devspace_cli.create(),
        name="devspace",
        help="[Module] Manage devspaces",
        cls=GeneralCommand,
        context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    )
    app.add_typer(
        experiment_cli.create(),
        name="experiment",
        help="[Module] Manage Experiments",
        cls=GeneralCommand,
        context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    )

    if add_verbose_callback:
        app.callback()(global_options)

    return app


def global_options(verbose: bool = Option(False, "-v", "--verbose")):
    configure_logging(verbose)


def configure_logging(verbose: bool):
    handler = RichHandler(
        rich_tracebacks=True,
        show_path=False,
    )

    logger = logging.getLogger("torchx")
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    for hdlr in logger.handlers[:]:
        logger.removeHandler(hdlr)
    logger.addHandler(handler)


if __name__ == "__main__":
    app = create_cli()
    app()
