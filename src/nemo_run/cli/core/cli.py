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
import os
import sys
from typing import Any, Dict, Optional

import importlib_metadata as metadata
import typer
from rich.logging import RichHandler
from typer import rich_utils, Typer
from typer.core import TyperCommand, TyperGroup

from nemo_run.cli import devspace as devspace_cli
from nemo_run.cli import experiment as experiment_cli
from nemo_run.cli.cache import ImportTracker
from nemo_run.lazy import LazyEntrypoint


def configure_global_options(
    app: Typer,
    rich_exceptions=False,
    rich_traceback=True,
    rich_locals=True,
    rich_theme=None,
    verbose=False,
):
    configure_logging(verbose)

    app.pretty_exceptions_enable = rich_exceptions
    app.pretty_exceptions_short = False if rich_exceptions else rich_traceback
    app.pretty_exceptions_show_locals = True if rich_exceptions else rich_locals

    if rich_theme:
        from rich.traceback import Traceback

        Traceback.theme = rich_theme


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



def add_typer_nested(typer: Typer, to_add: dict):
    for key, value in to_add.items():
        if isinstance(value, dict):
            nested = get_or_add_typer(typer, name=key)
            add_typer_nested(nested, value)  # type: ignore
        elif hasattr(value, "cli"):
            value.cli(typer)
        else:
            raise ValueError(f"Cannot add {value} to typer")


def get_or_add_typer(typer: Typer, name: str, help=None, **kwargs):
    for r in typer.registered_groups:
        if name == r.name:
            return r.typer_instance

    help = help or name
    help = f"[Module] {help}"

    output = Typer(pretty_exceptions_enable=False)
    typer.add_typer(
        output,
        name=name,
        help=help,
        context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
        **kwargs,
    )

    return output


def add_global_options(app: Typer):
    @app.callback()
    def global_options(
        verbose: bool = Option(False, "-v", "--verbose"),
        rich_exceptions: bool = typer.Option(
            False, "--rich-exceptions/--no-rich-exceptions", help="Enable rich exception formatting"
        ),
        rich_traceback: bool = typer.Option(
            False,
            "--rich-traceback-short/--rich-traceback-full",
            help="Control traceback verbosity",
        ),
        rich_locals: bool = typer.Option(
            True,
            "--rich-show-locals/--rich-hide-locals",
            help="Toggle local variables in exceptions",
        ),
        rich_theme: Optional[str] = typer.Option(
            None, "--rich-theme", help="Color theme (dark/light/monochrome)"
        ),
    ):
        _configure_global_options(
            app, rich_exceptions, rich_traceback, rich_locals, rich_theme, verbose
        )

    return global_options


class GeneralCommand(TyperGroup):
    def format_usage(self, ctx, formatter) -> None:
        pieces = self.collect_usage_pieces(ctx) + ["[ARGUMENTS]"]
        formatter.write_usage(ctx.command_path, " ".join(pieces))

    def format_help(self, ctx, formatter):
        out = rich_utils.rich_format_help(
            obj=self,
            ctx=ctx,
            markup_mode=self.rich_markup_mode,
        )

        # TODO: Check if args are passed in to provide help for
        # print(sys.argv[1:])

        return out
