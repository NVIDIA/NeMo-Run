# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated

import typer
from rich.console import Console

from nemo_run.cli.core.workspace import sync as workspace_sync


def sync(
    verbose: Annotated[bool, typer.Option(help="Show detailed progress information")] = True,
    no_cache: Annotated[bool, typer.Option(help="Force a full rebuild of the cache")] = False,
):
    """
    Synchronize the NeMo Run workspace and CLI cache.
    
    Ensures all CLI commands, factories, and workspace files are properly cached
    for optimal performance. Run this command after installing new packages or
    making changes to your workspace.
    """
    # Call the actual sync implementation from workspace.py
    success = workspace_sync(verbose=verbose, no_cache=no_cache)
    
    # If the sync was not successful and we're not in verbose mode, show a simple error
    if not success and not verbose:
        console = Console()
        console.print("[bold red]Error:[/bold red] Failed to synchronize workspace cache.")
    
    return 0 if success else 1
