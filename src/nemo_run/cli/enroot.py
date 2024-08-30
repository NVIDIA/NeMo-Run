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

import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

import typer
from rich.panel import Panel
from rich.progress import Progress
from rich.syntax import Syntax
from rich.table import Table

from nemo_run.core.frontend.console.api import CONSOLE


class EnrootError(Exception):
    """Base exception for Enroot-related errors."""

    pass


class EnrootNotInstalledError(EnrootError):
    """Raised when Enroot is not installed or not found in PATH."""

    def __init__(self):
        super().__init__(
            "Enroot is not installed or not found in your PATH. "
            "Please install Enroot and ensure it's accessible from your command line. "
            "Visit https://github.com/NVIDIA/enroot for installation instructions."
        )


class SlurmNotInstalledError(EnrootError):
    """Raised when SLURM is not installed or not found in PATH."""

    def __init__(self):
        super().__init__(
            "SLURM is not installed or not found in your PATH. "
            "Please ensure SLURM is properly installed and configured on your system. "
            "Contact your system administrator if you believe this is an error."
        )


class SlurmSubmissionError(EnrootError):
    """Raised when there's an error submitting a job to SLURM."""

    def __init__(self, message):
        super().__init__(
            f"Failed to submit job to SLURM: {message}\n"
            "Please check your SLURM configuration and ensure you have the necessary permissions."
        )


class EnrootImportError(EnrootError):
    """Raised when there's an error during Enroot import."""

    def __init__(self, message):
        super().__init__(
            f"Error during Enroot import: {message}\n"
            "Please check the image URL and your Enroot configuration."
        )


app = typer.Typer(
    help="""
Asynchronously fetch docker image from registry into a Singularity/Enroot compatible image file.

The output on stdout is the SLURM job ID, which can be used to make another sbatch wait for the
image fetch to finish (use '--dependency=afterok<jobid>').
"""
)


def print_enroot_notes():
    """Print notes about Enroot usage."""
    notes = """
Note that enroot is somewhat picky about image URLs. If you are having trouble, check:
1. You do NOT include the "https://" or "docker://" part of the URL
2. You do NOT include the ":5005" in the address
3. Enroot will get confused if you ask for a registry server not listed in your
   ~/.config/enroot/.credentials file. It will think that your registry server
   name is part of a path on docker://docker.io/. Make sure that your
   ~/.config/enroot/.credentials file contains lines like:

   machine gitlab-master.nvidia.com login <your-id> password <your-registry-read-token>
   machine nvcr.io login $oauthtoken password <your-ngc-token>
   machine authn.nvidia.com login $oauthtoken password <your-ngc-token>
    """
    return notes


def _check_dependencies():
    """Check if required dependencies (Enroot and SLURM) are installed."""
    if not shutil.which("enroot"):
        raise EnrootNotInstalledError()
    if not shutil.which("sbatch") or not shutil.which("sacctmgr"):
        raise SlurmNotInstalledError()


def _get_default_account() -> str:
    """Get the user's default SLURM account."""
    try:
        result = subprocess.run(
            [
                "sacctmgr",
                "-nP",
                "show",
                "assoc",
                f"where user={subprocess.getoutput('whoami')}",
                "format=account",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        accounts = result.stdout.strip().split("\n")
        if not accounts:
            raise SlurmSubmissionError("No SLURM accounts found for the current user.")
        return accounts[0]
    except subprocess.CalledProcessError as e:
        raise SlurmSubmissionError(f"Failed to retrieve SLURM account: {e.stderr.strip()}")


def print_command_summary(sbatch_cmd: List[str], srun_script: str):
    """Print a summary of the commands to be executed."""
    CONSOLE.print("\n[bold cyan]SLURM command:[/bold cyan]")
    CONSOLE.print(Syntax(" ".join(sbatch_cmd), "bash", theme="monokai", line_numbers=True))

    CONSOLE.print("\n[bold cyan]srun script:[/bold cyan]")
    CONSOLE.print(Syntax(srun_script, "bash", theme="monokai", line_numbers=True))


def validate_image_url(url: str) -> str:
    """Validate the image URL format."""
    parsed = urlparse(url)
    if not parsed.netloc or not parsed.path:
        raise typer.BadParameter("Invalid image URL format. Expected: registry.com/image:tag")
    return url


def validate_file_path(path: Path) -> Path:
    """Validate file path and create parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@app.command()
def import_image(
    image_url: str = typer.Argument(
        ..., help="Docker image URL to import", callback=validate_image_url
    ),
    account: str = typer.Option(None, "--account", "-A", help="SLURM account"),
    job_name: str = typer.Option(None, "--job-name", "-J", help="SLURM job name"),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file path", callback=validate_file_path
    ),
    partition: Optional[str] = typer.Option(None, "--partition", "-p", help="SLURM partition"),
    time: str = typer.Option("00:15:00", "--time", "-t", help="SLURM time limit"),
    slurm_extra: List[str] = typer.Option(
        None, "--slurm-extra", "-x", help="Extra SLURM arguments"
    ),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug mode"),
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Print commands without executing"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
):
    """
    Import a Docker image and convert it to a Singularity/Enroot compatible image file using SLURM.

    Example: nemorun enroot import nvcr.io/nvidia/nemo:24.05 -o output.sqsh
    """
    try:
        _check_dependencies()

        account = account or _get_default_account()
        job_name = job_name or str(typer.get_app_dir("nemorun"))
        slurm_extra = slurm_extra or []

        sbatch_cmd = [
            "sbatch",
            "--nodes=1",
            f"--job-name={account}-mksqsh.:{job_name}",
            f"--time={time}",
            f"--account={account}",
        ]

        if partition:
            sbatch_cmd.append(f"--partition={partition}")
        if not debug:
            sbatch_cmd.append("--output=/dev/null")
        sbatch_cmd.extend(slurm_extra)

        output_arg = f'--output="{output}"' if output else ""
        srun_script = f"""#!/bin/bash
set -x
srun --mpi=none --ntasks-per-node=1 \\
     enroot import {output_arg} \\
            "docker://{image_url}"
"""

        if dry_run:
            print_command_summary(sbatch_cmd, srun_script)
        else:
            with Progress() as progress:
                task = progress.add_task("[cyan]Submitting SLURM job...", total=1)
                result = subprocess.run(
                    sbatch_cmd, input=srun_script, text=True, capture_output=True, check=True
                )
                progress.update(task, advance=1)
            job_id = result.stdout.strip().split()[-1]
            CONSOLE.print(f"[bold green]Job submitted successfully. Job ID: {job_id}[/bold green]")

        if verbose:
            CONSOLE.print("[bold cyan]Verbose output:[/bold cyan]")
            CONSOLE.print(f"Image URL: {image_url}")
            CONSOLE.print(f"Output file: {output}")
            CONSOLE.print(f"SLURM account: {account}")
            CONSOLE.print(f"SLURM partition: {partition}")
            CONSOLE.print(f"SLURM time limit: {time}")

        CONSOLE.print(Panel(print_enroot_notes(), title="Enroot Usage Notes", expand=False))

    except EnrootError as e:
        CONSOLE.print(Panel(str(e), title="[bold red]Error[/bold red]", expand=False))
        if verbose:
            CONSOLE.print_exception()
        raise typer.Exit(1)
    except subprocess.CalledProcessError as e:
        raise EnrootImportError(e.stderr.strip())
    except Exception as e:
        CONSOLE.print(
            Panel(
                f"An unexpected error occurred: {str(e)}",
                title="[bold red]Error[/bold red]",
                expand=False,
            )
        )
        if debug:
            CONSOLE.print_exception()
        raise typer.Exit(1)


@app.command()
def squash(
    input_file: Path = typer.Argument(
        ..., help="Input image file to squash", callback=validate_file_path
    ),
    output: Path = typer.Option(
        ..., "--output", "-o", help="Output file path", callback=validate_file_path
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Force overwrite of existing file"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug mode"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
):
    """
    Squash a Singularity/Enroot image file to reduce its size.

    Example: nemorun enroot squash input.sqsh -o output.sqsh
    """
    try:
        _check_dependencies()

        cmd = ["enroot", "convert"]
        if force:
            cmd.append("--force")
        cmd.extend(["-o", str(output), str(input_file)])

        if verbose:
            CONSOLE.print(f"[bold cyan]Running command: {' '.join(cmd)}[/bold cyan]")

        with Progress() as progress:
            task = progress.add_task("[cyan]Squashing image...", total=1)
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            progress.update(task, advance=1)

        CONSOLE.print(f"[bold green]{result.stdout.strip()}[/bold green]")
        CONSOLE.print("[bold green]Image squashed successfully.[/bold green]")

        if verbose:
            CONSOLE.print(f"[bold cyan]Input file: {input_file}[/bold cyan]")
            CONSOLE.print(f"[bold cyan]Output file: {output}[/bold cyan]")
            CONSOLE.print(f"[bold cyan]Force overwrite: {force}[/bold cyan]")

    except subprocess.CalledProcessError as e:
        CONSOLE.print(f"[bold red]Error during Enroot squash: {e.stderr.strip()}[/bold red]")
        if verbose:
            CONSOLE.print_exception()
        raise typer.Exit(1)
    except Exception as e:
        CONSOLE.print(f"[bold red]An unexpected error occurred: {str(e)}[/bold red]")
        if debug:
            CONSOLE.print_exception()
        raise typer.Exit(1)


@app.command()
def list_images(
    directory: Path = typer.Option(
        Path.home() / ".local" / "share" / "enroot",
        "--directory",
        "-d",
        help="Directory to search for Enroot images",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
):
    """
    List available Enroot images.

    Example: nemorun enroot list
    """
    try:
        images = [f for f in directory.glob("*.sqsh") if f.is_file()]

        if not images:
            CONSOLE.print("[yellow]No Enroot images found.[/yellow]")
            return

        table = Table(title="Enroot Images")
        table.add_column("Image Name", style="cyan")
        table.add_column("Size", style="magenta")
        table.add_column("Last Modified", style="green")

        for image in images:
            size = f"{image.stat().st_size / (1024 * 1024):.2f} MB"
            last_modified = image.stat().st_mtime
            last_modified_str = datetime.fromtimestamp(last_modified).strftime("%Y-%m-%d %H:%M:%S")
            table.add_row(image.name, size, last_modified_str)

        CONSOLE.print(table)

        if verbose:
            CONSOLE.print(f"[bold cyan]Search directory: {directory}[/bold cyan]")
            CONSOLE.print(f"[bold cyan]Total images found: {len(images)}[/bold cyan]")

    except Exception as e:
        CONSOLE.print(f"[bold red]Error listing Enroot images: {str(e)}[/bold red]")
        if verbose:
            CONSOLE.print_exception()
        raise typer.Exit(1)


@app.callback()
def main(ctx: typer.Context):
    """
    Enroot operations for NeMo.

    {notes}
    """
    if ctx.invoked_subcommand is None:
        CONSOLE.print(ctx.get_help())


main.__doc__ = main.__doc__.format(notes=print_enroot_notes())


def create() -> typer.Typer:
    app = typer.Typer()

    app.command(
        "import",
        context_settings={"allow_extra_args": False},
    )(import_image)

    app.command(
        "squash",
        context_settings={"allow_extra_args": False},
    )(squash)

    app.command(
        "list",
        context_settings={"allow_extra_args": False},
    )(list_images)

    @app.callback()
    def callback():
        """
        Enroot operations for NeMo.

        {notes}
        """

    callback.__doc__ = callback.__doc__.format(notes=print_enroot_notes())

    return app


if __name__ == "__main__":
    app()
