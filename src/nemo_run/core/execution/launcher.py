import os
import pathlib
from dataclasses import dataclass, field
from typing import Optional, Type

import jinja2

from nemo_run.config import ConfigurableMixin, Script
from nemo_run.core.execution.utils import fill_template


@dataclass(kw_only=True)
class Launcher(ConfigurableMixin):
    nsys_profile: bool = False
    nsys_folder: str = "nsys_profile"
    nsys_trace: list[str] = field(default_factory=lambda: ["nvtx", "cuda"])

    def get_nsys_prefix(self, profile_dir: str) -> Optional[list[str]]:
        """Make a command prefix for nsys profiling"""
        if self.nsys_profile:
            profile_out_path = os.path.join(profile_dir, self.nsys_folder)
            args = [
                "profile",
                "-s",
                "none",
                "-t",
                ",".join(self.nsys_trace),
                "-o",
                f"{profile_out_path}/profile_%p",
                "--force-overwrite",
                "true",
                "--capture-range=cudaProfilerApi",
                "--capture-range-end=stop",
                "--cuda-graph-trace=node",
            ]
            return args

    def transform(self, cmd: list[str]) -> Optional[Script]: ...


@dataclass(kw_only=True)
class Torchrun(Launcher):
    rdzv_backend: str = "c10d"
    rdzv_port: int = 29500


@dataclass(kw_only=True)
class FaultTolerance(Launcher):
    cfg_path: str = ""
    finished_flag_file: str = ""
    job_results_file: str = ""
    rdzv_backend: str = "c10d"
    rdzv_port: int = 29500
    workload_check_interval: Optional[float] = None
    initial_rank_heartbeat_timeout: Optional[float] = None
    rank_heartbeat_timeout: Optional[float] = None
    rank_termination_signal: Optional[str] = None
    log_level: Optional[str] = None
    max_restarts: Optional[int] = None


@dataclass(kw_only=True)
class SlurmTemplate(Launcher):
    """
    A generic launcher that uses Jinja2 templates to wrap commands.
    The template can be provided either as inline content or as a path to a template file.
    """

    template_path: Optional[str] = None
    template_inline: Optional[str] = None
    template_vars: dict = field(default_factory=dict)

    def __post_init__(self):
        # Ensure at least one template source is provided
        if not self.template_path and not self.template_inline:
            raise ValueError("Either template_path or template_inline must be provided")

    def get_template_content(self) -> str:
        """
        Get the template content either from the file or inline content.
        """
        if self.template_inline:
            return self.template_inline

        if self.template_path:
            # Check if the path is absolute
            path = pathlib.Path(self.template_path)
            if path.is_absolute():
                # Read the template from the absolute path
                with open(path, "r") as f:
                    return f.read()
            else:
                # Use the template from the templates directory
                template_dir = os.path.join(os.path.dirname(__file__), "templates")
                template_path = os.path.join(template_dir, self.template_path)
                if os.path.exists(template_path):
                    with open(template_path, "r") as f:
                        return f.read()
                else:
                    raise FileNotFoundError(f'Template "{self.template_path}" does not exist.')

        # This should not happen due to the check in __post_init__
        raise ValueError("No template available")

    def render_template(self, cmd: list[str]) -> str:
        """
        Render the template with the command and additional variables.
        """
        # If using a template file from the templates directory
        if self.template_path and not os.path.isabs(self.template_path):
            # Create variables dictionary with command and additional variables
            vars_dict = {"command": " ".join(cmd), **self.template_vars}
            # Use the project's template rendering utility
            return fill_template(self.template_path, vars_dict)

        # If using inline template or absolute path template
        template_content = self.get_template_content()
        template = jinja2.Template(template_content)

        # Create variables dictionary with command and additional variables
        vars_dict = {"command": " ".join(cmd), **self.template_vars}

        # Render the template
        return template.render(**vars_dict)

    def transform(self, cmd: list[str]) -> Optional[Script]:
        """
        Transform the command using the template.
        """
        rendered_script = self.render_template(cmd)
        return Script(inline=rendered_script)


@dataclass(kw_only=True)
class SlurmRay(SlurmTemplate):
    """
    Transforms a provided cmd into a Ray launcher bash script for SlurmExecutor.
    The Ray launcher script sets up a Ray cluster on Slurm nodes, with the head node starting Ray head
    and executing the provided command. Worker nodes start Ray and wait.
    """

    port: int = 6379

    def __post_init__(self):
        # Set the template path to the Ray template
        self.template_path = "ray.sh.j2"
        # Add port to template variables
        self.template_vars["port"] = self.port
        # Call parent's post_init
        super().__post_init__()


LAUNCHER_MAP: dict[str, Type[Launcher]] = {
    "torchrun": Torchrun,
    "ft": FaultTolerance,
    "slurm_ray": SlurmRay,
    "slurm_template": SlurmTemplate,
}
