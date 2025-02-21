import os
from dataclasses import dataclass, field
from typing import Optional, Type

from nemo_run.config import ConfigurableMixin, Script


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
class SlurmRay(Launcher):
    """
    Transforms a provided cmd into a Ray launcher bash script for SlurmExecutor.
    The Ray launcher script sets up a Ray cluster on Slurm nodes, with the head node starting Ray head
    and executing the provided command. Worker nodes start Ray and wait.
    """

    port: int = 6379

    def transform(self, cmd: list[str]) -> Optional[Script]:
        """
        Transforms the provided cmd into a Ray launcher bash script for SlurmExecutor.
        """
        cmd_to_run = " ".join(cmd)
        # Build the Ray launcher bash script. Braces in shell variables are escaped as {{ and }}
        ray_script = f"""
# Check that a command was provided.
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <command>"
    exit 1
fi

# Function to start the Ray head node.
start_head() {{
    echo "Starting Ray head node on ${{HEAD_IP}}"
    ray start --head --node-ip-address=${{HEAD_IP}} --port={self.port}
    export RAY_ADDRESS="${{HEAD_IP}}:{self.port}"
}}

# Function to start a Ray worker node.
start_worker() {{
    # Obtain the head node's hostname from the SLURM_NODELIST.
    echo "Starting Ray worker node. Connecting to head ${{HEAD_IP}}"
    ray start --address=${{HEAD_IP}}:{self.port}
}}

# If this is the head node, start the head; otherwise, start a worker.
if [ -z "$SLURM_NODEID" ] || [ "$SLURM_NODEID" == "0" ]; then
    start_head
else
    start_worker
fi

# Only the head node executes the command.
if [ -z "$SLURM_NODEID" ] || [ "$SLURM_NODEID" == "0" ]; then
    echo "Running command: {cmd_to_run}"
    # Use eval so the given command is executed with its arguments.
    eval "{cmd_to_run}"
    echo "Command finished. Shutting down Ray on head node."
    ray stop
    # Optionally, you could touch a file to signal the worker nodes to shut down.
fi

# For worker nodes, simply wait so that Ray stays active.
if [ -n "$SLURM_NODEID" ] && [ "$SLURM_NODEID" != "0" ]; then
    echo "Worker node running. Waiting for the Ray head to finish."
    while true; do
        sleep 15
    done
fi
"""
        # Return a new Script object with the inline content
        return Script(inline=ray_script)


LAUNCHER_MAP: dict[str, Type[Launcher]] = {
    "torchrun": Torchrun,
    "ft": FaultTolerance,
    "slurm_ray": SlurmRay,
}
