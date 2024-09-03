import nemo_run as run
from nemo_run.run.task import ExperimentTask
from dataclasses import dataclass
from time import sleep
import os

from nemo import lightning as nl
from nemo.collections.llm.recipes.precision.mixed_precision import bf16_mixed_plugin

@dataclass
class TaskStatus:
    UNSUBMITTED = "UNSUBMITTED"
    SUBMITTED = "SUBMITTED"
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    UNKNOWN = "UNKNOWN"

def get_exp_from_id(exp_id: str) -> run.Experiment:
    return run.Experiment.from_id(exp_id)

def get_slurm_job_id(task: ExperimentTask) -> str:
    try:
        _, _, path_str = task.handle.partition("://")
        path = path_str.split("/")
        app_id = path[1]
    except Exception:
        app_id = "-1"

    return app_id

def get_task_status(exp: run.Experiment) -> str:
    try:
        return str(exp.tasks[0].status(runner=exp._runner))
    except:
        return "UNKNOWN"

def track_task(exp_id: str, slurm_job_id: str) -> None:
    exp = get_exp_from_id(exp_id)
    while (status:=get_task_status(exp)) not in [TaskStatus.SUCCEEDED, TaskStatus.CANCELLED, TaskStatus.FAILED]:
        print(f"{slurm_job_id=}, {status=}")
        sleep(10)
    else:
        status = get_task_status(exp)
        print(f"{slurm_job_id=} completed with status: {status}")
        if status == TaskStatus.SUCCEEDED:
            exit(0)
        else:
            exit(1)

def write_task_info_to_disk(
        log_dir: str,
        filename = "task_info.log",
        exp_id: str="",
        slurm_job_id: str="-1"
        ) -> None:
    assert os.path.isdir(log_dir),NotADirectoryError(f"{log_dir}")

    with open(f"{log_dir}/{filename}", "w") as fp:
        fp.write(f"{exp_id=}\n")
        fp.write(f"{slurm_job_id=}\n")

def bf16_with_fp8_mixed_plugin() -> run.Config[nl.MegatronMixedPrecision]:
    cfg = bf16_mixed_plugin()
    cfg.fp8 = 'hybrid'
    cfg.fp8_margin = 0
    cfg.fp8_interval = 1
    cfg.fp8_amax_history_len = 1024
    cfg.fp8_amax_compute_algo = "max"
    return cfg
