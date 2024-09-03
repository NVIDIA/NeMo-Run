import nemo_run as run
import torch
import pytorch_lightning as pl
from nemo import lightning as nl
from nemo.collections import llm
import torch.nn.functional as F
import os

from nemo.lightning.pytorch.optim import (
    OptimizerModule,
    MegatronOptimizerModule,
    CosineAnnealingScheduler,
)
from megatron.core.optimizer import OptimizerConfig

from nemo.lightning.pytorch.callbacks import PreemptionCallback
from nemo.utils.exp_manager import TimingCallback
from typing import Optional
from nemo.collections.llm.gpt.model.mixtral import MixtralConfig8x7B, MixtralModel

from executors.base import get_base_executor
import fiddle as fdl
import fiddle._src.experimental.dataclasses as fdl_dc
from pytorch_lightning.loggers import TensorBoardLogger
from nemo.collections.llm.gpt.data.mock import MockDataModule
from nemo.collections.common.tokenizers.huggingface import AutoTokenizer
from nemo.collections.llm.recipes.precision.mixed_precision import bf16_mixed_plugin

import sys

# Add the 'Project' directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import track_task, get_slurm_job_id, write_task_info_to_disk, bf16_with_fp8_mixed_plugin

LOCAL_TUNNEL = True

@run.cli.factory
def eos_executor(
    nodes=8,
    devices=8,
    ft=False,
    retries=0,
    time="00:20:00",
    container_image="",
    job_dir=f"/lustre/fsw/coreai_dlalgo_llm/malayn/result_logs/nemo2.0",
) -> run.Config[run.SlurmExecutor]:
    # today_date = datetime.now(timezone("US/Eastern")).strftime("%Y-%m-%d")
    # use sqsh image for faster loading
    if not container_image:
        container_image = "/lustre/fsw/coreai_dlalgo_llm/containers/malayn_nemo_sdk_24-08-06.sqsh"

    _executor = get_base_executor(
        cluster="eos",
        user="malayn",
        nodes=nodes,
        devices=devices,
        container_image=container_image,
        custom_mounts=[
            f"/lustre/fsw/coreai_dlalgo_llm/malayn/result_logs/nemo2.0:/output",
            "/lustre/fsw/coreai_dlalgo_llm/malayn/git_repos/NeMo:/opt/NeMo",
            "/lustre/fsw/coreai_dlalgo_llm/malayn/git_repos/NeMo-Run:/workspaces/NeMo-Run",
        ],
        use_ft=ft,
        retries=retries,
        time=time,
        job_dir=job_dir,
    )

    _executor.srun_args = ["--mpi=pmix"]

    if LOCAL_TUNNEL:
        _executor.tunnel = run.LocalTunnel(job_dir=f"/lustre/fsw/coreai_dlalgo_llm/malayn/result_logs/nemo2.0")

    return fdl.cast(run.Config, fdl_dc.convert_dataclasses_to_configs(_executor, allow_post_init=True))

@run.cli.factory
def model() -> run.Config[pl.LightningModule]:
    model_obj = run.Config(MixtralModel, config=run.Config(MixtralConfig8x7B))
    model_obj.config.apply_query_key_layer_scaling=True
    model_obj.config.bias_activation_fusion=False
    model_obj.config.bias_dropout_fusion=False
    model_obj.config.share_embeddings_and_output_weights=False
    model_obj.config.gated_linear_unit=False

    return model_obj

@run.cli.factory
def mixtral_trainer(num_nodes=8, callbacks=None) -> run.Config[nl.Trainer]:
    callbacks=[
                run.Config(TimingCallback),
                run.Config(PreemptionCallback),
            ]

    trainer = run.Config(
        nl.Trainer,
        num_nodes=num_nodes,
        devices=8,
        accelerator="gpu",
        enable_checkpointing=False,
        use_distributed_sampler=False,
        max_steps=10,
        log_every_n_steps=1,
        val_check_interval=12,
        limit_val_batches=1,
        limit_test_batches=1,
        accumulate_grad_batches=1,
        gradient_clip_val=1.0,
        strategy=run.Config(nl.MegatronStrategy),
        plugins=bf16_mixed_plugin(),
        callbacks=callbacks
    )

    return trainer

@run.cli.factory
def mixtral_distributed_fused_adam_cosine_annealing() -> run.Config[OptimizerModule]:
    opt_cfg = run.Config(
        OptimizerConfig,
        optimizer="adam",
        lr=0.0001,
        weight_decay=0.1,
        bf16=True,
        params_dtype=torch.bfloat16,
        adam_beta1=0.9,
        adam_beta2=0.95,
        use_distributed_optimizer=True,
        overlap_grad_reduce=True,
        overlap_param_gather=True,
    )

    sched = run.Config(
        CosineAnnealingScheduler,
        warmup_steps=500,
        constant_steps=0,
        min_lr=1.0e-05,
    )

    return run.Config(MegatronOptimizerModule, opt_cfg, sched)

@run.cli.factory
def train_data(
    cluster: str = "eos",
    gbs: int = 256,
    mbs: int = 1,
    seq_length: int = 4096,
    tokenizer_path: Optional[str] = None,
    base_data_path: Optional[str] = None,
    index_mapping_dir: Optional[str] = None,
) -> run.Config[pl.LightningDataModule]:
    """
    paths: Path | List | Dict[str, List],
    seq_length: int = 2048,
    tokenizer: Optional["TokenizerSpec"] = None,
    micro_batch_size: int = 4,
    global_batch_size: int = 8,
    rampup_batch_size: Optional[List[int]] = None,
    num_workers: int = 8,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    reset_position_ids: bool = False,
    reset_attention_mask: bool = False,
    eod_mask_loss: bool = False,
    seed: int = 1234,
    split: str = "900,50,50",
    index_mapping_dir: Optional[str] = None,
    """
    return run.Config(
        MockDataModule,
        tokenizer=run.Config(AutoTokenizer, pretrained_model_name="meta-llama/Meta-Llama-3-8B", use_fast=True),
        seq_length=seq_length,
        global_batch_size=gbs,
        micro_batch_size=mbs
        )

@run.cli.factory
def set_log_dir(job_dir, exp_name) -> run.Config[llm.default_log]:
    log_dir_obj = llm.default_log(ckpt_dir=job_dir, name=exp_name)
    log_dir_obj.ckpt.save_last=False
    log_dir_obj.ckpt.save_top_k=1

    log_dir_obj.tensorboard = run.Config(
        TensorBoardLogger,
        save_dir="/output",
        name=exp_name,
        )

    return log_dir_obj

@run.cli.entrypoint(
        require_confirmation=False,
        type="experiment",
)
def train_model(
        ctx: run.cli.RunContext = run.cli.RunContext(name="pretrain_mixtral_8x7b_bf16_ctx"),
        _executor = eos_executor(),
        model_cfg=model(),
        trainer=mixtral_trainer(),
        optim=mixtral_distributed_fused_adam_cosine_annealing(),
        data = train_data(
            gbs=256,
            mbs=1,
            seq_length=model().config.seq_length,
            index_mapping_dir=None,
            tokenizer_path=None,
            base_data_path=None,
        ),
        log=set_log_dir(
            job_dir=eos_executor().tunnel.job_dir,
            exp_name="pretrain_mixtral_8x7b_bf16"
            ),
        resume=llm.default_resume(),
        detach=True,
        track=False,
        exp_name="pretrain_mixtral_8x7b_bf16",
    ):
    exp_id, slurm_job_id = "", "-1"

    with run.Experiment(title=exp_name, executor=fdl.build(_executor), log_level="INFO") as exp:
        exp_id = exp._id

        pretrain = run.Partial(
            llm.train,
            model=model_cfg,
            trainer=trainer,
            optim=optim,
            data=data,
            tokenizer="data",
            log=log,
            resume=resume
        )

        exp.add(pretrain, name="exp_1")
        exp.run(detach=detach)

        slurm_job_id = get_slurm_job_id(exp.tasks[0])
        write_task_info_to_disk(
            log_dir=os.path.join(exp.executor.tunnel.job_dir, exp_name, exp_id, "exp_1"),
            exp_id=exp_id,
            slurm_job_id=slurm_job_id
            )

    if track:
        track_task(exp_id, slurm_job_id)


if __name__ == "__main__":
    run.cli.main(
        train_model,
        )
