from nemotron_22b_BF16 import *

@run.cli.factory
def nemotron_fp8_trainer(num_nodes=8, callbacks=None) -> run.Config[nl.Trainer]:
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
        plugins=bf16_with_fp8_mixed_plugin(),
        callbacks=callbacks
    )

    return trainer

@run.cli.entrypoint(
        require_confirmation=False,
        type="experiment",
)
def train_model(
        ctx: run.cli.RunContext = run.cli.RunContext(name="pretrain_nemotron_22b_fp8_ctx"),
        _executor = eos_executor(nodes=8),
        model_cfg=model(),
        trainer=nemotron_fp8_trainer(),
        optim=nemotron_distributed_fused_adam_cosine_annealing(),
        data = train_data(
            gbs=256,
            mbs=2,
            seq_length=model().config.seq_length,
            index_mapping_dir=None,
            tokenizer_path=None,
            base_data_path=None,
        ),
        log=set_log_dir(
            job_dir=eos_executor().tunnel.job_dir,
            exp_name="pretrain_nemotron_22b_fp8"
            ),
        resume=llm.default_resume(),
        detach=True,
        track=False,
        exp_name="pretrain_nemotron_22b_fp8",
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
