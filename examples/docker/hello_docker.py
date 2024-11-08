import nemo_run as run

if __name__ == "__main__":
    inline_script = run.Script(
        inline="""
echo "Hello 1"
nvidia-smi
sleep 5
"""
    )
    inline_script_sleep = run.Script(
        inline="""
echo "Hello sleep"
sleep infinity
"""
    )
    executor = run.DockerExecutor(
        container_image="python:3.12",
        num_gpus=-1,
        runtime="nvidia",
        ipc_mode="host",
        shm_size="30g",
        env_vars={"PYTHONUNBUFFERED": "1"},
        packager=run.Packager(),
    )
    with run.Experiment("docker-experiment", executor=executor, log_level="INFO") as exp:
        id1 = exp.add([inline_script, inline_script_sleep], tail_logs=False, name="task-1")
        id2 = exp.add([inline_script, inline_script_sleep], tail_logs=False, name="task-2")
        id3 = exp.add(
            [inline_script, inline_script_sleep],
            tail_logs=False,
            name="task-3",
            dependencies=[id1, id2],
        )

        exp.run(detach=False, tail_logs=True, sequential=False)
