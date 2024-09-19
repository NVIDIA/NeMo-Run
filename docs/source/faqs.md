# Frequently Asked Questions

This section provides answers to common questions organized by Nemo-Run functions.

- [Frequently Asked Questions](#frequently-asked-questions)
  - [Configuration](#configuration)
      - [**Q:** UnserializableValueError when using `run.Partial` or `run.Config`](#q-unserializablevalueerror-when-using-runpartial-or-runconfig)
      - [**Q:** Deserialization error when using `run.Partial` or `run.Config`](#q-deserialization-error-when-using-runpartial-or-runconfig)
      - [**Q:** How to use control flow in autoconvert?](#q-how-to-use-control-flow-in-autoconvert)
      - [**Q:** I made a change locally in my git repo and tested it using the local executor. However, the change is not reflected in the remote job.](#q-i-made-a-change-locally-in-my-git-repo-and-tested-it-using-the-local-executor-however-the-change-is-not-reflected-in-the-remote-job)
      - [**Q:** I made a change locally *outside* my git repo and tested it using the local executor. However, the change is not reflected in the remote job.](#q-i-made-a-change-locally-outside-my-git-repo-and-tested-it-using-the-local-executor-however-the-change-is-not-reflected-in-the-remote-job)
  - [Execution](#execution)
      - [**Q:** For SlurmExecutor, how can I execute directly from the login node of the cluster.](#q-for-slurmexecutor-how-can-i-execute-directly-from-the-login-node-of-the-cluster)
  - [Management](#management)
      - [**Q:** I can't retrieve logs for an experiment.](#q-i-cant-retrieve-logs-for-an-experiment)


## Configuration

#### **Q:** UnserializableValueError when using `run.Partial` or `run.Config`
```
fiddle._src.experimental.serialization.UnserializableValueError: Unserializable value .tmp of type <class 'pathlib.PosixPath'>. Error occurred at path '<root>.something'."
```
**A:** Every nested object inside `run.Partial` or `run.Config` needs to be serializable. As a result, if you are trying to configure objects, it's better to wrap them in `run.Config`. For example, the above error arises when you do the following:
```python
from nemorun.config import ZlibJSONSerializer

partial = run.Partial(some_function, something=Path("/tmp"))
ZlibJSONSerializer().serialize(partial)
```

You can fix it by doing:
```python
from nemorun.config import ZlibJSONSerializer

partial = run.Partial(some_function, something=run.Config(Path, "/tmp"))
ZlibJSONSerializer().serialize(partial)
```

#### **Q:** Deserialization error when using `run.Partial` or `run.Config`
One example shown below
```
ValueError: Using the Buildable constructor to convert a buildable to a new type or to override arguments is forbidden; please use either `fdl.cast(new_type, buildable)` (for casting) or `fdl.copy_with(buildable, **kwargs)` (for overriding arguments).
```
**A:** Ensure that only `Config` or `Partial` objects are present in your nested configuration. You can run a quick sanity check by doing
```python
from nemorun.config import ZlibJSONSerializer

serializer = ZlibJSONSerializer()
partial = run.Partial(some_function, something=run.Config(Path, "/tmp"))
serializer.deserialize(serializer.serialize(partial)) == partial
```

#### **Q:** How to use control flow in autoconvert?
If I use control flow with `run.autoconvert`, I get `UnsupportedLanguageConstructError: Control flow (ListComp) is unsupported by auto_config.`. For example, the below doesn't work.
```python
@run.autoconvert
def control_flow() -> llm.PreTrainingDataModule:
    return llm.PreTrainingDataModule(
        paths=[Path(f"some_doc_{i}") for i in range(10)],
        weights=[1 for i in range(10)]
    )
```
**A:** As the error mentions, control flow in `run.autoconvert` is not supported. To overcome, just return a config directly and use it like a regular python function. So the example would become
```python
def control_flow_config() -> run.Config[llm.PreTrainingDataModule]:
    return run.Config(
        llm.PreTrainingDataModule,
        paths=[run.Config(Path, f"some_doc_{i}") for i in range(10)],
        weights=[1 for i in range(10)]
    )
```


#### **Q:** I made a change locally in my git repo and tested it using the local executor. However, the change is not reflected in the remote job.
**A**: This is most likely because you haven't committed the changes. See details about `GitArchivePackager` [here](./execution.md#packagers) to learn more.


#### **Q:** I made a change locally *outside* my git repo and tested it using the local executor. However, the change is not reflected in the remote job.
**A**: Currently, we only package your current repo. To transport changes to other repos on the remote cluster, you need to check out the package on the remote cluster and then mount it at the correct path in your docker image. We will add support for packaging multiple repos in the future.

## Execution
#### **Q:** For SlurmExecutor, how can I execute directly from the login node of the cluster.
**A**: For example, to execute the SlurmExecutor from your local machine via SSH, you may have:
```python
ssh_tunnel = run.SSHTunnel(
    host="your-slurm-host",
    user="your-user",
    job_dir="/your/home/directory/nemo-run-experiments",
)
executor = run.SlurmExecutor(
    ...
    tunnel=ssh_tunnel,
    ...
)
```

If you are on the login node of the Slurm cluster, simply change the tunnel as shown below:
```python
executor = run.SlurmExecutor(
    ...
    tunnel=run.LocalTunnel(),
    ...
)
```


## Management
#### **Q:** I can't retrieve logs for an experiment.
**A**: There could be a few reasons for this, described below:
- The Nemo-Run home has changed. By default home is at `~/.nemorun`, but you can overwrite it using `NEMORUN_HOME`. Retrieving logs can be difficult if there's a discrepancy in the home between when you launched the experiment and when you try to retrieve it.
- Nemo-Run home is deleted or overwritten from the time when you ran the experiment.
- Logs are not available on the remote cluster. For example, if launching on Kubernetes using the `SkypilotExecutor`, and the Skypilot cluster is terminated or the pod is deleted, the logs won’t be available.
