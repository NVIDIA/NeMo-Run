# Management

NeMo-Run also provides ways to inspect and reproduce past experiments. This allows you to check logs, sync artifacts (in the future), cancel running tasks, and rerun an old experiment. When you run an experiment using `run.run` or `run.Experiment`, it creates a run under the experiment title. Once finished, you see the following output at the end:
```python
# The experiment was run with the following tasks: ['echo.sh', 'env_echo_', 'simple.add.add_object']
# You can inspect and reconstruct this experiment at a later point in time using:
experiment = run.Experiment.from_id("experiment_with_scripts_1720556256")
experiment.status() # Gets the overall status
experiment.logs("echo.sh") # Gets the log for the provided task
experiment.cancel("echo.sh") # Cancels the provided task if still running
```

```bash
# You can inspect this experiment at a later point in time using the CLI as well:
nemorun experiment status experiment_with_scripts_1720556256
nemorun experiment logs experiment_with_scripts_1720556256 0
nemorun experiment cancel experiment_with_scripts_1720556256 0
```
This information is specific to each experiment on how to manage it.
See [this notebook](examples/hello-world/hello_experiments.ipynb) for more details and a playable experience.
