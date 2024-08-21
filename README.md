# NeMo-Run

> [!IMPORTANT]
> NeMo-Run is still in active development and this is a pre-release. The API is subject to change without notice while in pre-release. First official release will be 0.1.0 and will be included in NeMo FW 24.09 as well.

NeMo-Run is a powerful tool designed to streamline the configuration, execution, and management of machine learning experiments across various computing environments. NeMo-Run has three core responsibilities:

1. [Configuration](./docs/source/guides/configuration.md)
2. [Execution](./docs/source/guides/execution.md)
3. [Management](./docs/source/guides/management.md)

To learn more, click on each link. This represents the typical order that Nemo-Run users follow for setting up and launching experiments.

- [NeMo-Run](#nemo-run)
  - [Why Use Nemo-Run?](#why-use-nemo-run)
  - [Install NeMo-Run](#install-nemo-run)
  - [Get Started](#get-started)
  - [Design Philosophy and Inspiration](#design-philosophy-and-inspiration)
    - [Pythonic](#pythonic)
    - [Modular](#modular)
    - [Opinionated but Flexible](#opinionated-but-flexible)
    - [Set Up Once and Scale Easily](#set-up-once-and-scale-easily)
  - [Tutorials](#tutorials)
      - [Hello world](#hello-world)
  - [Contribute to NeMo-Run](#contribute-to-nemo-run)
  - [FAQs](#faqs)


## Why Use Nemo-Run?
Please see this [detailed guide](./docs/source/guides/why-use-nemo-run.md) for reasons to use Nemo-Run.

## Install NeMo-Run
To install the project, use the following command:

```bash
pip install git+https://github.com/NVIDIA/NeMo-Run.git
```

Make sure you have `pip` installed and configured properly.

## Get Started
To get started with Nemo-Run, follow these three steps based on the core responsibilities mentioned above. For this example, we’ll showcase a pre-training example in Nemo 2.0 using Llama3.

1. Configure your function:
```python
from nemo.collections import llm
partial_func = llm.llama3_8b.pretrain_recipe(name="llama3-8b", ckpt_dir="/path/to/store/checkpoints", num_nodes=1, num_gpus_per_node=8)
```

2. Define your Executor:
```python
import nemo_run as run
# Local executor
local_executor = run.LocalExecutor()
```

3. Run your experiment:
```python
run.run(partial_func, executor=local_executor, name="llama3_8b_pretraining")
```

## Design Philosophy and Inspiration
In building NeMo-Run, we drew inspiration from and relied on the following primary libraries. We would like to extend our gratitude for their work.

- [Fiddle](https://github.com/google/fiddle)
- [TorchX](https://github.com/pytorch/torchx/)
- [Skypilot](https://github.com/skypilot-org/skypilot/)
- [XManager](https://github.com/google-deepmind/xmanager/tree/main)
- [Fabric](https://github.com/fabric/fabric) and [Paramiko](https://github.com/paramiko/paramiko)
- [Rich](https://github.com/Textualize/rich)
- [Jinja](https://github.com/pallets/jinja/)

Apart from these, we also build on other libraries. A full list of dependencies can be found in [pyproject.toml](pyproject.toml).

NeMo-Run was designed keeping the following principles in mind:

### Pythonic
In NeMo-Run, you can build and configure everything using Python, eliminating the need for multiple combinations of tools to manage your experiments. The only exception is when setting up the environment for remote execution, where we rely on Docker.

### Modular
The decoupling of task and executor allows you to form different combinations of execution units with relative ease. You configure different remote environments once, and you can reuse it across a variety of tasks in a Pythonic way.

### Opinionated but Flexible
NeMo-Run is opinionated in some places, like storing of metadata information for experiments in a particular manner. However, it remains flexible enough to accommodate most user experiments.

### Set Up Once and Scale Easily
While it may take some time initially for users to become familiar with NeMo-Run concepts, the tool is designed to scale experimentation in a fluid and easy manner.

## Tutorials

#### Hello world

The `hello_world` tutorial series provides a comprehensive introduction to NeMo-Run, demonstrating its capabilities through a simple example. The tutorial covers:

- Configuring Python functions using `Partial` and `Config` classes.
- Executing configured functions locally and on remote clusters.
- Visualizing configurations with `graphviz`.
- Creating and managing experiments using `run.Experiment`.

You can find the tutorial series below:
- [Part 1](examples/hello-world/hello_world.ipynb).
- [Part 2](examples/hello-world/hello_experiments.ipynb).
- [Part 3](examples/hello-world/hello_scripts.py).

## Contribute to NeMo-Run
Please see the [contribution guide](./CONTRIBUTING.md) to contribute to NeMo Run.

## FAQs
Please find a list of frequently asked questions [here](./docs/source/faqs.md).
