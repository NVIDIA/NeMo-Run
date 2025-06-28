# NeMo Run

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://docs.nvidia.com/nemo-run)
[![PyPI](https://img.shields.io/badge/pypi-nemo--run-blue.svg)](https://pypi.org/project/nemo-run/)

> [!IMPORTANT]
> NeMo Run is currently in active development and this is a pre-release. The API is subject to change without notice while in pre-release. The first official release will be 0.1.0 and will be included in NeMo FW 24.09.

**NeMo Run** is a comprehensive Python framework for configuring, executing, and managing machine learning experiments across diverse computing environments. Built with a focus on reproducibility, flexibility, and scalability, NeMo Run decouples experiment configuration from execution, enabling researchers and ML engineers to seamlessly transition between local development, cloud platforms, and high-performance computing clusters.

## 🚀 Key Features

- **🔧 Type-Safe Configuration**: Python-based configuration using Fiddle with automatic validation and type safety
- **🌐 Multi-Environment Execution**: Support for local, Docker, Slurm, Kubernetes, and cloud platforms (AWS, GCP, Azure, DGX Cloud)
- **📊 Experiment Management**: Comprehensive experiment tracking with metadata preservation and reproducibility
- **🎯 Modular Architecture**: Clean separation between configuration, execution, and management layers
- **⚡ Ray Integration**: Native support for distributed computing with Ray
- **🔍 Rich CLI**: Intelligent command-line interface with type-safe argument parsing

## 🏗️ Core Architecture

NeMo Run is built around three core pillars:

::::{grid} 1 1 1 3
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Configuration
:link: docs/guides/configuration.md
:link-type: doc
:link-alt: Configuration guide

Python-based configuration using Fiddle, supporting complex nested structures and type safety
:::

:::{grid-item-card} {octicon}`play;1.5em;sd-mr-1` Execution
:link: docs/guides/execution.md
:link-type: doc
:link-alt: Execution guide

Multi-environment execution with executors for local, Docker, Slurm, Kubernetes, and cloud platforms
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Management
:link: docs/guides/management.md
:link-type: doc
:link-alt: Management guide

Experiment lifecycle management with metadata tracking, logging, and reproducibility
:::

::::

## 📦 Installation

```bash
pip install git+https://github.com/NVIDIA-NeMo/Run.git
```

### Requirements

- Python 3.8+
- pip (for package installation)
- Access to computing resources (local, cloud, or cluster)

## 🚀 Quick Start

Get started with NeMo Run in three simple steps:

### 1. Configure Your Function

```python
from nemo.collections import llm

# Configure a pre-training recipe
partial_func = llm.llama3_8b.pretrain_recipe(
    name="llama3-8b",
    ckpt_dir="/path/to/store/checkpoints",
    num_nodes=1,
    num_gpus_per_node=8
)
```

### 2. Define Your Executor

```python
import nemo_run as run

# Choose your execution environment
local_executor = run.LocalExecutor()  # Local execution
# docker_executor = run.DockerExecutor()  # Docker execution
# slurm_executor = run.SlurmExecutor()    # Slurm cluster
# ray_executor = run.RayExecutor()        # Ray distributed
```

### 3. Run Your Experiment

```python
# Execute your experiment
run.run(partial_func, executor=local_executor, name="llama3_8b_pretraining")
```

## 🎯 Why Use NeMo Run?

NeMo Run addresses critical challenges in ML experiment management:

- **🔧 Configuration Flexibility**: Type-safe, composable configurations with Python's type system
- **🚀 Execution Modularity**: True environment independence with executor abstraction
- **📊 Experiment Management**: Comprehensive tracking with full metadata preservation
- **🔄 Reproducibility**: One-command experiment reconstruction from metadata
- **⚡ Scalability**: Seamless transition from local development to distributed clusters

For detailed information, see our [Why Use NeMo Run guide](docs/about/why-nemo-run.md).

## 📚 Documentation

- **[Getting Started](docs/get-started/index.md)** - Quick setup and tutorials
- **[Configuration Guide](docs/guides/configuration.md)** - Type-safe configuration management
- **[Execution Guide](docs/guides/execution.md)** - Multi-environment execution
- **[Management Guide](docs/guides/management.md)** - Experiment lifecycle management
- **[CLI Reference](docs/reference/cli.md)** - Command-line interface documentation
- **[FAQs](docs/reference/faqs.md)** - Frequently asked questions

## 🎓 Tutorials

### Hello World Series

The `hello_world` tutorial series provides a comprehensive introduction to NeMo Run:

- **[Part 1](examples/hello-world/hello_world.ipynb)** - Basic configuration and execution
- **[Part 2](examples/hello-world/hello_experiments.ipynb)** - Experiment management
- **[Part 3](examples/hello-world/hello_scripts.py)** - Script-based execution

## 🏛️ Design Philosophy

NeMo Run was designed with these core principles:

### Pythonic

Build and configure everything using Python, eliminating the need for multiple tools to manage experiments.

### Modular

Decoupled task and executor design allows easy combination of different execution environments.

### Opinionated but Flexible

Opinionated in metadata storage and experiment structure, but flexible enough for most use cases.

### Set Up Once and Scale Easily

Initial learning curve pays off with fluid and easy experimentation scaling.

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on:

- Code of Conduct
- Development Setup
- Pull Request Process
- Issue Reporting

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

NeMo Run builds upon the excellent work of these open-source projects:

- [Fiddle](https://github.com/google/fiddle) - Configuration framework
- [TorchX](https://github.com/pytorch/torchx/) - Job submission framework
- [Skypilot](https://github.com/skypilot-org/skypilot/) - Multi-cloud execution
- [XManager](https://github.com/google-deepmind/xmanager) - Experiment management
- [Ray](https://github.com/ray-project/ray) - Distributed computing
- [Rich](https://github.com/Textualize/rich) - Rich terminal output
- [Typer](https://github.com/tiangolo/typer) - CLI framework

## 📞 Support

- **Documentation**: [docs.nvidia.com/nemo-run](https://docs.nvidia.com/nemo-run)
- **Issues**: [GitHub Issues](https://github.com/NVIDIA-NeMo/Run/issues)
- **Discussions**: [GitHub Discussions](https://github.com/NVIDIA-NeMo/Run/discussions)

---

**NeMo Run** is developed by [NVIDIA](https://www.nvidia.com/) as part of the NeMo framework for large language models and AI research.
