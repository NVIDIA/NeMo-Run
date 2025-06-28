---
description: "Comprehensive installation guide for NeMo Run with optional dependencies for different computing environments, cloud platforms, and execution backends."
tags: ["installation", "setup", "dependencies", "skypilot", "lepton", "kubernetes", "cloud"]
categories: ["get-started"]
---

# Install NeMo Run

This guide covers the installation of NeMo Run and its optional dependencies for different computing environments and execution backends.

## Prerequisites

Before installing NeMo Run, ensure you have the following prerequisites:

### System Requirements

- **Python**: 3.8 or higher
- **pip**: Latest version recommended
- **Git**: For cloning repositories and installing from source
- **Operating System**: Linux, macOS, or Windows (with WSL2 recommended for Windows)

### Python Environment

We recommend using a virtual environment to isolate dependencies:

```bash
# Create a virtual environment
python -m venv nemo-run-env

# Activate the virtual environment
# On Linux/macOS:
source nemo-run-env/bin/activate
# On Windows:
nemo-run-env\Scripts\activate

# Upgrade pip to latest version
pip install --upgrade pip
```

## Core Installation

### Basic Installation

Install NeMo Run from the official GitHub repository:

```bash
pip install git+https://github.com/NVIDIA-NeMo/Run.git
```

### Verification

Verify the installation by checking the version and importing the package:

```bash
# Check installed version
python -c "import nemo_run; print(nemo_run.__version__)"

# Test basic import
python -c "import nemo_run; print('NeMo Run installed successfully')"
```

## Optional Dependencies

NeMo Run supports various execution backends and cloud platforms through optional dependencies.

### SkyPilot Integration

SkyPilot enables cloud-native execution across multiple cloud providers with automatic resource provisioning and cost optimization.

#### Installation Options

::::{tab-set}

:::{tab-item} Kubernetes Support
:sync: sync-skypilot

Install SkyPilot with Kubernetes support for local and cloud Kubernetes clusters:

```bash
pip install git+https://github.com/NVIDIA-NeMo/Run.git[skypilot]
```

This includes:

- SkyPilot core functionality
- Kubernetes cluster management
- Local Kubernetes support (Docker Desktop, minikube, kind)
- Cloud Kubernetes support (GKE, EKS, AKS)
:::

:::{tab-item} All Cloud Support
:sync: sync-skypilot

Install SkyPilot with support for all major cloud providers:

```bash
pip install git+https://github.com/NVIDIA-NeMo/Run.git[skypilot-all]
```

This includes:

- All features from Kubernetes support
- AWS EC2 and EKS support
- Google Cloud Platform (GCP) support
- Microsoft Azure support
- Oracle Cloud Infrastructure (OCI) support
- IBM Cloud support
- Lambda Cloud support
:::

::::

#### Manual SkyPilot Installation

For custom SkyPilot configurations or specific cloud provider support, install manually:

```bash
# Install SkyPilot core
pip install skypilot

# Install specific cloud provider support
pip install skypilot[aws]      # AWS support
pip install skypilot[gcp]      # Google Cloud support
pip install skypilot[azure]    # Azure support
pip install skypilot[lambda]   # Lambda Cloud support
```

For detailed SkyPilot installation instructions, refer to the [official SkyPilot documentation](https://skypilot.readthedocs.io/en/latest/getting-started/installation.html).

### DGX Cloud Lepton Integration

DGX Cloud Lepton provides managed AI infrastructure with pre-configured environments and GPU resources.

#### Lepton CLI Installation

Install the Lepton CLI for DGX Cloud integration:

```bash
pip install leptonai
```

#### Authentication Setup

To authenticate with DGX Cloud Lepton:

1. **Access the Lepton UI**: Navigate to your DGX Cloud Lepton dashboard
2. **Generate Access Token**: Go to **Settings > Tokens** page
3. **Copy Login Command**: Copy the `lep login` command displayed on the page
4. **Authenticate**: Run the copied command in your terminal

Example authentication flow:

```bash
# The command will look similar to this:
lep login --token <your-access-token>

# Verify authentication
lep whoami
```

#### Environment Configuration

Configure your Lepton environment for optimal performance:

```bash
# Set default project (optional)
lep config set project <your-project-name>

# Configure default region (optional)
lep config set region <your-preferred-region>

# List available resources
lep resource list
```

### Additional Execution Backends

#### Kubernetes Support

For Kubernetes-based execution (without SkyPilot):

```bash
# Install Kubernetes dependencies
pip install kubernetes
pip install kubeconfig

# For KubeRay support
pip install kuberay
```

#### Slurm Support

For HPC cluster execution via Slurm:

```bash
# Install Slurm dependencies
pip install pyslurm

# For SSH tunnel support
pip install paramiko
```

#### Ray Support

For Ray-based distributed computing:

```bash
# Install Ray with Kubernetes support
pip install "ray[kubernetes]"
```

## Development Installation

### Install from Source

For development or to use the latest features, install from source:

```bash
# Clone the repository
git clone https://github.com/NVIDIA-NeMo/Run.git
cd Run

# Install in development mode
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

### Build Documentation

To build the documentation locally:

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build HTML documentation
cd docs
make html

# Serve locally
python -m http.server 8000 -d _build/html
```

## Environment Configuration

### Set Up Environment Variables

Configure NeMo Run environment variables:

```bash
# Set NeMo Run home directory (optional)
export NEMORUN_HOME=~/.nemo_run

# Set log level (optional)
export NEMORUN_LOG_LEVEL=INFO

# Enable verbose logging (optional)
export NEMORUN_VERBOSE_LOGGING=false
```

### Verify Installation

Test your installation with a simple example:

```python
import nemo_run as run

# Test basic configuration
config = run.Config(lambda x: x, x=42)
result = config.build()
print(f"Test result: {result}")

# Test executor creation
executor = run.LocalExecutor()
print(f"Executor created: {executor}")
```

## Troubleshooting

### Common Installation Issues

#### Permission Errors

If you encounter permission errors during installation:

```bash
# Use user installation
pip install --user git+https://github.com/NVIDIA-NeMo/Run.git

# Or use a virtual environment
python -m venv nemo-run-env
source nemo-run-env/bin/activate
pip install git+https://github.com/NVIDIA-NeMo/Run.git
```

#### Dependency Conflicts

If you encounter dependency conflicts:

```bash
# Install with --no-deps and manually resolve
pip install git+https://github.com/NVIDIA-NeMo/Run.git --no-deps

# Then install dependencies manually
pip install inquirerpy catalogue fabric fiddle torchx typer rich jinja2 cryptography networkx omegaconf leptonai packaging toml
```

#### Git Installation Issues

If you have issues with Git installation:

```bash
# Ensure Git is installed
git --version

# Use HTTPS instead of SSH
pip install git+https://github.com/NVIDIA-NeMo/Run.git

# Or download and install manually
git clone https://github.com/NVIDIA-NeMo/Run.git
cd Run
pip install .
```

### Verification Commands

Run these commands to verify your installation:

```bash
# Check Python version
python --version

# Check pip version
pip --version

# Check NeMo Run installation
python -c "import nemo_run; print(f'NeMo Run version: {nemo_run.__version__}')"

# Check CLI availability
python -c "from nemo_run.__main__ import app; print('CLI available')"

# Check executor imports
python -c "from nemo_run.core.execution import LocalExecutor, SlurmExecutor; print('Executors available')"
```

## Next Steps

After successful installation:

1. **Read the Configuration Guide**: Learn about `run.Config` and `run.Partial`
2. **Try the CLI Tutorial**: Create your first CLI entrypoint
3. **Explore Execution Backends**: Test different execution environments
4. **Check the Examples**: Review example configurations and workflows

For more detailed information, refer to the [Configuration Guide](../guides/configuration), [CLI Reference](../reference/cli), and [Execution Guide](../guides/execution).

---

::{note}
**Important**: Ensure you have `pip` installed and configured properly before proceeding with the installation. For production deployments, consider using containerized environments for consistent execution across different platforms.
:::
