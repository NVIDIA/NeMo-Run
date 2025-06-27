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

#### Docker Support

For containerized execution:

```bash
# Install Docker dependencies
pip install docker
```

## Development Installation

### From Source

For development or custom modifications, install from source:

```bash
# Clone the repository
git clone https://github.com/NVIDIA-NeMo/Run.git
cd Run

# Install in development mode
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

### Development Dependencies

Install additional dependencies for development:

```bash
# Install all development dependencies
pip install -e ".[dev,test,docs]"

# Or install specific development packages
pip install pytest pytest-cov black isort mypy
pip install sphinx sphinx-rtd-theme myst-parser
```

## Environment Configuration

### Configuration Files

NeMo Run uses configuration files to manage execution settings:

```bash
# Create configuration directory
mkdir -p ~/.config/nemo-run

# Create default configuration file
cat > ~/.config/nemo-run/config.yaml << EOF
# Default execution settings
execution:
  default_backend: local
  timeout: 3600
  retry_attempts: 3

# Logging configuration
logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Resource limits
resources:
  max_memory: 32GB
  max_cpus: 16
EOF
```

### Environment Variables

Set environment variables for custom configurations:

```bash
# Set default execution backend
export NEMO_RUN_BACKEND=kubernetes

# Configure logging level
export NEMO_RUN_LOG_LEVEL=DEBUG

# Set resource limits
export NEMO_RUN_MAX_MEMORY=64GB
export NEMO_RUN_MAX_CPUS=32
```

## Platform-Specific Instructions

### Linux Installation

On Linux systems, additional system packages may be required:

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y python3-dev build-essential

# CentOS/RHEL
sudo yum groupinstall -y "Development Tools"
sudo yum install -y python3-devel

# Install NeMo Run
pip install git+https://github.com/NVIDIA-NeMo/Run.git
```

### macOS Installation

On macOS, use Homebrew for system dependencies:

```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install system dependencies
brew install python git

# Install NeMo Run
pip install git+https://github.com/NVIDIA-NeMo/Run.git
```

### Windows Installation

On Windows, use WSL2 for optimal compatibility:

```bash
# Install WSL2 (if not already installed)
wsl --install

# Install NeMo Run in WSL2
pip install git+https://github.com/NVIDIA-NeMo/Run.git
```

## Troubleshooting

### Common Installation Issues

#### Permission Errors

If you encounter permission errors during installation:

```bash
# Use user installation (recommended)
pip install --user git+https://github.com/NVIDIA-NeMo/Run.git

# Or use virtual environment
python -m venv nemo-run-env
source nemo-run-env/bin/activate
pip install git+https://github.com/NVIDIA-NeMo/Run.git
```

#### Network Issues

For network-restricted environments:

```bash
# Use alternative package index
pip install --index-url https://pypi.org/simple/ git+https://github.com/NVIDIA-NeMo/Run.git

# Or download and install locally
git clone https://github.com/NVIDIA-NeMo/Run.git
cd Run
pip install .
```

#### Dependency Conflicts

Resolve dependency conflicts:

```bash
# Upgrade conflicting packages
pip install --upgrade pip setuptools wheel

# Install with dependency resolution
pip install --no-deps git+https://github.com/NVIDIA-NeMo/Run.git
pip install -r requirements.txt
```

### Verification Commands

Verify your installation with these commands:

```bash
# Check Python version
python --version

# Check pip version
pip --version

# Verify NeMo Run installation
python -c "import nemo_run; print(f'NeMo Run version: {nemo_run.__version__}')"

# Test basic functionality
python -c "from nemo_run import run; print('Core functionality working')"
```

## Next Steps

After successful installation:

1. **Read the Quick Start Guide**: Learn basic usage patterns
2. **Explore Configuration Options**: Customize execution settings
3. **Try Example Workflows**: Run sample experiments
4. **Join the Community**: Get help and share experiences

For additional support and resources, visit the [NeMo Run documentation](https://docs.nemo.run) or [GitHub repository](https://github.com/NVIDIA-NeMo/Run).

---

:::{note}
**Important**: Ensure you have `pip` installed and configured properly before proceeding with the installation. For production deployments, consider using containerized environments for consistent execution across different platforms.
:::
