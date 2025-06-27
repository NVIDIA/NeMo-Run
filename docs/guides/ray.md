---
description: "Comprehensive guide to deploying and managing Ray clusters and jobs with NeMo Run for distributed computing on Kubernetes and Slurm environments."
tags: ["ray", "distributed", "kubernetes", "slurm", "clusters", "jobs", "distributed-computing"]
categories: ["guides"]
---

# Deploy Ray Clusters and Jobs

> **Audience**: Users familiar with NeMo Run executors who need distributed computing capabilities using Ray on Kubernetes or Slurm environments.
>
> **Overview**: NeMo Run provides unified abstractions for Ray cluster and job management across different execution backends, enabling seamless distributed computing workflows.

## Architecture Overview

NeMo Run's Ray integration provides a unified interface for distributed computing across multiple execution environments. The architecture consists of two primary abstractions:

### Core Components

| Component | Purpose | Supported Backends |
|-----------|---------|-------------------|
| `RayCluster` | Manages long-lived Ray clusters for interactive development | KubeRay (Kubernetes), Slurm |
| `RayJob` | Submits batch jobs to Ray clusters with automatic lifecycle management | KubeRay (Kubernetes), Slurm |

### Execution Model

```text
graph TB
    A[NeMo Run API] --> B[RayCluster/RayJob]
    B --> C{KubeRay Executor}
    B --> D[Slurm Executor]
    C --> E[Kubernetes Cluster]
    D --> F[Slurm Cluster]
    E --> G[Ray Head Node]
    E --> H[Ray Worker Nodes]
    F --> I[Ray Head Node]
    F --> J[Ray Worker Nodes]
```

## RayCluster vs RayJob: Choosing the Right Approach

NeMo Run offers two distinct approaches for Ray-based distributed computing, each optimized for different use cases and workflows.

### RayCluster: Interactive Development

RayCluster provides persistent, long-lived Ray clusters ideal for interactive development and iterative workflows.

**Key Characteristics:**
- **Lifetime**: Remains active until explicitly stopped via `.stop()`
- **Resource Efficiency**: Single cluster setup cost amortized across multiple jobs
- **Multi-tenancy**: Supports multiple concurrent jobs on the same cluster
- **Dashboard Access**: Full Ray dashboard access via port forwarding
- **Use Cases**: Interactive development, debugging, hyperparameter tuning, iterative experimentation

**When to Use RayCluster:**
- Interactive development with Jupyter notebooks or Ray CLI
- Multiple sequential job submissions requiring shared state
- Long-running experiments with frequent parameter adjustments
- Development workflows requiring persistent cluster state

### RayJob: Batch Processing

RayJob provides ephemeral clusters optimized for batch processing and automated workflows.

**Key Characteristics:**
- **Lifetime**: Ephemeral - automatically terminates after job completion
- **Resource Efficiency**: Resources freed immediately after job completion
- **Single-tenancy**: One job per cluster instance
- **Dashboard Access**: Limited access (cluster terminates with job)
- **Use Cases**: CI/CD pipelines, scheduled training, production inference, automated workflows

**When to Use RayJob:**
- Automated batch processing pipelines
- CI/CD workflows requiring reproducible execution
- Production inference jobs with predictable resource requirements
- Scenarios requiring automatic cleanup and resource management

### Decision Matrix

| Factor | RayCluster | RayJob |
|--------|------------|--------|
| **Development Phase** | Interactive/Exploratory | Production/Batch |
| **Job Frequency** | Multiple jobs per session | Single job per submission |
| **Resource Utilization** | High (shared cluster) | Low (ephemeral) |
| **Setup Overhead** | One-time | Per submission |
| **State Persistence** | Yes | No |
| **Automation** | Manual management | Fully automated |

## Kubernetes Integration with KubeRay

KubeRay provides native Ray support on Kubernetes, enabling cloud-native distributed computing with container orchestration.

### KubeRay Architecture

KubeRay extends Kubernetes with custom resources for Ray cluster management:

- **RayCluster**: Custom resource defining Ray cluster topology
- **RayJob**: Custom resource for job submission to Ray clusters
- **RayService**: Custom resource for serving Ray applications

### KubeRay Executor Configuration

The `KubeRayExecutor` provides comprehensive configuration options for Kubernetes-based Ray deployments.

#### Basic Configuration

```python
from nemo_run.core.execution.kuberay import KubeRayExecutor, KubeRayWorkerGroup

executor = KubeRayExecutor(
    namespace="ml-team",
    ray_version="2.43.0",
    image="anyscale/ray:2.43.0-py312-cu125",
    head_cpu="4",
    head_memory="12Gi",
    worker_groups=[
        KubeRayWorkerGroup(
            group_name="worker",
            replicas=2,
            gpus_per_worker=8,
            cpu_per_worker="16",
            memory_per_worker="64Gi",
        )
    ],
)
```

#### Advanced Configuration

```python
executor = KubeRayExecutor(
    namespace="ml-team",
    ray_version="2.43.0",
    image="anyscale/ray:2.43.0-py312-cu125",
    head_cpu="8",
    head_memory="32Gi",

    # Worker group configuration
    worker_groups=[
        KubeRayWorkerGroup(
            group_name="gpu-workers",
            replicas=4,
            gpus_per_worker=8,
            cpu_per_worker="32",
            memory_per_worker="128Gi",
            min_replicas=2,
            max_replicas=8,
        ),
        KubeRayWorkerGroup(
            group_name="cpu-workers",
            replicas=2,
            cpu_per_worker="16",
            memory_per_worker="64Gi",
        )
    ],

    # Volume management
    volume_mounts=[
        {"name": "workspace", "mountPath": "/workspace"},
        {"name": "datasets", "mountPath": "/datasets"},
    ],
    volumes=[
        {
            "name": "workspace",
            "persistentVolumeClaim": {"claimName": "ml-workspace-pvc"},
        },
        {
            "name": "datasets",
            "persistentVolumeClaim": {"claimName": "datasets-pvc"},
        }
    ],

    # Environment configuration
    env_vars={
        "UV_PROJECT_ENVIRONMENT": "/home/ray/venvs/driver",
        "NEMO_RL_VENV_DIR": "/home/ray/venvs",
        "HF_HOME": "/workspace/hf_cache",
        "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
    },

    # Security and scheduling
    spec_kwargs={
        "schedulerName": "runai-scheduler",
        "priorityClassName": "high-priority",
    },
    container_kwargs={
        "securityContext": {
            "allowPrivilegeEscalation": False,
            "runAsUser": 1000,
            "runAsGroup": 1000,
            "fsGroup": 1000,
        }
    },

    # Resource management
    reuse_volumes_in_worker_groups=True,
    enable_in_tree_autoscaling=True,
    autoscaling_mode="Default",
)
```

### Complete KubeRay Workflow Example

```python
from nemo_run.core.execution.kuberay import KubeRayExecutor, KubeRayWorkerGroup
from nemo_run.run.ray.cluster import RayCluster
from nemo_run.run.ray.job import RayJob

# 1. Configure KubeRay executor with production settings
executor = KubeRayExecutor(
    namespace="ml-production",
    ray_version="2.43.0",
    image="anyscale/ray:2.43.0-py312-cu125",
    head_cpu="8",
    head_memory="32Gi",
    worker_groups=[
        KubeRayWorkerGroup(
            group_name="gpu-workers",
            replicas=4,
            gpus_per_worker=8,
            cpu_per_worker="32",
            memory_per_worker="128Gi",
            min_replicas=2,
            max_replicas=8,
        )
    ],
    volume_mounts=[{"name": "workspace", "mountPath": "/workspace"}],
    volumes=[{
        "name": "workspace",
        "persistentVolumeClaim": {"claimName": "ml-workspace-pvc"},
    }],
    env_vars={
        "UV_PROJECT_ENVIRONMENT": "/home/ray/venvs/driver",
        "HF_HOME": "/workspace/hf_cache",
    },
)

# 2. Pre-start commands for environment setup
pre_ray_start = [
    "pip install uv",
    "echo 'unset RAY_RUNTIME_ENV_HOOK' >> /home/ray/.bashrc",
    "mkdir -p /workspace/hf_cache",
]

# 3. Deploy persistent cluster for development
cluster = RayCluster(name="ml-dev-cluster", executor=executor)
cluster.start(
    timeout=900,
    pre_ray_start_commands=pre_ray_start,
    wait_until_ready=True
)

# 4. Expose Ray dashboard for monitoring
cluster.port_forward(port=8265, target_port=8265, wait=False)
print("Ray dashboard available at: http://localhost:8265")

# 5. Submit training job to the cluster
job = RayJob(name="training-job-001", executor=executor)
job.start(
    command="uv run python train.py --config configs/train.yaml",
    workdir="/workspace/project/",
    runtime_env_yaml="/workspace/project/runtime_env.yaml",
    pre_ray_start_commands=pre_ray_start,
)

# 6. Monitor job execution
job.logs(follow=True)

# 7. Clean up resources
cluster.stop()
```

### KubeRay Best Practices

#### Resource Management
- **Autoscaling**: Enable autoscaling for variable workloads
- **Resource Limits**: Set appropriate CPU/memory limits to prevent resource exhaustion
- **GPU Scheduling**: Use GPU-aware schedulers for optimal GPU utilization

#### Volume Management
- **Persistent Storage**: Use PVCs for data persistence across job restarts
- **Code Synchronization**: Leverage automatic workdir synchronization for seamless development
- **Cache Management**: Mount dedicated volumes for model caches and datasets

#### Security Configuration
- **RBAC**: Implement proper role-based access control
- **Network Policies**: Restrict network access between pods
- **Security Contexts**: Configure appropriate security contexts for containers

## Slurm Integration

Slurm integration enables Ray clusters on traditional HPC systems, leveraging existing job scheduling infrastructure and resource management.

### Slurm Architecture

Slurm-based Ray clusters utilize Slurm's job scheduling capabilities:

- **Array Jobs**: Ray clusters are deployed as Slurm array jobs
- **Resource Allocation**: Leverages Slurm's native resource management
- **SSH Tunneling**: Remote access via SSH tunnels to login nodes

### Slurm Executor Configuration

The `SlurmExecutor` provides configuration options for HPC environments.

#### Basic Configuration

```python
from nemo_run.core.execution.slurm import SlurmExecutor, SSHTunnel

# SSH tunnel configuration for remote access
ssh = SSHTunnel(
    host="login.cluster.com",
    user="username",
    job_dir="/scratch/username/runs",
    identity="~/.ssh/id_ed25519",
)

executor = SlurmExecutor(
    account="gpu-dept",
    partition="a100",
    nodes=2,
    gpus_per_node=8,
    time="04:00:00",
    container_image="nvcr.io/nvidia/pytorch:24.05-py3",
    tunnel=ssh,
)
```

#### Advanced Configuration

```python
from pathlib import Path
from nemo_run.core.execution.slurm import SlurmExecutor, SlurmJobDetails, SSHTunnel

# Enhanced SSH tunnel with custom configuration
ssh = SSHTunnel(
    host="login.cluster.com",
    user="username",
    job_dir="/scratch/username/runs",
    identity="~/.ssh/id_ed25519",
    port=22,
    timeout=30,
)

# Custom job details for enhanced logging
class CustomJobDetails(SlurmJobDetails):
    @property
    def stdout(self) -> Path:
        assert self.folder
        return Path(self.folder) / "slurm_stdout.log"

    @property
    def stderr(self) -> Path:
        assert self.folder
        return Path(self.folder) / "slurm_stderr.log"

executor = SlurmExecutor(
    # Slurm job configuration
    account="gpu-dept",
    partition="a100",
    nodes=4,
    gpus_per_node=8,
    gres="gpu:8",
    time="08:00:00",
    qos="high",

    # Container configuration
    container_image="nvcr.io/nvidia/pytorch:24.05-py3",
    container_mounts=[
        "/scratch:/scratch",
        "/home:/home",
        "/datasets:/datasets"
    ],

    # Environment variables
    env_vars={
        "HF_HOME": "/scratch/hf_cache",
        "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
        "NCCL_DEBUG": "INFO",
    },

    # SSH tunnel configuration
    tunnel=ssh,

    # Custom job details
    job_details=CustomJobDetails(),

    # Additional Slurm options
    slurm_options={
        "mail-type": "ALL",
        "mail-user": "user@example.com",
        "exclusive": None,  # Flag without value
    }
)
```

### Complete Slurm Workflow Example

```python
import os
from pathlib import Path

import nemo_run as run
from nemo_run.core.execution.slurm import SlurmExecutor, SSHTunnel
from nemo_run.run.ray.cluster import RayCluster
from nemo_run.run.ray.job import RayJob

# 1. Configure SSH tunnel for remote cluster access
ssh = SSHTunnel(
    host="login.hpc.cluster.com",
    user="jdoe",
    job_dir="/scratch/jdoe/runs",
    identity="~/.ssh/id_ed25519",
)

# 2. Configure Slurm executor for large-scale training
executor = SlurmExecutor(
    account="gpu-dept",
    partition="a100",
    nodes=8,
    gpus_per_node=8,
    gres="gpu:8",
    time="12:00:00",
    qos="high",
    container_image="nvcr.io/nvidia/pytorch:24.05-py3",
    container_mounts=[
        "/scratch:/scratch",
        "/home:/home",
        "/datasets:/datasets"
    ],
    env_vars={
        "HF_HOME": "/scratch/hf_cache",
        "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
        "NCCL_DEBUG": "INFO",
        "NCCL_IB_DISABLE": "0",
    },
    tunnel=ssh,
)

# 3. Environment setup commands
pre_ray_start = [
    "pip install uv",
    "mkdir -p /scratch/hf_cache",
    "export NCCL_DEBUG=INFO",
]

# 4. Deploy Ray cluster on Slurm
cluster = RayCluster(name="hpc-training-cluster", executor=executor)
cluster.start(
    timeout=1800,
    pre_ray_start_commands=pre_ray_start,
    wait_until_ready=True
)

# 5. Set up port forwarding for dashboard access
cluster.port_forward(port=8265, target_port=8265)
print("Ray dashboard available at: http://localhost:8265")

# 6. Submit distributed training job
job = RayJob(name="distributed-training", executor=executor)
job.start(
    command="uv run python train.py --config configs/distributed.yaml",
    workdir="/scratch/jdoe/project/",
    pre_ray_start_commands=pre_ray_start,
)

# 7. Monitor training progress
job.logs(follow=True)

# 8. Clean up resources
cluster.stop()
```

### Slurm Best Practices

#### Resource Optimization
- **Partition Selection**: Choose appropriate partitions based on resource requirements
- **QoS Configuration**: Use appropriate quality of service levels for job priority
- **Node Allocation**: Optimize node allocation for specific workload requirements

#### Network Configuration
- **NCCL Settings**: Configure NCCL for optimal inter-node communication
- **InfiniBand**: Enable InfiniBand for high-performance networking
- **Firewall Rules**: Ensure proper network access for Ray cluster communication

#### Storage Management
- **Scratch Space**: Utilize scratch directories for temporary data
- **Persistent Storage**: Use home directories for code and configuration files
- **Data Locality**: Mount datasets close to compute resources

## API Reference

### RayCluster API

The `RayCluster` class provides comprehensive cluster management capabilities.

#### Core Methods

```python
# Cluster lifecycle management
cluster = RayCluster(name="my-cluster", executor=executor)

# Start cluster with configuration
cluster.start(
    timeout=600,                    # Maximum wait time in seconds
    wait_until_ready=True,          # Block until cluster is ready
    pre_ray_start_commands=[        # Commands to run before Ray starts
        "pip install -r requirements.txt",
        "mkdir -p /workspace/data"
    ]
)

# Check cluster status
status = cluster.status(display=True)  # Display status and return info

# Access Ray dashboard
cluster.port_forward(
    port=8265,                      # Local port
    target_port=8265,               # Remote port
    wait=False                      # Don't block on port forwarding
)

# Stop and clean up cluster
cluster.stop()
```

#### Advanced Methods

```python
# Get cluster configuration
config = cluster.get_config()

# Scale worker groups
cluster.scale_worker_group("worker", replicas=4)

# Get cluster logs
logs = cluster.get_logs()

# Check cluster health
health = cluster.health_check()
```

### RayJob API

The `RayJob` class provides job submission and management capabilities.

#### Core Methods

```python
# Job lifecycle management
job = RayJob(name="my-job", executor=executor)

# Submit job to cluster
job.start(
    command="python train.py --config config.yaml",  # Job command
    workdir="/workspace/project/",                   # Working directory
    runtime_env_yaml="/path/to/runtime_env.yaml",    # Runtime environment
    pre_ray_start_commands=[                         # Pre-start commands
        "pip install -r requirements.txt"
    ]
)

# Check job status
status = job.status()

# Stream job logs
job.logs(follow=True, tail=100)  # Follow logs, show last 100 lines

# Stop job execution
job.stop()
```

#### Advanced Methods

```python
# Get job configuration
config = job.get_config()

# Get job metrics
metrics = job.get_metrics()

# Submit multiple jobs
jobs = []
for i in range(5):
    job = RayJob(name=f"job-{i}", executor=executor)
    job.start(command=f"python script.py --seed {i}")
    jobs.append(job)

# Wait for all jobs to complete
for job in jobs:
    job.logs(follow=True)
```

## Advanced Configuration

### Runtime Environment Management

Ray runtime environments provide isolated execution contexts for jobs.

```python
# Runtime environment configuration
runtime_env = {
    "working_dir": "/workspace/project",
    "pip": {
        "packages": ["torch", "transformers", "datasets"]
    },
    "env_vars": {
        "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        "HF_HOME": "/workspace/hf_cache"
    },
    "container": {
        "image": "anyscale/ray:2.43.0-py312-cu125"
    }
}

# Save runtime environment to file
import yaml
with open("runtime_env.yaml", "w") as f:
    yaml.dump(runtime_env, f)

# Use in job submission
job.start(
    command="python train.py",
    runtime_env_yaml="runtime_env.yaml"
)
```

### Custom Resource Scheduling

Configure custom resource requirements for specialized workloads.

```python
# Custom resource configuration for KubeRay
executor = KubeRayExecutor(
    # ... other configuration ...
    worker_groups=[
        KubeRayWorkerGroup(
            group_name="specialized-workers",
            replicas=2,
            gpus_per_worker=8,
            custom_resources={
                "nvidia.com/mig-1g.5gb": 1,
                "nvidia.com/mig-3g.20gb": 2,
            }
        )
    ]
)
```

### Monitoring and Observability

Implement comprehensive monitoring for Ray clusters and jobs.

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Monitor cluster metrics
cluster = RayCluster(name="monitored-cluster", executor=executor)
cluster.start()

# Access Ray dashboard metrics
cluster.port_forward(port=8265, target_port=8265)

# Monitor job progress
job = RayJob(name="monitored-job", executor=executor)
job.start(command="python train.py")

# Stream logs with custom formatting
for line in job.logs(follow=True):
    if "loss" in line:
        print(f"Training loss: {line}")
```

## Troubleshooting

### Common Issues and Solutions

#### Cluster Startup Failures

**Issue**: Cluster fails to start within timeout period
**Solutions**:
- Increase timeout value in `cluster.start(timeout=1800)`
- Check resource availability in target partition/namespace
- Verify network connectivity and firewall rules
- Review pre-start commands for errors

#### Job Submission Failures

**Issue**: Jobs fail to submit or execute
**Solutions**:
- Verify cluster is in ready state before job submission
- Check runtime environment configuration
- Ensure working directory exists and is accessible
- Review job command syntax and dependencies

#### Performance Issues

**Issue**: Poor distributed training performance
**Solutions**:
- Configure NCCL settings for optimal communication
- Verify GPU topology and network configuration
- Use appropriate batch sizes and gradient accumulation
- Monitor resource utilization and bottlenecks

#### Network Connectivity

**Issue**: Ray dashboard or job communication failures
**Solutions**:
- Verify port forwarding configuration
- Check firewall rules and network policies
- Ensure proper DNS resolution
- Review SSH tunnel configuration for Slurm deployments

### Debugging Techniques

#### Log Analysis

```python
# Enable verbose logging
import logging
logging.getLogger("nemo_run").setLevel(logging.DEBUG)

# Collect detailed logs
cluster_logs = cluster.get_logs()
job_logs = job.get_logs()

# Analyze log patterns
for log in cluster_logs:
    if "ERROR" in log:
        print(f"Cluster error: {log}")
```

#### Health Checks

```python
# Perform comprehensive health check
health_status = cluster.health_check()
print(f"Cluster health: {health_status}")

# Check individual components
head_health = cluster.check_head_node()
worker_health = cluster.check_worker_nodes()
```

## Integration Patterns

### CI/CD Integration

Integrate Ray jobs into continuous integration and deployment pipelines.

```python
# GitHub Actions workflow example
def run_training_job():
    executor = KubeRayExecutor(
        namespace="ci-cd",
        worker_groups=[KubeRayWorkerGroup(group_name="worker", replicas=1, gpus_per_worker=4)]
    )

    job = RayJob(name="ci-training", executor=executor)
    job.start(
        command="python train.py --config configs/ci.yaml",
        workdir="./",
    )

    # Wait for completion and check exit code
    job.logs(follow=True)
    if job.status() != "SUCCEEDED":
        raise Exception("Training job failed")
```

### Multi-Environment Deployment

Deploy Ray applications across different environments with consistent configuration.

```python
# Environment-specific configuration
environments = {
    "dev": {
        "namespace": "ml-dev",
        "replicas": 1,
        "gpus_per_worker": 2,
    },
    "staging": {
        "namespace": "ml-staging",
        "replicas": 2,
        "gpus_per_worker": 4,
    },
    "production": {
        "namespace": "ml-prod",
        "replicas": 4,
        "gpus_per_worker": 8,
    }
}

def deploy_to_environment(env_name):
    config = environments[env_name]
    executor = KubeRayExecutor(
        namespace=config["namespace"],
        worker_groups=[KubeRayWorkerGroup(
            group_name="worker",
            replicas=config["replicas"],
            gpus_per_worker=config["gpus_per_worker"]
        )]
    )

    cluster = RayCluster(name=f"{env_name}-cluster", executor=executor)
    cluster.start()
    return cluster
```

### Custom CLI Applications

Build custom command-line interfaces for Ray cluster and job management.

```python
import argparse
import sys
from nemo_run.core.execution.kuberay import KubeRayExecutor, KubeRayWorkerGroup
from nemo_run.run.ray.cluster import RayCluster
from nemo_run.run.ray.job import RayJob

def create_cluster(args):
    executor = KubeRayExecutor(
        namespace=args.namespace,
        worker_groups=[KubeRayWorkerGroup(
            group_name="worker",
            replicas=args.replicas,
            gpus_per_worker=args.gpus
        )]
    )

    cluster = RayCluster(name=args.name, executor=executor)
    cluster.start()
    print(f"Cluster {args.name} started successfully")

def submit_job(args):
    executor = KubeRayExecutor(namespace=args.namespace)
    job = RayJob(name=args.name, executor=executor)
    job.start(command=args.command, workdir=args.workdir)
    print(f"Job {args.name} submitted successfully")

def main():
    parser = argparse.ArgumentParser(description="Ray Cluster and Job Manager")
    subparsers = parser.add_subparsers(dest="command")

    # Cluster management
    cluster_parser = subparsers.add_parser("cluster", help="Manage clusters")
    cluster_parser.add_argument("--name", required=True, help="Cluster name")
    cluster_parser.add_argument("--namespace", default="default", help="Kubernetes namespace")
    cluster_parser.add_argument("--replicas", type=int, default=1, help="Number of workers")
    cluster_parser.add_argument("--gpus", type=int, default=1, help="GPUs per worker")

    # Job management
    job_parser = subparsers.add_parser("job", help="Submit jobs")
    job_parser.add_argument("--name", required=True, help="Job name")
    job_parser.add_argument("--command", required=True, help="Job command")
    job_parser.add_argument("--workdir", default="./", help="Working directory")
    job_parser.add_argument("--namespace", default="default", help="Kubernetes namespace")

    args = parser.parse_args()

    if args.command == "cluster":
        create_cluster(args)
    elif args.command == "job":
        submit_job(args)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
```

## Performance Optimization

### Resource Allocation Strategies

Optimize resource allocation for different workload types.

```python
# CPU-intensive workloads
cpu_executor = KubeRayExecutor(
    worker_groups=[KubeRayWorkerGroup(
        group_name="cpu-workers",
        replicas=8,
        cpu_per_worker="16",
        memory_per_worker="64Gi"
    )]
)

# GPU-intensive workloads
gpu_executor = KubeRayExecutor(
    worker_groups=[KubeRayWorkerGroup(
        group_name="gpu-workers",
        replicas=4,
        gpus_per_worker=8,
        cpu_per_worker="32",
        memory_per_worker="128Gi"
    )]
)

# Mixed workloads
mixed_executor = KubeRayExecutor(
    worker_groups=[
        KubeRayWorkerGroup(
            group_name="gpu-workers",
            replicas=2,
            gpus_per_worker=8,
        ),
        KubeRayWorkerGroup(
            group_name="cpu-workers",
            replicas=4,
            cpu_per_worker="16",
        )
    ]
)
```

### Network Optimization

Configure network settings for optimal distributed training performance.

```python
# Optimize NCCL settings for high-performance networking
env_vars = {
    "NCCL_DEBUG": "INFO",
    "NCCL_IB_DISABLE": "0",
    "NCCL_IB_HCA": "mlx5_0",
    "NCCL_IB_SL": "0",
    "NCCL_IB_TC": "41",
    "NCCL_IB_QPS_PER_CONNECTION": "4",
    "NCCL_IB_TIMEOUT": "23",
    "NCCL_IB_RETRY_CNT": "7",
    "NCCL_IB_PKEY": "0xffff",
    "NCCL_IB_USE_INLINE": "1",
    "NCCL_IB_ADAPTIVE_ROUTING": "1",
}

executor = KubeRayExecutor(
    env_vars=env_vars,
    # ... other configuration
)
```

### Memory Management

Optimize memory usage for large-scale training workloads.

```python
# Configure memory-efficient settings
env_vars = {
    "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128",
    "CUDA_LAUNCH_BLOCKING": "1",
    "TORCH_USE_CUDA_DSA": "1",
}

# Use gradient checkpointing for memory efficiency
training_command = """
python train.py \
    --config configs/train.yaml \
    --gradient_checkpointing \
    --max_memory_MB 8192
"""
```

---

This comprehensive guide covers all aspects of Ray distributed computing with NeMo Run, from basic concepts to advanced optimization techniques. The unified API across Kubernetes and Slurm environments enables seamless distributed computing workflows regardless of the underlying infrastructure.
