---
description: "Learn about NeMo Run's core concepts, key features, and fundamental architecture for ML experiment management and distributed computing."
tags: ["overview", "concepts", "architecture", "features", "ml", "distributed-computing"]
categories: ["about"]
---

(about-overview)=

# About NeMo Run

NeMo Run is a comprehensive Python framework for configuring, executing, and managing machine learning experiments across diverse computing environments. Built with a focus on reproducibility, flexibility, and scalability, NeMo Run decouples experiment configuration from execution, enabling researchers and ML engineers to seamlessly transition between local development, cloud platforms, and high-performance computing clusters.

## What is NeMo Run?

NeMo Run provides a unified interface for ML experiment lifecycle management, addressing the common challenges of:

- **Configuration Management**: Complex, nested configurations for models, data, and training parameters
- **Execution Orchestration**: Running experiments across different environments (local, Docker, Slurm, Kubernetes, cloud)
- **Experiment Tracking**: Managing, monitoring, and reproducing experiments with full metadata preservation

The framework is built on three core pillars:

::::{grid} 1 1 1 3
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Configuration
:link: ../guides/configuration
:link-type: doc
:link-alt: Configuration guide

Python-based configuration using Fiddle, supporting complex nested structures and type safety
:::

:::{grid-item-card} {octicon}`play;1.5em;sd-mr-1` Execution
:link: ../guides/execution
:link-type: doc
:link-alt: Execution guide

Multi-environment execution with executors for local, Docker, Slurm, Kubernetes, and cloud platforms
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Management
:link: ../guides/management
:link-type: doc
:link-alt: Management guide

Experiment lifecycle management with metadata tracking, logging, and reproducibility
:::

::::

## Why Use NeMo Run?

NeMo Run addresses critical issues in ML experiment management through its unique approach:

### 🔧 **Configuration Flexibility**

NeMo Run's Python-based configuration system provides unprecedented flexibility:

- **Type-Safe Configurations**: Automatic validation using Python's type annotations
- **Nested Configuration Support**: Intuitive dot notation for complex parameter hierarchies
- **Fiddle Integration**: Built on Google's Fiddle framework for robust configuration management
- **YAML Interoperability**: Support for external configuration files with seamless Python integration

### 🚀 **Execution Modularity**

The framework's execution system enables true environment independence:

- **Executor Abstraction**: Mix and match tasks with different execution environments
- **Multi-Platform Support**: Local, Docker, Slurm, Kubernetes, and cloud platforms
- **Code Packaging**: Intelligent packaging strategies (Git archive, pattern-based, hybrid)
- **Launcher Integration**: Support for torchrun, fault tolerance, and custom launchers

### 📊 **Experiment Management**

Comprehensive experiment tracking and management capabilities:

- **Metadata Preservation**: Automatic capture of configurations, logs, and artifacts
- **Reproducibility**: One-command experiment reconstruction from metadata
- **Status Monitoring**: Real-time experiment status and log access
- **Dependency Management**: Complex workflow orchestration with task dependencies

## Target Users

NeMo Run is designed for ML practitioners who need robust experiment management:

- **ML Researchers**: Conducting experiments across multiple environments with full reproducibility
- **ML Engineers**: Building production ML pipelines with consistent configuration management
- **DevOps Engineers**: Managing ML infrastructure across diverse computing platforms
- **Data Scientists**: Prototyping and scaling ML experiments with minimal infrastructure overhead

## Key Technologies

NeMo Run leverages modern Python ecosystem technologies:

- **Fiddle**: Google's configuration framework for type-safe, composable configurations
- **TorchX**: PyTorch's job submission framework for distributed execution
- **Docker**: Container-based execution for consistent environments
- **Ray**: Distributed computing framework integration for scalable ML workloads
- **Typer**: Modern CLI framework for rich command-line interfaces

## Core Architecture

NeMo Run's architecture follows a clean separation of concerns:

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`code;1.5em;sd-mr-1` Configuration Layer
:link: ../guides/configuration
:link-type: doc
:link-alt: Configuration guide

Fiddle-based configuration system with type safety and validation
:::

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Execution Layer
:link: ../guides/execution
:link-type: doc
:link-alt: Execution guide

Executor abstraction with multi-platform support and intelligent packaging
:::

:::{grid-item-card} {octicon}`database;1.5em;sd-mr-1` Management Layer
:link: ../guides/management
:link-type: doc
:link-alt: Management guide

Experiment lifecycle management with metadata tracking and reproducibility
:::

:::{grid-item-card} {octicon}`terminal;1.5em;sd-mr-1` Interface Layer
:link: ../reference/faqs
:link-type: doc
:link-alt: Reference

Rich CLI interface with type-safe argument parsing and configuration overrides
:::

::::

## Getting Started

Ready to start using NeMo Run? Begin with these essential guides:

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Quick Start
:link: ../get-started/index
:link-type: doc
:link-alt: Get started guide

Set up your first NeMo Run experiment in minutes
:::

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Key Features
:link: key-features
:link-type: doc
:link-alt: Key features

Explore the technical capabilities and implementation details
:::

::::

For detailed information about specific features, explore the [Configuration](../guides/configuration), [Execution](../guides/execution), and [Management](../guides/management) guides.
