---
description: "Learn about NeMo Run's core concepts, architectural patterns, and fundamental design principles for ML experiment management."
tags: ["concepts", "architecture", "design-patterns", "ml", "experiment-management"]
categories: ["about"]
---

(about-concepts)=
# Core Concepts

NeMo Run is built on several fundamental concepts that work together to provide a comprehensive ML experiment management solution. Understanding these concepts will help you make the most of NeMo Run's capabilities.

## Configuration Concepts

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`settings;1.5em;sd-mr-1` Configuration Objects
:link: about-concepts-configuration
:link-type: ref
:link-alt: Configuration concepts

Learn about run.Config, run.Partial, and how NeMo Run handles configuration management
:::

:::{grid-item-card} {octicon}`code;1.5em;sd-mr-1` Autoconvert Pattern
:link: about-concepts-autoconvert
:link-type: ref
:link-alt: Autoconvert concepts

Understand the @run.autoconvert decorator and automatic configuration conversion
:::

::::

## Execution Concepts

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Executor Pattern
:link: about-concepts-executors
:link-type: ref
:link-alt: Executor concepts

Learn about the executor abstraction and how it enables multi-environment execution
:::

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Packaging Strategies
:link: about-concepts-packaging
:link-type: ref
:link-alt: Packaging concepts

Understand code packaging strategies and how they work across different environments
:::

::::

## Management Concepts

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Experiment Lifecycle
:link: about-concepts-experiments
:link-type: ref
:link-alt: Experiment concepts

Learn about experiment creation, execution, monitoring, and reconstruction
:::

:::{grid-item-card} {octicon}`workflow;1.5em;sd-mr-1` Task Dependencies
:link: about-concepts-dependencies
:link-type: ref
:link-alt: Dependency concepts

Understand how to create complex workflows with task dependencies and orchestration
:::

::::

## Integration Concepts

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`terminal;1.5em;sd-mr-1` CLI Architecture
:link: about-concepts-cli
:link-type: ref
:link-alt: CLI concepts

Learn about the CLI architecture and how type-safe command-line interfaces work
:::

:::{grid-item-card} {octicon}`plug;1.5em;sd-mr-1` Plugin System
:link: about-concepts-plugins
:link-type: ref
:link-alt: Plugin concepts

Understand the plugin system for extending NeMo Run's functionality
:::

::::

## Architectural Principles

NeMo Run follows several key architectural principles that guide its design:

### 🔄 **Separation of Concerns**
- **Configuration**: Separate from execution logic
- **Execution**: Environment-agnostic task execution
- **Management**: Independent experiment lifecycle management
- **Interface**: Clean CLI and programmatic interfaces

### 🎯 **Type Safety**
- **Python Type Annotations**: Leveraged for validation and documentation
- **Runtime Validation**: Automatic parameter validation and error checking
- **IDE Support**: Full autocomplete and type checking support

### 🔌 **Extensibility**
- **Plugin Architecture**: Extensible system for custom functionality
- **Executor Abstraction**: Easy addition of new execution environments
- **Configuration Flexibility**: Support for custom configuration patterns

### 📊 **Reproducibility**
- **Metadata Capture**: Automatic experiment metadata preservation
- **Configuration Versioning**: Full configuration history and tracking
- **Artifact Management**: Comprehensive artifact collection and storage

### 🚀 **Scalability**
- **Distributed Execution**: Support for multi-node, multi-GPU experiments
- **Resource Management**: Intelligent resource allocation and cleanup
- **Parallel Processing**: Concurrent execution of independent tasks

## Design Patterns

NeMo Run employs several design patterns that make it powerful and flexible:

### **Builder Pattern**
Used in configuration objects (`run.Config`, `run.Partial`) to construct complex configurations step by step.

### **Strategy Pattern**
Implemented through the executor system, allowing different execution strategies for the same task.

### **Observer Pattern**
Used in experiment management for real-time status updates and log streaming.

### **Factory Pattern**
Applied in the CLI system for creating complex objects from simple string specifications.

### **Command Pattern**
Used in the CLI architecture to encapsulate command execution and provide rich interfaces.

## Key Abstractions

### **Configuration Objects**
- `run.Config`: Direct configuration objects that build to instances
- `run.Partial`: Partial configurations that build to callable objects
- `run.Script`: Configuration for script-based execution

### **Executors**
- `run.LocalExecutor`: Local execution with process isolation
- `run.DockerExecutor`: Containerized execution
- `run.SlurmExecutor`: HPC cluster execution
- `run.SkypilotExecutor`: Cloud platform execution

### **Packagers**
- `run.Packager`: Base packaging strategy
- `run.GitArchivePackager`: Git-based code packaging
- `run.PatternPackager`: Pattern-based file packaging
- `run.HybridPackager`: Combined packaging strategies

### **Experiments**
- `run.Experiment`: Main experiment management class
- Task dependencies and orchestration
- Metadata management and reconstruction

These concepts work together to provide a cohesive, powerful system for ML experiment management. Understanding these fundamentals will help you design effective workflows and make the most of NeMo Run's capabilities.

```{toctree}
:hidden:
:maxdepth: 2

Configuration <configuration>
Autoconvert <autoconvert>
Executors <executors>
Packaging <packaging>
Experiments <experiments>
Dependencies <dependencies>
CLI Architecture <cli>
Plugin System <plugins>
```
