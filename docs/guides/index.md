---
description: "Comprehensive guides for NeMo Run features including configuration, execution, management, and Ray integration."
tags: ["guides", "configuration", "execution", "management", "ray", "tutorials"]
categories: ["guides"]
---

(guides)=

# About NeMo Run Guides

Welcome to the NeMo Run guides. These comprehensive guides will help you master the core features and capabilities of NeMo Run for ML experiment management.

## Guides

::::{grid} 1 1 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Configuration
:link: configuration
:link-type: doc
:link-alt: Configuration guide

Learn how to configure your ML experiments with type-safe, flexible configuration management.
:::

:::{grid-item-card} {octicon}`play;1.5em;sd-mr-1` Execution
:link: execution
:link-type: doc
:link-alt: Execution guide

Execute your experiments across local, Docker, Slurm, Kubernetes, and cloud environments.
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Management
:link: management
:link-type: doc
:link-alt: Management guide

Manage and monitor your experiments with comprehensive tracking and reproducibility.
:::

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Deploy Ray Clusters and Jobs
:link: ray
:link-type: doc
:link-alt: Deploy Ray Clusters and Jobs

Deploy and manage Ray clusters and jobs for scalable distributed computing.
:::

:::{grid-item-card} {octicon}`terminal;1.5em;sd-mr-1` CLI Interface
:link: cli
:link-type: doc
:link-alt: CLI guide

Create command-line interfaces with rich argument parsing and factory functions.
:::

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Packaging
:link: packaging
:link-type: doc
:link-alt: Packaging guide

Deploy your code using Git archives, pattern matching, or hybrid packaging strategies.
:::

:::{grid-item-card} {octicon}`shield;1.5em;sd-mr-1` Administration
:link: admin
:link-type: doc
:link-alt: Administration guide

Deploy, maintain, and operate NeMo Run in production environments with comprehensive administrative procedures.
:::

::::

## Get Started

If you're new to NeMo Run, we recommend following these guides in order:

1. **Configuration** - Start here to understand how to configure your experiments
2. **Execution** - Learn how to run your configured experiments
3. **Management** - Discover how to track and manage your experiments
4. **Packaging** - Understand how to deploy your code to remote environments
5. **CLI Interface** - Create command-line tools for your workflows
6. **Deploy Ray Clusters and Jobs** - Scale up with distributed Ray clusters
7. **Administration** - Deploy and maintain NeMo Run in production environments

## What You'll Learn

Each guide provides:

- **Step-by-step instructions** with practical examples
- **Code samples** that you can run immediately
- **Best practices** for production use
- **Troubleshooting tips** for common issues
- **Advanced features** for power users

## Prerequisites

Before diving into these guides, make sure you have:

- NeMo Run installed (see [Installation Guide](../get-started/install))
- Basic Python knowledge
- Access to computing resources (local, cloud, or cluster)

## Need Help?

- Check the [FAQs](../faqs) for common questions
- Explore the [About section](../about/index) for conceptual information
- Review the [tutorials](../get-started/tutorials) for hands-on examples

```{toctree}
:hidden:
:maxdepth: 2

configuration
execution
management
ray
cli
packaging
admin
```
