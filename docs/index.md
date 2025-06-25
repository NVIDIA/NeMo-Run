---
description: "Explore comprehensive documentation for our software platform, including tutorials, feature guides, and deployment instructions."
tags: ["overview", "quickstart", "getting-started"]
categories: ["getting-started"]
---

(template-home)=

# {{ product_name }} Documentation

Welcome to the {{ product_name_short }} documentation.

## Introduction to {{ product_name_short }}

Learn about the {{ product_name_short }}, how it works at a high level, and its key features.

## NeMo Run Documentation

::::{grid} 1 1 1 3
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`settings;1.5em;sd-mr-1` Configuration
:link: configuration
:link-type: ref
:link-alt: Configuration guide

Learn how to configure your ML experiments and environments.
:::

:::{grid-item-card} {octicon}`play;1.5em;sd-mr-1` Execution
:link: execution
:link-type: ref
:link-alt: Execution guide

Execute your configured experiments across various computing environments.
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Management
:link: management
:link-type: ref
:link-alt: Management guide

Manage and monitor your running experiments and results.
:::

::::

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Ray Integration
:link: admin/integrations/ray
:link-type: ref
:link-alt: Ray integration guide

Learn how to use NeMo Run with Ray clusters.
:::

:::{grid-item-card} {octicon}`terminal;1.5em;sd-mr-1` CLI Reference
:link: reference/cli
:link-type: ref
:link-alt: CLI reference guide

Command-line interface reference and usage.
:::

::::

## NeMo Run Workflows

::::{grid} 1 1 2 3
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`play;1.5em;sd-mr-1` Execution
:link: execution
:link-type: ref
:link-alt: Execution documentation home

Run and manage jobs across local, cluster, and cloud environments.
:::

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Packaging
:link: packaging
:link-type: ref
:link-alt: Packaging documentation home

Package your code and dependencies for portable, reproducible runs.
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Experiments
:link: experiments
:link-type: ref
:link-alt: Experiments documentation home

Track, organize, and analyze your machine learning experiments.
:::

:::{grid-item-card} {octicon}`shield-lock;1.5em;sd-mr-1` Tunneling
:link: tunneling
:link-type: ref
:link-alt: Tunneling documentation home

Securely connect to remote resources and forward ports for distributed jobs.
:::

:::{grid-item-card} {octicon}`settings;1.5em;sd-mr-1` Configuration
:link: configuration
:link-type: ref
:link-alt: Configuration documentation home

Define, manage, and validate parameters and scripts for your workflows.
:::

:::{grid-item-card} {octicon}`terminal;1.5em;sd-mr-1` CLI
:link: cli
:link-type: ref
:link-alt: CLI documentation home

Command-line interface for all NeMo Run features and workspace management.
:::

::::

## Tutorial Highlights

::::{grid} 1 1 1 3
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Execution Tutorials
:link: execution/tutorials/index.md
:link-type: doc
:link-alt: Execution tutorial collection

Step-by-step guides for using execution backends.
:::

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Packaging Tutorials
:link: packaging/tutorials/index.md
:link-type: doc
:link-alt: Packaging tutorial collection

How to package and distribute your code.
:::

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Experiments Tutorials
:link: experiments/tutorials/index.md
:link-type: doc
:link-alt: Experiments tutorial collection

Guides for managing and tracking experiments.
:::

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Tunneling Tutorials
:link: tunneling/tutorials/index.md
:link-type: doc
:link-alt: Tunneling tutorial collection

Secure remote access and port forwarding guides.
:::

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Configuration Tutorials
:link: configuration/tutorials/index.md
:link-type: doc
:link-alt: Configuration tutorial collection

Parameter and script management tutorials.
:::

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` CLI Tutorials
:link: cli/tutorials/index.md
:link-type: doc
:link-alt: CLI tutorial collection

Command-line usage and workspace management.
:::

::::

## Install & Deploy Guides

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Deployment Patterns
:link: admin-deployment
:link-type: ref
:link-alt: Deployment and configuration guides

Learn how to deploy and configure your environment
:::

:::{grid-item-card} {octicon}`plug;1.5em;sd-mr-1` Integration Patterns
:link: admin-integrations
:link-type: ref
:link-alt: Integration and connection guides

Connect with external systems and services
:::

::::

---

::::{toctree}
:hidden:
Home <self>
::::

::::{toctree}
:hidden:
:caption: About
:maxdepth: 1
about/index.md
about/key-features.md
about/concepts/index.md
about/release-notes/index.md
::::

::::{toctree}
:hidden:
:caption: Get Started
:maxdepth: 2

get-started/index.md
Feature Set A Quickstart <get-started/feature-set-a.md>
Feature Set B Quickstart <get-started/feature-set-b.md> :only: not ga
::::

::::{toctree}
:hidden:
:caption: Execution
:maxdepth: 2
execution/index.md
execution/tutorials/index.md
execution/backends/index.md
::::

::::{toctree}
:hidden:
:caption: Packaging
:maxdepth: 2
packaging/index.md
packaging/tutorials/index.md
packaging/strategies/index.md
::::

::::{toctree}
:hidden:
:caption: Experiments
:maxdepth: 2
experiments/index.md
experiments/tutorials/index.md
experiments/tracking/index.md
::::

::::{toctree}
:hidden:
:caption: Tunneling
:maxdepth: 2
tunneling/index.md
tunneling/tutorials/index.md
tunneling/connections/index.md
::::

::::{toctree}
:hidden:
:caption: Configuration
:maxdepth: 2
configuration/index.md
configuration/tutorials/index.md
configuration/schemas/index.md
::::

::::{toctree}
:hidden:
:caption: CLI
:maxdepth: 2
cli/index.md
cli/tutorials/index.md
cli/workspace/index.md
::::
