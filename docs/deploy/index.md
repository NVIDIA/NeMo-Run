---
description: "Deploy and manage NeMo Run applications in production environments with comprehensive deployment guides."
tags: ["deploy", "production", "packaging", "ray", "administration"]
categories: ["deploy"]
---

(deploy)=

# Deploy NeMo Run

This section contains documentation for deploying and managing NeMo Run applications in production environments.

## Overview

The deployment section covers all aspects of getting your NeMo Run applications ready for production use, including packaging strategies, administrative tasks, and Ray cluster deployment.

## Deployment Options

::::{grid} 1 1 1 3
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Ray Clusters and Jobs
:link: ray
:link-type: doc
:link-alt: Deploy Ray Clusters and Jobs

Deploy and manage Ray clusters and jobs for scalable distributed computing.
:::

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Packaging Strategies
:link: packaging
:link-type: doc
:link-alt: NeMo Run Packaging Strategies

Deploy your code using Git archives, pattern matching, or hybrid packaging strategies.
:::

:::{grid-item-card} {octicon}`shield;1.5em;sd-mr-1` Administration
:link: admin
:link-type: doc
:link-alt: Manage NeMo Run Deployments

Deploy, configure, and maintain NeMo Run in production environments.
:::

::::

## What You'll Learn

- **Ray Deployment**: Set up and manage Ray clusters for distributed computing
- **Code Packaging**: Choose the right packaging strategy for your deployment needs
- **Production Configuration**: Configure NeMo Run for production environments
- **Monitoring and Maintenance**: Monitor and maintain your NeMo Run deployments

::::{toctree}
:hidden:
:maxdepth: 2
admin
packaging
ray
::::
