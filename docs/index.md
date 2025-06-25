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

## Featureset Workflows

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Feature Set A
:link: feature-set-a
:link-type: ref
:link-alt: Feature Set A documentation home

Comprehensive tools and workflows for data processing and analysis.
Get started with our core feature set.
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Feature Set B  
:link: feature-set-b
:only: not ga
:link-type: ref
:link-alt: Feature Set B documentation home

Advanced integration capabilities and specialized processing tools.
Available in Early Access.
:::

::::

## Tutorial Highlights

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Feature Set A Tutorials
:link: feature-set-a-tutorials
:link-type: ref
:link-alt: Feature Set A tutorial collection

Step-by-step guides for getting the most out of Feature Set A
:::

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Feature Set B Tutorials
:link: feature-set-b-tutorials
:only: not ga
:link-type: ref
:link-alt: Feature Set B tutorial collection

Hands-on tutorials for Feature Set B workflows
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
:caption: (GA) Feature Set A
:maxdepth: 2
feature-set-a/index.md
Tutorials <feature-set-a/tutorials/index.md>
feature-set-a/category-a/index.md
::::

::::{toctree}
:hidden:
:caption: (EA) Feature Set B
:maxdepth: 2
:only: not ga 

feature-set-b/index.md
Tutorials <feature-set-b/tutorials/index.md>
feature-set-b/category-a/index.md
::::

::::{toctree}
:hidden:
:caption: Admin
:maxdepth: 2
admin/index.md
Deployment <admin/deployment/index.md>
Integrations <admin/integrations/index.md>
CI/CD <admin/cicd/index.md>
::::

::::{toctree}
:hidden:
:caption: Reference
:maxdepth: 2
reference/index.md
::::
