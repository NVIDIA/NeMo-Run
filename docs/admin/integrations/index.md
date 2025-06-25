(admin-integrations)=
# Integrations

Use the following Admin guides to set up integrations for NeMo Run in a production environment.

## Prerequisites

- TBD

---

## Integration Options

::::{grid} 1 1 1 3
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Ray
:link: admin-integrations-ray
:link-type: ref
Integrate NeMo Run with Ray clusters for distributed computing
+++
{bdg-secondary}`distributed`
{bdg-secondary}`kubernetes`
{bdg-secondary}`slurm`
:::

:::{grid-item-card} {octicon}`database;1.5em;sd-mr-1` Spark
:link: admin-integrations-spark
:link-type: ref
Integrate NeMo Curator with Apache Spark for distributed processing
+++
{bdg-secondary}`batch-processing`
{bdg-secondary}`performance`
{bdg-secondary}`optimization`
:::

:::{grid-item-card} {octicon}`search;1.5em;sd-mr-1` Pinecone
:link: admin-integrations-pinecone
:link-type: ref
Enable semantic search for your documentation using Pinecone's hosted embeddings
+++
{bdg-secondary}`semantic-search`
{bdg-secondary}`embeddings`
{bdg-secondary}`documentation`
:::

::::

```{toctree}
:maxdepth: 4
:titlesonly:
:hidden:

Ray <ray>
Spark <spark>
Pinecone <pinecone>

```
