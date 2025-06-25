(admin-integrations)=
# Integrations

Use the following Admin guides to set up integrations for NeMo Curator in a production environment.

## Prerequisites

- TBD

---

## Integration Options

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

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

Spark <spark>
Pinecone <pinecone>

```
