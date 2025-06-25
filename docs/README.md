---
orphan: true
---

# Documentation Template

This is a comprehensive Sphinx documentation template designed for technical writers who need to create sophisticated, well-structured documentation for complex software products.

## What This Template Demonstrates

This template showcases advanced Sphinx documentation patterns and features:

(️-complex-structure)=
### 🏗️ **Complex Structure**
- Multi-level navigation with toctrees
- Product-based content organization (Product A, B, C)
- Hierarchical information architecture

### 🎨 **Modern Design**
- Grid-based layouts with responsive cards
- Rich visual elements (icons, badges, images)
- Professional styling with the Furo theme

### 🔗 **Advanced Navigation**
- Cross-references and internal linking
- Conditional content rendering
- Multi-section organization

(️-sphinx-extensions)=
### 🛠️ **Sphinx Extensions**
- MyST Markdown with advanced features
- Sphinx Design for grid layouts
- Custom extensions for specialized functionality

### 📊 **Content Patterns**
- Concept documentation with detailed explanations
- Tutorial and how-to guide structures
- Reference documentation organization
- Administrative and deployment guides

## Template Structure

```
docs/
├── about/                    # About the product/template
│   ├── concepts/            # Core concepts by product area
│   │   ├── product-a-concepts/
│   │   ├── product-b-concepts/
│   │   └── product-c-concepts/
│   ├── key-features.md
│   └── release-notes/
├── get-started/             # Quickstart guides
├── product-a-workflows/     # Product A documentation
│   ├── tutorials/
│   ├── load-data/
│   ├── process-data/
│   └── generate-reports/
├── product-b-integration/   # Product B documentation
│   ├── tutorials/
│   ├── load-data/
│   ├── process-data/
│   └── save-export/
├── product-c-analytics/     # Product C documentation
├── admin/                   # Administrative guides
│   ├── deployment/
│   └── integrations/
└── reference/              # Reference documentation
    └── infrastructure/
```

## How to Use This Template

1. **Clone or Download** this template
2. **Replace Content**: Update the placeholder content with your actual product information
3. **Customize Structure**: Modify the directory structure to match your product needs
4. **Update Configuration**: Edit `conf.py` with your project details
5. **Build Documentation**: Run `make html` to generate your documentation

## Key Files to Customize

- `conf.py` - Main Sphinx configuration
- `index.md` - Homepage with navigation grids
- `about/index.md` - Product overview
- Product directories - Replace with your actual product documentation

## Features Demonstrated

### Grid Layouts
The template uses `sphinx-design` for responsive grid layouts:

```markdown
::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} Title
:link: target-reference
:link-type: ref
Description text
+++
{bdg-secondary}`tag1` {bdg-secondary}`tag2`
:::

::::
```

### Conditional Content
Content can be conditionally included based on build configuration:

```markdown
:::{grid-item-card} Advanced Feature
:only: not ga
This content only appears in non-GA builds
:::
```

### Cross-References
Comprehensive linking system with labeled references:

```markdown
(my-reference-label)=
# Section Title

Link to this section: {ref}`my-reference-label`
```

## Building the Documentation

```bash
# Install dependencies
pip install -r requirements.txt

# Build HTML documentation
make html

# Serve locally for development
python -m http.server 8000 -d _build/html
```

## Customization Tips

1. **Replace Product Names**: Search and replace "Product A/B/C" with your actual product names
2. **Update Icons**: Change octicon icons to match your product themes
3. **Modify Color Scheme**: Update badge colors and theme settings
4. **Add Your Content**: Replace placeholder text with real documentation
5. **Extend Structure**: Add new sections as needed for your products

## Requirements

- Python 3.8+
- Sphinx 4.0+
- MyST Parser
- Sphinx Design
- Furo theme (or your preferred theme)

This template provides a solid foundation for creating professional, maintainable documentation that scales with your product complexity.

---

*This template is designed to be easily customizable for your specific documentation needs. The structure and patterns demonstrated here can be adapted to any software product or technical documentation project.*

# Documentation Development

- [Documentation Template](#documentation-template)
  - [What This Template Demonstrates](#what-this-template-demonstrates)
    - [🏗️ **Complex Structure**](#️-complex-structure)
    - [🎨 **Modern Design**](#-modern-design)
    - [🔗 **Advanced Navigation**](#-advanced-navigation)
    - [🛠️ **Sphinx Extensions**](#️-sphinx-extensions)
    - [📊 **Content Patterns**](#-content-patterns)
  - [Template Structure](#template-structure)
  - [How to Use This Template](#how-to-use-this-template)
  - [Key Files to Customize](#key-files-to-customize)
  - [Features Demonstrated](#features-demonstrated)
    - [Grid Layouts](#grid-layouts)
    - [Conditional Content](#conditional-content)
    - [Cross-References](#cross-references)
  - [Building the Documentation](#building-the-documentation)
  - [Customization Tips](#customization-tips)
  - [Requirements](#requirements)
- [Documentation Development](#documentation-development)
  - [Set Up the Documentation Environment](#set-up-the-documentation-environment)
  - [Build the Documentation](#build-the-documentation)
    - [Build Variants](#build-variants)
  - [Live Building](#live-building)
  - [Conditional Content for Different Build Types](#conditional-content-for-different-build-types)
    - [1. File-Level Exclusion (Recommended for Entire Sections)](#1-file-level-exclusion-recommended-for-entire-sections)
    - [2. Grid Card Conditional Rendering](#2-grid-card-conditional-rendering)
    - [3. Toctree Conditional Rendering](#3-toctree-conditional-rendering)
    - [Best Practices](#best-practices)
    - [Testing Conditional Content](#testing-conditional-content)
  - [Run Doctests (if present)](#run-doctests-if-present)
  - [Example: How to Write Doctests in Documentation](#example-how-to-write-doctests-in-documentation)
  - [MyST Substitutions in Code Blocks](#myst-substitutions-in-code-blocks)
    - [Configuration](#configuration)
    - [Usage](#usage)
      - [Basic MyST Substitutions in Text](#basic-myst-substitutions-in-text)
      - [MyST Substitutions in Code Blocks](#myst-substitutions-in-code-blocks-1)
    - [Template Language Protection](#template-language-protection)
      - [Protected Languages](#protected-languages)
      - [Pattern Protection](#pattern-protection)
    - [Mixed Usage Examples](#mixed-usage-examples)
      - [YAML with Mixed Syntax](#yaml-with-mixed-syntax)
      - [Ansible with Mixed Syntax](#ansible-with-mixed-syntax)
      - [Benefits](#benefits)


## Set Up the Documentation Environment

Before building or serving the documentation, set up the docs environment using the Makefile:

```sh
make docs-env
source .venv-docs/bin/activate
```

This will create a virtual environment in `.venv-docs` and install all required dependencies for building the documentation.

## Build the Documentation

To build the NeMo Curator documentation, run:

```sh
make docs-html
```

* The resulting HTML files are generated in a `_build/html` folder under the project `docs/` folder.
* The generated Python API docs are placed in `apidocs` under the `docs/` folder.

### Build Variants

The documentation supports different build variants:

- `make docs-html` - Default build (includes all content)
- `make docs-html-ga` - GA (General Availability) build (excludes EA-only content)
- `make docs-html-ea` - EA (Early Access) build (includes all content)

## Live Building

To serve the documentation with live updates as you edit, run:

```sh
make docs-live
```

Open a web browser and go to `http://localhost:8000` (or the port shown in your terminal) to view the output.

## Conditional Content for Different Build Types

The documentation system supports three ways to conditionally include/exclude content based on build tags (e.g., GA vs EA builds). All methods use the unified `:only:` syntax.

### 1. File-Level Exclusion (Recommended for Entire Sections)

Use frontmatter to exclude entire documents from specific builds:

```yaml
---
only: not ga
---

# This entire document will be excluded from GA builds
```

**Supported conditions:**
- `only: not ga` - Exclude from GA builds (EA-only content)
- `only: ga` - Include only in GA builds  
- `only: not ea` - Exclude from EA builds
- `only: internal` - Include only in internal builds

**Directory inheritance:** If a parent directory's `index.md` has an `only` condition, all child documents inherit it automatically.

### 2. Grid Card Conditional Rendering

Hide specific grid cards from certain builds:

```markdown
:::{grid-item-card} Video Curation Features
:link: video-overview  
:link-type: ref
:only: not ga
Content for EA-only features
+++
{bdg-secondary}`early-access`
:::
```

### 3. Toctree Conditional Rendering

Control navigation entries conditionally:

```markdown
# Global toctree condition (hides entire section)
::::{toctree}
:hidden:
:caption: Early Access Features
:only: not ga

ea-feature1.md
ea-feature2.md  
::::

# Inline entry conditions (hides individual entries)
::::{toctree}
:hidden:
:caption: Documentation

standard-doc.md
ea-only-doc.md :only: not ga
another-standard-doc.md
::::
```

### Best Practices

- **Use file-level exclusion** for entire documentation sections (better performance, no warnings)
- **Use grid/toctree conditions** for fine-grained control within shared documents
- **Be consistent** with condition syntax across all methods
- **Test both build variants** to ensure content appears/disappears correctly

### Testing Conditional Content

```bash
# Test default build (includes all content)
make docs-html

# Test GA build (excludes EA-only content)  
make docs-html-ga

# Verify content is properly hidden/shown in each build
```

## Run Doctests (if present)

Sphinx is configured to support running doctests in both Python docstrings and in Markdown code blocks with the `{doctest}` directive. However, as of now, there are **no real doctest examples** in the codebase—only the example below in this README. If you add doctest examples, you can run them manually with:

```sh
source .venv-docs/bin/activate
cd docs
sphinx-build -b doctest . _build/doctest
```

There is currently **no Makefile target** for running doctests; you must use the above command directly.

## Example: How to Write Doctests in Documentation

Any code in triple backtick blocks with the `{doctest}` directive will be tested if you add real examples. The format follows Python's doctest module syntax, where `>>>` indicates Python input and the following line shows the expected output. Here's an example:

```python
def add(x: int, y: int) -> int:
    """
    Adds two integers together.

    Args:
        x (int): The first integer to add.
        y (int): The second integer to add.

    Returns:
        int: The sum of x and y.

    Examples:
    ```{doctest}
    >>> add(1, 2)
    3
    ```

    """
    return x + y
```

## MyST Substitutions in Code Blocks

The documentation uses a custom Sphinx extension (`myst_codeblock_substitutions`) that enables MyST substitutions to work inside standard code blocks. This allows you to maintain consistent variables (like version numbers, URLs, product names) across your documentation while preserving template syntax in YAML and other template languages.

### Configuration

The extension is configured in `docs/conf.py`:

```python
# Add the extension
extensions = [
    # ... other extensions
    "myst_codeblock_substitutions",  # Our custom MyST substitutions in code blocks
]

# Define reusable variables
myst_substitutions = {
    "product_name": "NeMo Curator",
    "product_name_short": "Curator", 
    "company": "NVIDIA",
    "version": release,  # Uses the release variable from conf.py
    "current_year": "2025",
    "github_repo": "https://github.com/NVIDIA/NeMo-Curator",
    "docs_url": "https://docs.nvidia.com/nemo-curator",
    "support_email": "nemo-curator-support@nvidia.com",
    "min_python_version": "3.8",
    "recommended_cuda": "12.0+",
}
```

### Usage

#### Basic MyST Substitutions in Text
Use `{{ variable }}` syntax in regular markdown text:

```markdown
Welcome to {{ product_name }} version {{ version }}!

{{ product_name_short }} is developed by {{ company }}.
For support, contact {{ support_email }}.
```

#### MyST Substitutions in Code Blocks

The extension enables substitutions in standard code blocks:

```bash
# Install {{ product_name }} version {{ version }}
helm install my-release oci://nvcr.io/nvidia/nemo-curator --version {{ version }}
kubectl get pods -n {{ product_name_short }}-namespace
docker run --rm nvcr.io/nvidia/nemo-curator:{{ version }}
pip install nemo-curator=={{ version }}
```

### Template Language Protection

The extension intelligently protects template languages from unwanted substitutions:

#### Protected Languages

These languages are treated carefully to preserve their native `{{ }}` syntax:
- `yaml`, `yml` (Kubernetes, Docker Compose)
- `helm`, `gotmpl`, `go-template` (Helm charts)
- `jinja`, `jinja2`, `j2` (Ansible, Python templates)
- `ansible` (Ansible playbooks)
- `handlebars`, `hbs`, `mustache` (JavaScript templates)
- `django`, `twig`, `liquid`, `smarty` (Web framework templates)

#### Pattern Protection

The extension automatically detects and preserves common template patterns:
- `{{ .Values.something }}` (Helm values)
- `{{ ansible_variable }}` (Ansible variables)
- `{{ item.property }}` (Template loops)
- `{{- variable }}` (Whitespace control)
- `{{ range ... }}`, `{{ if ... }}` (Control structures)

### Mixed Usage Examples

#### YAML with Mixed Syntax

```yaml
# values.yaml - MyST substitutions work alongside Helm templates
image:
  repository: nvcr.io/nvidia/nemo-curator
  tag: {{ .Values.image.tag | default "latest" }}        # ← Helm template (preserved)

# Documentation URLs using MyST substitutions  
downloads:
  releaseUrl: "https://github.com/NVIDIA/NeMo-Curator/releases/download/v{{ version }}/nemo-curator.tar.gz"  # ← MyST substitution
  docsUrl: "{{ docs_url }}"                              # ← MyST substitution
  supportEmail: "{{ support_email }}"                    # ← MyST substitution

service:
  name: {{ include "nemo-curator.fullname" . }}          # ← Helm template (preserved)
  
env:
  - name: CURATOR_VERSION
    value: "{{ .Chart.AppVersion }}"                     # ← Helm template (preserved)
  - name: DOCS_VERSION  
    value: "{{ version }}"                               # ← MyST substitution
```

#### Ansible with Mixed Syntax  

```yaml
# MyST substitutions for documentation
- name: "Install {{ product_name }} version {{ version }}"     # ← MyST substitution
  shell: |
    wget {{ github_repo }}/releases/download/v{{ version }}/nemo-curator.tar.gz  # ← MyST substitution
    
  # Ansible templates preserved
  when: "{{ ansible_distribution }} == 'Ubuntu'"              # ← Ansible template (preserved)
  notify: "{{ handlers.restart_service }}"                    # ← Ansible template (preserved)
```

#### Benefits

1. **Single Source of Truth**: Update version numbers, URLs, and product names in one place (`conf.py`)
2. **Template Safety**: Won't break existing Helm charts, Ansible playbooks, or other templates
3. **Intelligent Processing**: Only processes simple variable names, preserves complex template syntax
4. **Cross-Format Support**: Works in bash, python, dockerfile, and other code blocks
5. **Maintainability**: Reduces copy-paste errors and keeps documentation in sync with releases

The extension automatically handles the complexity of mixed template syntax, so you can focus on writing great documentation without worrying about breaking existing templates.