---
description: "Complete guide to NeMo Run packaging strategies including GitArchive, Pattern, and Hybrid packagers for code deployment."
tags: ["packaging", "deployment", "code", "archives", "remote-execution"]
categories: ["guides"]
---

(packaging)=

# NeMo Run Packaging Strategies

NeMo Run provides flexible packaging strategies to deploy your code to remote execution environments. Understanding these packaging options is crucial for ensuring your experiments run correctly across different computing environments.

## Overview

Packaging determines how your local code is transferred to remote execution environments. NeMo Run supports multiple packaging strategies:

- **Base Packager**: Simple pass-through packaging
- **Git Archive Packager**: Version-controlled code packaging
- **Pattern Packager**: File pattern-based packaging
- **Hybrid Packager**: Combine multiple packaging strategies

## Packaging Support Matrix

| Executor | Supported Packagers |
|----------|-------------------|
| LocalExecutor | `run.Packager` |
| DockerExecutor | All packagers |
| SlurmExecutor | All packagers |
| SkypilotExecutor | All packagers |
| DGXCloudExecutor | All packagers |
| LeptonExecutor | All packagers |

## Base Packager

The `run.Packager` is a simple pass-through packager that doesn't perform any special packaging operations.

```python
import nemo_run as run

# Simple passthrough packager
packager = run.Packager()

executor = run.DockerExecutor(
    container_image="pytorch/pytorch:latest",
    packager=packager
)
```

**Use Cases:**

- When your code is already available in the container
- For simple scripts that don't require complex packaging
- When using pre-built images with your code

## Git Archive Packager

The `run.GitArchivePackager` uses `git archive` to package version-controlled code, ensuring only committed changes are deployed.

### How It Works

1. **Base Path Detection**: Uses `git rev-parse --show-toplevel` to find the repository root
2. **Subpath (sub-directory path) Configuration**: Optionally defines a subpath within the repository
3. **Archive Creation**: Creates a tar.gz archive of the specified code
4. **Working Directory**: The extracted archive becomes the working directory for your job

### Basic Usage

```python
import nemo_run as run

# Package the entire repository
packager = run.GitArchivePackager()

# Package a specific subdirectory
packager = run.GitArchivePackager(subpath="src")

executor = run.SlurmExecutor(
    account="my_account",
    partition="gpu",
    packager=packager
)
```

### Directory Structure Examples

**Repository Structure:**

```
my_project/
├── docs/
├── src/
│   ├── models/
│   ├── data/
│   └── utils/
├── tests/
├── configs/
└── README.md
```

**With `subpath="src"`:**

```python
packager = run.GitArchivePackager(subpath="src")
```

**Working directory on remote:**

```
models/
data/
utils/
```

**With `subpath=""` (default):**

```python
packager = run.GitArchivePackager()  # or subpath=""
```

**Working directory on remote:**

```
docs/
src/
tests/
configs/
README.md
```

### Advanced Configuration

```python
import nemo_run as run

# Custom subpath and working directory
packager = run.GitArchivePackager(
    subpath="ml_experiments",  # Package from ml_experiments/
    working_dir="/workspace"   # Extract to /workspace on remote
)

# Package specific branches or commits
packager = run.GitArchivePackager(
    subpath="src",
    ref="feature/new-model"  # Git reference (branch, tag, commit)
)
```

### Best Practices

1. **Commit Your Changes**

   ```bash
   # Always commit before running remote jobs
   git add .
   git commit -m "Update model configuration"
   ```

2. **Use Meaningful Subpaths (sub-directory paths)**

   ```python
   # Good: Clear subpath
   packager = run.GitArchivePackager(subpath="experiments/transformer")

   # Avoid: Too broad
   packager = run.GitArchivePackager()  # Packages everything
   ```

3. **Handle Large Repositories**

   ```python
   # Package only necessary components
   packager = run.GitArchivePackager(subpath="src/models")
   ```

### Limitations

- **Uncommitted Changes**: `git archive` doesn't include uncommitted changes
- **Git Dependencies**: Requires a Git repository
- **Archive Size**: Large repositories create large archives

## Pattern Packager

The `run.PatternPackager` uses file patterns to package code that may not be under version control or when you need fine-grained control over what gets packaged.

### How It Works

1. **Pattern Matching**: Uses `find` command with specified patterns
2. **File Selection**: Includes only files matching the patterns
3. **Relative Paths**: Maintains relative directory structure
4. **Archive Creation**: Creates a tar.gz archive of matched files

### Basic Usage

```python
import nemo_run as run
import os

# Package all Python files in current directory
packager = run.PatternPackager(
    include_pattern="*.py",
    relative_path=os.getcwd()
)

# Package specific directories
packager = run.PatternPackager(
    include_pattern="src/**",
    relative_path=os.getcwd()
)
```

### Pattern Examples

```python
import nemo_run as run
import os

# Package Python files only
packager = run.PatternPackager(
    include_pattern="*.py",
    relative_path=os.getcwd()
)

# Package entire src directory
packager = run.PatternPackager(
    include_pattern="src/**",
    relative_path=os.getcwd()
)

# Package multiple patterns
packager = run.PatternPackager(
    include_pattern="src/**/*.py configs/*.yaml",
    relative_path=os.getcwd()
)

# Package with exclusions
packager = run.PatternPackager(
    include_pattern="src/**/*.py",
    exclude_pattern="src/**/*_test.py",
    relative_path=os.getcwd()
)

# Package from different base directory
packager = run.PatternPackager(
    include_pattern="**/*.py",
    relative_path="/path/to/project"
)
```

### Advanced Configuration

```python
import nemo_run as run
import os

# Complex pattern matching
packager = run.PatternPackager(
    include_pattern="src/**/*.py models/**/*.py configs/*.yaml",
    exclude_pattern="**/*_test.py **/__pycache__/**",
    relative_path=os.getcwd(),
    working_dir="/workspace/code"  # Custom working directory
)

# Package with custom archive name
packager = run.PatternPackager(
    include_pattern="src/**",
    relative_path=os.getcwd(),
    archive_name="my_experiment_code.tar.gz"
)
```

### Use Cases

1. **Non-Git Projects**

   ```python
   # Package code not under version control
   packager = run.PatternPackager(
       include_pattern="experiments/**/*.py",
       relative_path=os.getcwd()
   )
   ```

2. **Selective Packaging**

   ```python
   # Package only specific components
   packager = run.PatternPackager(
       include_pattern="models/transformer.py utils/data_loader.py",
       relative_path=os.getcwd()
   )
   ```

3. **Generated Code**

   ```python
   # Package generated artifacts
   packager = run.PatternPackager(
       include_pattern="generated/**/*.py",
       relative_path=os.getcwd()
   )
   ```

## Hybrid Packager

The `run.HybridPackager` allows you to combine multiple packaging strategies into a single archive, useful when you need different packaging approaches for different parts of your project.

### How It Works

1. **Multiple Packagers**: Combines several packagers into one
2. **Directory Organization**: Each packager's output goes to a specified directory
3. **Archive Merging**: Creates a single archive with organized structure
4. **Conflict Resolution**: Handles file name conflicts between packagers

### Basic Usage

```python
import nemo_run as run
import os

# Combine Git archive with pattern packager
hybrid_packager = run.HybridPackager(
    sub_packagers={
        "code": run.GitArchivePackager(subpath="src"),
        "configs": run.PatternPackager(
            include_pattern="configs/*.yaml",
            relative_path=os.getcwd()
        )
    }
)

executor = run.SlurmExecutor(
    account="my_account",
    packager=hybrid_packager
)
```

### Directory Structure

**Local Structure:**

```
project/
├── src/
│   ├── models/
│   └── utils/
├── configs/
│   ├── model.yaml
│   └── data.yaml
├── generated/
│   └── artifacts/
└── README.md
```

**Hybrid Packager Configuration:**

```python
hybrid_packager = run.HybridPackager(
    sub_packagers={
        "code": run.GitArchivePackager(subpath="src"),
        "configs": run.PatternPackager(
            include_pattern="configs/*.yaml",
            relative_path=os.getcwd()
        ),
        "artifacts": run.PatternPackager(
            include_pattern="generated/**",
            relative_path=os.getcwd()
        )
    }
)
```

**Remote Working Directory:**

```
code/
├── models/
└── utils/
configs/
├── model.yaml
└── data.yaml
artifacts/
└── generated/
    └── artifacts/
```

### Advanced Configuration

```python
import nemo_run as run
import os

# Extract at root (no subdirectories)
hybrid_packager = run.HybridPackager(
    sub_packagers={
        "": run.GitArchivePackager(subpath="src"),  # Extract to root
        "configs": run.PatternPackager(
            include_pattern="configs/*.yaml",
            relative_path=os.getcwd()
        )
    },
    extract_at_root=True  # All contents go to root
)

# Custom working directory
hybrid_packager = run.HybridPackager(
    sub_packagers={
        "code": run.GitArchivePackager(subpath="src"),
        "configs": run.PatternPackager(
            include_pattern="configs/*.yaml",
            relative_path=os.getcwd()
        )
    },
    working_dir="/workspace/experiment"
)
```

### Use Cases

1. **Mixed Version Control**

   ```python
   # Some code in Git, some not
   hybrid_packager = run.HybridPackager(
       sub_packagers={
           "code": run.GitArchivePackager(subpath="src"),
           "experiments": run.PatternPackager(
               include_pattern="experiments/**",
               relative_path=os.getcwd()
           )
       }
   )
   ```

2. **Different Packaging Strategies**

   ```python
   # Git for code, pattern for configs and data
   hybrid_packager = run.HybridPackager(
       sub_packagers={
           "code": run.GitArchivePackager(subpath="src"),
           "configs": run.PatternPackager(
               include_pattern="configs/*.yaml",
               relative_path=os.getcwd()
           ),
           "data": run.PatternPackager(
               include_pattern="data/processed/**",
               relative_path=os.getcwd()
           )
       }
   )
   ```

3. **Generated Code with Source**

   ```python
   # Source code from Git, generated code from patterns
   hybrid_packager = run.HybridPackager(
       sub_packagers={
           "source": run.GitArchivePackager(subpath="src"),
           "generated": run.PatternPackager(
               include_pattern="generated/**",
               relative_path=os.getcwd()
           )
       }
   )
   ```

## Best Practices

### 1. Choose the Right Packager

```python
# Use GitArchivePackager for version-controlled code
if is_git_repo():
    packager = run.GitArchivePackager(subpath="src")
else:
    packager = run.PatternPackager(include_pattern="src/**")

# Use HybridPackager for complex projects
if has_multiple_sources():
    packager = run.HybridPackager(sub_packagers={...})
```

### 2. Optimize Package Size

```python
# Package only necessary files
packager = run.PatternPackager(
    include_pattern="src/**/*.py configs/*.yaml",
    exclude_pattern="**/*_test.py **/__pycache__/** **/.git/**"
)
```

### 3. Handle Dependencies

```python
# Ensure dependencies are available
packager = run.GitArchivePackager(subpath="src")
executor = run.DockerExecutor(
    container_image="pytorch/pytorch:latest",  # Has dependencies
    packager=packager
)
```

### 4. Test Packaging Locally

```python
# Test packaging before remote execution
packager = run.GitArchivePackager(subpath="src")
# Use LocalExecutor to test packaging
executor = run.LocalExecutor(packager=packager)
```

## Troubleshoot

### Common Issues

1. **Git Archive Issues**

   ```bash
   # Error: Not a git repository
   # Solution: Ensure you're in a git repository
   git init
   git add .
   git commit -m "Initial commit"
   ```

2. **Pattern Matching Issues**

   ```python
   # Error: No files found
   # Solution: Check pattern and relative path
   packager = run.PatternPackager(
       include_pattern="src/**/*.py",
       relative_path=os.getcwd()  # Ensure this is correct
   )
   ```

3. **Large Archive Issues**

   ```python
   # Solution: Use more specific patterns
   packager = run.PatternPackager(
       include_pattern="src/models/**/*.py",  # More specific
       relative_path=os.getcwd()
   )
   ```

### Debug

1. **Check Package Contents**

   ```python
   # Use LocalExecutor to inspect packaging
   executor = run.LocalExecutor(packager=packager)
   ```

2. **Verify Patterns**

   ```bash
   # Test patterns locally
   find . -name "*.py" -path "src/**"
   ```

3. **Check Archive Size**

   ```python
   # Monitor archive size for large projects
   packager = run.GitArchivePackager(subpath="src")
   ```

## Examples

### Complete Example: ML Experiment Packaging

```python
import nemo_run as run
import os

def create_experiment_packager():
    """Create a comprehensive packager for ML experiments."""

    # Check if we're in a git repository
    if os.path.exists(".git"):
        # Use hybrid packager for git + generated content
        return run.HybridPackager(
            sub_packagers={
                "code": run.GitArchivePackager(subpath="src"),
                "configs": run.PatternPackager(
                    include_pattern="configs/*.yaml experiments/*.yaml",
                    relative_path=os.getcwd()
                ),
                "data": run.PatternPackager(
                    include_pattern="data/processed/**",
                    relative_path=os.getcwd()
                ),
                "artifacts": run.PatternPackager(
                    include_pattern="generated/**",
                    relative_path=os.getcwd()
                )
            }
        )
    else:
        # Use pattern packager for non-git projects
        return run.PatternPackager(
            include_pattern="src/**/*.py configs/*.yaml data/processed/**",
            exclude_pattern="**/*_test.py **/__pycache__/**",
            relative_path=os.getcwd()
        )

# Usage
packager = create_experiment_packager()
executor = run.SlurmExecutor(
    account="my_account",
    partition="gpu",
    packager=packager
)
```

This packaging system provides the flexibility to handle various project structures and deployment scenarios, ensuring your code is properly packaged and deployed to remote execution environments.
