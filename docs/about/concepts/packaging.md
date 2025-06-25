---
description: "Learn about NeMo Run's code packaging strategies and how they enable reproducible execution across different environments."
tags: ["packaging", "concepts", "deployment", "reproducibility", "code-distribution"]
categories: ["about"]
---

(about-concepts-packaging)=
# Packaging Strategies

NeMo Run's packaging system ensures that your code and dependencies are properly distributed to execution environments, enabling reproducible experiments across different platforms and configurations.

## Core Concepts

### Packaging Abstraction

A packager is responsible for:
- **Code Collection**: Gathering source files and dependencies
- **Dependency Resolution**: Identifying and including required packages
- **Environment Preparation**: Creating deployment packages
- **Distribution**: Transferring code to execution environments
- **Reproducibility**: Ensuring consistent execution across environments

```python
import nemo_run as run

# Configure a packager
packager = run.Config(
    run.GitArchivePackager,
    include_patterns=["*.py", "*.yaml", "*.json"],
    exclude_patterns=["__pycache__", "*.pyc", ".git"]
)

# Use with an executor
executor = run.Config(
    run.LocalExecutor,
    packager=packager,
    working_dir="./experiments"
)
```

## Built-in Packers

### Git Archive Packager

The `run.GitArchivePackager` creates packages from Git repositories:

```python
git_packager = run.Config(
    run.GitArchivePackager,
    include_patterns=["*.py", "*.yaml", "*.json", "*.md"],
    exclude_patterns=["__pycache__", "*.pyc", ".git", "*.log"],
    include_submodules=True,
    archive_format="tar.gz"
)
```

**Features:**
- Git-based versioning
- Submodule support
- Pattern-based filtering
- Archive compression

### Pattern Packager

The `run.PatternPackager` packages files based on patterns:

```python
pattern_packager = run.Config(
    run.PatternPackager,
    include_patterns=[
        "src/**/*.py",
        "configs/**/*.yaml",
        "scripts/**/*.sh",
        "requirements.txt"
    ],
    exclude_patterns=[
        "**/__pycache__/**",
        "**/*.pyc",
        "**/.DS_Store",
        "**/node_modules/**"
    ],
    base_dir="/path/to/project"
)
```

**Features:**
- Flexible pattern matching
- Directory-based packaging
- Custom base directories
- Fine-grained control

### Hybrid Packager

The `run.HybridPackager` combines multiple packaging strategies:

```python
hybrid_packager = run.Config(
    run.HybridPackager,
    packagers=[
        run.Config(run.GitArchivePackager, include_patterns=["*.py"]),
        run.Config(run.PatternPackager, include_patterns=["data/**/*"])
    ],
    merge_strategy="union"
)
```

**Features:**
- Multiple packaging strategies
- Flexible merge strategies
- Complex packaging requirements
- Strategy composition

## Packaging Configuration

### File Selection

Control which files are included in your package:

```python
packager = run.Config(
    run.PatternPackager,
    include_patterns=[
        # Python source files
        "src/**/*.py",
        "tests/**/*.py",

        # Configuration files
        "configs/**/*.yaml",
        "configs/**/*.json",

        # Scripts and utilities
        "scripts/**/*.sh",
        "scripts/**/*.py",

        # Documentation
        "README.md",
        "docs/**/*.md",

        # Dependencies
        "requirements.txt",
        "setup.py"
    ],
    exclude_patterns=[
        # Python cache
        "**/__pycache__/**",
        "**/*.pyc",
        "**/*.pyo",

        # Version control
        ".git/**",
        ".gitignore",

        # IDE files
        ".vscode/**",
        ".idea/**",

        # OS files
        "**/.DS_Store",
        "**/Thumbs.db",

        # Logs and temporary files
        "**/*.log",
        "**/tmp/**",
        "**/temp/**"
    ]
)
```

### Dependency Management

Handle Python dependencies:

```python
packager = run.Config(
    run.PatternPackager,
    include_patterns=["*.py", "requirements.txt"],
    dependency_strategy="requirements",  # or "pip", "conda"
    requirements_file="requirements.txt",
    include_dev_dependencies=False
)
```

### Archive Options

Configure archive format and compression:

```python
packager = run.Config(
    run.GitArchivePackager,
    archive_format="tar.gz",  # or "zip", "tar"
    compression_level=6,
    preserve_permissions=True
)
```

## Packaging Patterns

### Development Packaging

For development and testing:

```python
dev_packager = run.Config(
    run.PatternPackager,
    include_patterns=[
        "src/**/*.py",
        "tests/**/*.py",
        "configs/**/*.yaml"
    ],
    exclude_patterns=["**/__pycache__/**", "**/*.pyc"],
    base_dir="."
)
```

### Production Packaging

For production deployments:

```python
prod_packager = run.Config(
    run.GitArchivePackager,
    include_patterns=[
        "src/**/*.py",
        "configs/**/*.yaml",
        "scripts/**/*.sh",
        "requirements.txt"
    ],
    exclude_patterns=[
        "**/tests/**",
        "**/docs/**",
        "**/examples/**",
        "**/__pycache__/**"
    ],
    include_submodules=True
)
```

### Minimal Packaging

For lightweight deployments:

```python
minimal_packager = run.Config(
    run.PatternPackager,
    include_patterns=[
        "src/main.py",
        "configs/config.yaml"
    ],
    exclude_patterns=["**/*"],
    base_dir="."
)
```

## Integration with Executors

### Local Execution

```python
executor = run.Config(
    run.LocalExecutor,
    packager=run.Config(run.PatternPackager, include_patterns=["*.py"]),
    working_dir="./experiments"
)
```

### Docker Execution

```python
executor = run.Config(
    run.DockerExecutor,
    packager=run.Config(
        run.GitArchivePackager,
        include_patterns=["*.py", "requirements.txt"]
    ),
    image="python:3.9",
    working_dir="/workspace"
)
```

### Slurm Execution

```python
executor = run.Config(
    run.SlurmExecutor,
    packager=run.Config(
        run.HybridPackager,
        packagers=[
            run.Config(run.GitArchivePackager, include_patterns=["*.py"]),
            run.Config(run.PatternPackager, include_patterns=["data/**/*"])
        ]
    ),
    working_dir="/scratch/experiments"
)
```

## Custom Packers

Create custom packagers for specific needs:

```python
class CustomPackager(run.Packager):
    def __init__(self, custom_param=None, **kwargs):
        super().__init__(**kwargs)
        self.custom_param = custom_param

    def package(self, source_dir, target_dir):
        """Custom packaging logic."""
        # Implement your custom packaging logic
        print(f"Packaging with custom parameter: {self.custom_param}")

        # Example: Copy specific files with custom logic
        import shutil
        import os

        for root, dirs, files in os.walk(source_dir):
            for file in files:
                if file.endswith('.py'):
                    src_path = os.path.join(root, file)
                    dst_path = os.path.join(target_dir, file)
                    shutil.copy2(src_path, dst_path)

# Use custom packager
custom_packager = run.Config(
    CustomPackager,
    custom_param="special_value"
)
```

## Packaging Lifecycle

### 1. Analysis Phase

The packager analyzes the source directory:
- Scans for files matching include patterns
- Filters out files matching exclude patterns
- Resolves dependencies
- Determines packaging strategy

### 2. Collection Phase

The packager collects required files:
- Copies source files
- Gathers dependencies
- Preserves directory structure
- Handles special files (symlinks, permissions)

### 3. Packaging Phase

The packager creates the deployment package:
- Creates archive format
- Applies compression
- Generates metadata
- Validates package integrity

### 4. Distribution Phase

The packager distributes the package:
- Transfers to execution environment
- Extracts package contents
- Sets up execution environment
- Verifies package contents

## Best Practices

### 1. Use Appropriate Patterns

```python
# Good: Specific and comprehensive
packager = run.Config(
    run.PatternPackager,
    include_patterns=[
        "src/**/*.py",
        "configs/**/*.yaml",
        "scripts/**/*.sh",
        "requirements.txt"
    ],
    exclude_patterns=[
        "**/__pycache__/**",
        "**/*.pyc",
        "**/tests/**",
        "**/.git/**"
    ]
)

# Avoid: Too broad or too narrow
bad_packager = run.Config(
    run.PatternPackager,
    include_patterns=["**/*"],  # Too broad
    exclude_patterns=["*.pyc"]  # Too narrow
)
```

### 2. Consider Package Size

```python
# For large projects, be selective
large_project_packager = run.Config(
    run.PatternPackager,
    include_patterns=[
        "src/**/*.py",
        "configs/**/*.yaml"
    ],
    exclude_patterns=[
        "**/data/**",  # Exclude large data files
        "**/models/**",  # Exclude model files
        "**/logs/**",  # Exclude log files
        "**/__pycache__/**"
    ]
)
```

### 3. Handle Dependencies

```python
# Include dependency specifications
packager = run.Config(
    run.PatternPackager,
    include_patterns=["*.py", "requirements.txt"],
    dependency_strategy="requirements",
    requirements_file="requirements.txt"
)
```

### 4. Test Packaging

```python
# Test your packaging configuration
def test_packaging():
    packager = run.Config(run.PatternPackager, include_patterns=["*.py"])

    # Create a test package
    test_dir = "/tmp/test_package"
    packager.package(".", test_dir)

    # Verify contents
    import os
    files = os.listdir(test_dir)
    print(f"Packaged files: {files}")

    # Clean up
    import shutil
    shutil.rmtree(test_dir)
```

### 5. Version Control Integration

```python
# Use Git for versioned packaging
git_packager = run.Config(
    run.GitArchivePackager,
    include_patterns=["*.py", "*.yaml"],
    include_submodules=True,
    archive_format="tar.gz"
)
```

## Troubleshooting

### Common Issues

1. **Missing Files**: Check include/exclude patterns
2. **Large Packages**: Review what's being included
3. **Dependency Issues**: Verify requirements.txt inclusion
4. **Permission Errors**: Check file permissions and ownership

### Debugging

```python
# Enable debug logging
packager = run.Config(
    run.PatternPackager,
    include_patterns=["*.py"],
    debug=True,
    log_level="DEBUG"
)
```

The packaging system is essential for NeMo Run's reproducibility and portability, ensuring that your experiments can run consistently across different environments and platforms.
