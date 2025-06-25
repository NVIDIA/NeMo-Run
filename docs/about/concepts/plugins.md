---
description: "Learn about NeMo Run's plugin system for extending functionality with custom executors, packagers, and integrations."
tags: ["plugins", "concepts", "extensibility", "custom", "integrations"]
categories: ["about"]
---

(about-concepts-plugins)=
# Plugin System

NeMo Run's plugin system enables you to extend the framework's functionality by creating custom executors, packagers, and integrations. The plugin architecture provides a clean, extensible way to add new capabilities while maintaining compatibility with the core system.

## Core Concepts

### Plugin Architecture

NeMo Run's plugin system provides:
- **Extensibility**: Add new functionality without modifying core code
- **Modularity**: Plugins are self-contained and independently versioned
- **Compatibility**: Plugins work seamlessly with existing NeMo Run features
- **Discovery**: Automatic plugin discovery and registration
- **Configuration**: Plugin configuration through NeMo Run's config system

```python
import nemo_run as run
from nemo_run.plugins import Plugin, PluginRegistry

class CustomExecutorPlugin(Plugin):
    """Custom executor plugin for NeMo Run."""

    def register(self, registry: PluginRegistry):
        """Register the plugin with NeMo Run."""
        registry.register_executor("custom", CustomExecutor)
        registry.register_packager("custom", CustomPackager)
```

## Plugin Types

### Executor Plugins

Create custom executors for new execution environments:

```python
from nemo_run.executors import Executor
from nemo_run.plugins import Plugin, PluginRegistry

class KubernetesExecutor(Executor):
    """Custom executor for Kubernetes clusters."""

    def __init__(self, namespace="default", replicas=1, **kwargs):
        super().__init__(**kwargs)
        self.namespace = namespace
        self.replicas = replicas

    def execute(self, task, config):
        """Execute task on Kubernetes cluster."""
        # Implement Kubernetes-specific execution logic
        print(f"Executing on Kubernetes namespace: {self.namespace}")
        return super().execute(task, config)

class KubernetesPlugin(Plugin):
    """Plugin for Kubernetes integration."""

    def register(self, registry: PluginRegistry):
        registry.register_executor("kubernetes", KubernetesExecutor)

# Usage
executor = run.Config(KubernetesExecutor, namespace="ml-jobs", replicas=3)
```

### Packager Plugins

Create custom packagers for specialized packaging needs:

```python
from nemo_run.packagers import Packager
from nemo_run.plugins import Plugin, PluginRegistry

class DockerImagePackager(Packager):
    """Custom packager that creates Docker images."""

    def __init__(self, base_image="python:3.9", dockerfile_template=None, **kwargs):
        super().__init__(**kwargs)
        self.base_image = base_image
        self.dockerfile_template = dockerfile_template

    def package(self, source_dir, target_dir):
        """Create Docker image from source code."""
        # Implement Docker image creation logic
        print(f"Creating Docker image from {source_dir}")
        # Build Docker image, push to registry, etc.
        return target_dir

class DockerPlugin(Plugin):
    """Plugin for Docker integration."""

    def register(self, registry: PluginRegistry):
        registry.register_packager("docker", DockerImagePackager)

# Usage
packager = run.Config(
    DockerImagePackager,
    base_image="nvidia/cuda:11.8-devel-ubuntu20.04",
    dockerfile_template="custom.Dockerfile"
)
```

### Integration Plugins

Create plugins for external system integrations:

```python
from nemo_run.plugins import Plugin, PluginRegistry

class MLflowIntegration:
    """Integration with MLflow for experiment tracking."""

    def __init__(self, tracking_uri="http://localhost:5000"):
        self.tracking_uri = tracking_uri

    def log_experiment(self, experiment, results):
        """Log experiment results to MLflow."""
        import mlflow
        mlflow.set_tracking_uri(self.tracking_uri)

        with mlflow.start_run():
            mlflow.log_params(experiment.config)
            mlflow.log_metrics(results.metrics)
            mlflow.log_artifacts(results.artifacts)

class MLflowPlugin(Plugin):
    """Plugin for MLflow integration."""

    def register(self, registry: PluginRegistry):
        registry.register_integration("mlflow", MLflowIntegration)

# Usage
integration = run.Config(MLflowIntegration, tracking_uri="http://mlflow:5000")
```

## Plugin Development

### Basic Plugin Structure

```python
from nemo_run.plugins import Plugin, PluginRegistry
from typing import Dict, Any

class MyCustomPlugin(Plugin):
    """My custom NeMo Run plugin."""

    def __init__(self, plugin_config: Dict[str, Any] = None):
        super().__init__()
        self.config = plugin_config or {}

    def register(self, registry: PluginRegistry):
        """Register plugin components with NeMo Run."""
        # Register executors
        registry.register_executor("my_executor", MyCustomExecutor)

        # Register packagers
        registry.register_packager("my_packager", MyCustomPackager)

        # Register integrations
        registry.register_integration("my_integration", MyCustomIntegration)

        # Register CLI commands
        registry.register_command("my-command", my_cli_function)

    def initialize(self, config: Dict[str, Any]):
        """Initialize plugin with configuration."""
        print(f"Initializing MyCustomPlugin with config: {config}")

    def cleanup(self):
        """Clean up plugin resources."""
        print("Cleaning up MyCustomPlugin")
```

### Plugin Configuration

```python
class ConfigurablePlugin(Plugin):
    """Plugin with rich configuration options."""

    def __init__(self,
                 api_key: str = None,
                 endpoint: str = "https://api.example.com",
                 timeout: int = 30,
                 retries: int = 3):
        super().__init__()
        self.api_key = api_key
        self.endpoint = endpoint
        self.timeout = timeout
        self.retries = retries

    def register(self, registry: PluginRegistry):
        registry.register_executor("configurable", ConfigurableExecutor)

    def validate_config(self) -> bool:
        """Validate plugin configuration."""
        if not self.api_key:
            raise ValueError("API key is required")
        if self.timeout <= 0:
            raise ValueError("Timeout must be positive")
        return True

# Usage with configuration
plugin_config = run.Config(
    ConfigurablePlugin,
    api_key="your-api-key",
    endpoint="https://custom-api.example.com",
    timeout=60,
    retries=5
)
```

### Plugin Dependencies

```python
class PluginWithDependencies(Plugin):
    """Plugin that depends on other plugins or packages."""

    def __init__(self):
        super().__init__()
        self.dependencies = ["mlflow", "tensorboard"]

    def check_dependencies(self) -> bool:
        """Check if required dependencies are available."""
        try:
            import mlflow
            import tensorboard
            return True
        except ImportError:
            print("Missing dependencies: mlflow, tensorboard")
            return False

    def register(self, registry: PluginRegistry):
        if not self.check_dependencies():
            raise RuntimeError("Required dependencies not available")

        registry.register_integration("mlflow_tensorboard", MLflowTensorboardIntegration)
```

## Plugin Discovery and Registration

### Automatic Discovery

```python
# NeMo Run automatically discovers plugins in the nemo_run_plugins namespace
# Create a setup.py for your plugin
from setuptools import setup, find_packages

setup(
    name="nemo-run-custom-plugin",
    version="1.0.0",
    packages=find_packages(),
    entry_points={
        "nemo_run_plugins": [
            "custom = my_plugin:CustomPlugin",
        ],
    },
    install_requires=[
        "nemo-run>=1.0.0",
    ],
)
```

### Manual Registration

```python
import nemo_run as run
from my_plugin import CustomPlugin

# Manually register plugin
plugin = CustomPlugin()
run.register_plugin(plugin)

# Use the plugin
executor = run.Config(CustomExecutor, custom_param="value")
```

### Plugin Configuration

```python
# Configure plugins through NeMo Run
run.configure_plugins({
    "custom_plugin": {
        "api_key": "your-api-key",
        "endpoint": "https://api.example.com",
        "timeout": 30
    },
    "mlflow_plugin": {
        "tracking_uri": "http://mlflow:5000",
        "experiment_name": "my_experiments"
    }
})
```

## Advanced Plugin Patterns

### Plugin Composition

```python
class CompositePlugin(Plugin):
    """Plugin that composes multiple other plugins."""

    def __init__(self):
        super().__init__()
        self.sub_plugins = []

    def add_plugin(self, plugin: Plugin):
        """Add a sub-plugin."""
        self.sub_plugins.append(plugin)

    def register(self, registry: PluginRegistry):
        """Register all sub-plugins."""
        for plugin in self.sub_plugins:
            plugin.register(registry)

# Usage
composite = CompositePlugin()
composite.add_plugin(MLflowPlugin())
composite.add_plugin(TensorboardPlugin())
composite.add_plugin(CustomExecutorPlugin())
```

### Plugin Lifecycle Management

```python
class LifecyclePlugin(Plugin):
    """Plugin with full lifecycle management."""

    def __init__(self):
        super().__init__()
        self.initialized = False
        self.resources = []

    def initialize(self, config: Dict[str, Any]):
        """Initialize plugin resources."""
        if self.initialized:
            return

        # Initialize resources
        self.resources.append(self.create_connection_pool())
        self.resources.append(self.setup_logging())

        self.initialized = True
        print("Plugin initialized successfully")

    def register(self, registry: PluginRegistry):
        """Register plugin components."""
        if not self.initialized:
            raise RuntimeError("Plugin must be initialized before registration")

        registry.register_executor("lifecycle", LifecycleExecutor)

    def cleanup(self):
        """Clean up plugin resources."""
        for resource in self.resources:
            resource.close()
        self.resources.clear()
        self.initialized = False
        print("Plugin cleaned up successfully")
```

### Plugin Testing

```python
import pytest
from nemo_run.plugins import PluginRegistry
from my_plugin import CustomPlugin

class TestCustomPlugin:
    """Test suite for custom plugin."""

    def setup_method(self):
        """Set up test environment."""
        self.registry = PluginRegistry()
        self.plugin = CustomPlugin()

    def test_plugin_registration(self):
        """Test that plugin registers correctly."""
        self.plugin.register(self.registry)

        # Check that components are registered
        assert "custom_executor" in self.registry.executors
        assert "custom_packager" in self.registry.packagers

    def test_plugin_configuration(self):
        """Test plugin configuration."""
        config = {"api_key": "test-key", "timeout": 30}
        self.plugin.initialize(config)

        assert self.plugin.api_key == "test-key"
        assert self.plugin.timeout == 30

    def test_plugin_cleanup(self):
        """Test plugin cleanup."""
        self.plugin.initialize({})
        self.plugin.cleanup()

        # Verify cleanup occurred
        assert not self.plugin.initialized
```

## Best Practices

### 1. Follow Plugin Conventions

```python
class WellStructuredPlugin(Plugin):
    """Well-structured plugin following conventions."""

    def __init__(self, **kwargs):
        super().__init__()
        # Store configuration
        self.config = kwargs

    def register(self, registry: PluginRegistry):
        """Register plugin components."""
        # Use descriptive names
        registry.register_executor("well_structured", WellStructuredExecutor)
        registry.register_packager("well_structured", WellStructuredPackager)

    def initialize(self, config: Dict[str, Any]):
        """Initialize with validation."""
        # Validate configuration
        self.validate_config(config)
        # Initialize resources
        self.setup_resources()

    def cleanup(self):
        """Clean up resources."""
        self.cleanup_resources()
```

### 2. Provide Good Documentation

```python
class DocumentedPlugin(Plugin):
    """Plugin with comprehensive documentation.

    This plugin provides integration with external system X.

    Features:
        - Feature 1: Description
        - Feature 2: Description

    Configuration:
        - api_key: API key for authentication
        - endpoint: API endpoint URL
        - timeout: Request timeout in seconds

    Usage:
        plugin = DocumentedPlugin(api_key="your-key")
        run.register_plugin(plugin)
    """

    def __init__(self, api_key: str, endpoint: str = "https://api.example.com"):
        super().__init__()
        self.api_key = api_key
        self.endpoint = endpoint
```

### 3. Handle Errors Gracefully

```python
class RobustPlugin(Plugin):
    """Plugin with robust error handling."""

    def register(self, registry: PluginRegistry):
        try:
            registry.register_executor("robust", RobustExecutor)
        except Exception as e:
            print(f"Warning: Failed to register executor: {e}")
            # Continue with other registrations

    def initialize(self, config: Dict[str, Any]):
        try:
            self.validate_config(config)
            self.setup_resources()
        except Exception as e:
            print(f"Error initializing plugin: {e}")
            raise
```

### 4. Version Compatibility

```python
class VersionedPlugin(Plugin):
    """Plugin with version compatibility checking."""

    def __init__(self):
        super().__init__()
        self.required_nemo_version = ">=1.0.0"
        self.plugin_version = "1.0.0"

    def check_compatibility(self) -> bool:
        """Check compatibility with NeMo Run version."""
        import nemo_run
        from packaging import version

        nemo_version = version.parse(nemo_run.__version__)
        required_version = version.parse(self.required_nemo_version.replace(">=", ""))

        return nemo_version >= required_version

    def register(self, registry: PluginRegistry):
        if not self.check_compatibility():
            raise RuntimeError(f"Plugin requires NeMo Run {self.required_nemo_version}")

        registry.register_executor("versioned", VersionedExecutor)
```

The plugin system is essential for extending NeMo Run's capabilities, allowing you to add custom functionality while maintaining the framework's core design principles and compatibility.
