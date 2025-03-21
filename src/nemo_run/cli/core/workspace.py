# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Callable
import time
import os
import sys
import importlib
import functools
import json
import hashlib
import site
import importlib.util
import re
import tempfile
from importlib import metadata
from contextlib import nullcontext
import inspect


import catalogue
import importlib_metadata as metadata

from nemo_run.cli.cache import ImportTracker
from nemo_run.config import NEMORUN_HOME
from rich.console import Console

# Constants
NEMORUN_HOME = os.environ.get("NEMORUN_HOME", os.path.expanduser("~/.nemorun"))
SITE_PACKAGES = site.getsitepackages()[0]
CACHE_DIR = os.path.join(SITE_PACKAGES, "nemo_run", "cache")
CACHE_FILE = os.path.join(CACHE_DIR, "cli_cache.json")
INCLUDE_WORKSPACE_FILE = os.environ.get("INCLUDE_WORKSPACE_FILE", "true").lower() == "true"
POPULATE_CACHE = False  # This should be set appropriately based on your application logic

# File hash cache to avoid recomputing hashes for the same file
_file_hash_cache = {}


class ImportTracker:
    """Tracks imported modules during execution and efficiently validates their freshness."""
    
    def __init__(self):
        self.imported_modules = []
        self.module_info = {}
        
        # Get system temp directories for filtering
        self.temp_dirs = [
            os.path.realpath(tempfile.gettempdir()),
            '/tmp',
            '/var/tmp',
            '/var/folders'
        ]
        
        # Get stdlib paths for filtering
        self.stdlib_paths = self._get_stdlib_paths()
    
    def _get_stdlib_paths(self) -> list[str]:
        """Get paths to the standard library modules to filter them out."""
        stdlib_paths = []
        
        for path in sys.path:
            if path and os.path.exists(path):
                if any(pattern in path for pattern in [
                    "lib/python", 
                    "lib64/python",
                    "lib/Python",
                    "Python.framework"
                ]):
                    stdlib_paths.append(os.path.realpath(path))
                    
        lib_paths = [p for p in sys.path if os.path.basename(p) == 'lib-dynload']
        stdlib_paths.extend([os.path.realpath(p) for p in lib_paths])
        
        return stdlib_paths
    
    def find_spec(self, fullname: str, path: str | None = None, target: Any | None = None):
        """Record imported module and capture its location information."""
        self.imported_modules.append(fullname)
        
        # Immediately capture module info if it's already loaded
        if fullname in sys.modules:
            self._capture_module_info(fullname)
        
        return None
    
    def get_imported_modules(self) -> list[str]:
        """Return list of imported module names."""
        return self.imported_modules
    
    # Helper methods for filtering modules
    def _is_in_temp_dir(self, file_path: str) -> bool:
        if not file_path:
            return False
        real_path = os.path.realpath(file_path)
        return any(real_path.startswith(temp_dir) for temp_dir in self.temp_dirs)
    
    def _is_stdlib_module(self, file_path: str) -> bool:
        if not file_path:
            return False
        real_path = os.path.realpath(file_path)
        return any(real_path.startswith(stdlib_path) for stdlib_path in self.stdlib_paths)
    
    def _is_valid_python_module(self, file_path: str) -> bool:
        if not file_path:
            return False
        if not (file_path.endswith('.py') or file_path.endswith('.so') or file_path.endswith('.pyd')):
            return False
        if '__pycache__' in file_path or file_path.endswith('.pyc'):
            return False
        return os.path.isfile(file_path)
    
    def _is_transient_module(self, module_name: str, file_path: str) -> bool:
        if not module_name or not file_path:
            return True
        patterns = [
            r'_remote_module_non_scriptab',
            r'__temp__',
            r'_anonymous_',
            r'dynamically_generated_',
            r'ipykernel_launcher',
            r'xdist',
        ]
        return (any(re.search(pattern, module_name) for pattern in patterns) or
                any(re.search(pattern, file_path) for pattern in patterns))
    
    def _capture_module_info(self, module_name: str):
        """Capture metadata about a module for cache validation."""
        if module_name in self.module_info:
            return
            
        # Skip built-in modules
        if module_name in sys.builtin_module_names:
            return
            
        # Skip common Python standard library modules
        if module_name in ('sys', 'os', 're', 'time', 'json', 'math', 'random', 'datetime', 'collections'):
            return
            
        module = sys.modules.get(module_name)
        if not module:
            return
            
        # Skip modules without a file
        if not hasattr(module, "__file__") or not module.__file__:
            return
            
        file_path = module.__file__
        
        # Apply filters
        if (self._is_in_temp_dir(file_path) or
            self._is_stdlib_module(file_path) or
            not self._is_valid_python_module(file_path) or
            self._is_transient_module(module_name, file_path)):
            return
        
        # Determine if this is a site-packages module or editable mode
        is_site_package = file_path.startswith(SITE_PACKAGES)
        
        # For site packages, track the version
        if is_site_package:
            try:
                package_name = self._get_package_name(module_name)
                version = metadata.version(package_name) if package_name else None
                
                self.module_info[module_name] = {
                    "type": "site-package",
                    "file_path": file_path,
                    "package_name": package_name,
                    "version": version
                }
            except Exception:
                # Fallback to file metadata if we can't get version
                self._record_file_metadata(module_name, file_path)
        else:
            # For editable packages, track file metadata
            self._record_file_metadata(module_name, file_path)
    
    def _record_file_metadata(self, module_name: str, file_path: str):
        """Record file metadata (mtime) instead of full content hash."""
        try:
            mtime = os.path.getmtime(file_path)
            
            self.module_info[module_name] = {
                "type": "editable", 
                "file_path": file_path,
                "mtime": mtime
            }
        except Exception as e:
            print(f"Warning: Couldn't get metadata for {file_path}: {e}")
    
    def _get_package_name(self, module_name: str) -> str | None:
        """Try to determine the package name from the module name."""
        # First try the base name
        if self._is_valid_package(module_name):
            return module_name
            
        # Try parent packages
        parts = module_name.split('.')
        for i in range(1, len(parts)):
            potential_package = '.'.join(parts[:i])
            if self._is_valid_package(potential_package):
                return potential_package
                
        return None
    
    def _is_valid_package(self, name: str) -> bool:
        """Check if a name is a valid installed package."""
        try:
            metadata.version(name)
            return True
        except Exception:
            return False
    
    def update_cache(self):
        """Update the cache with module dependency information."""
        # Process any modules that were imported but not yet processed
        for module_name in self.imported_modules:
            if module_name not in self.module_info and module_name in sys.modules:
                self._capture_module_info(module_name)
        
        # Update the cache with the collected info
        cache_data = load_cache()
        
        # Initialize the module dependencies section if it doesn't exist
        if "module_dependencies" not in cache_data["metadata"]:
            cache_data["metadata"]["module_dependencies"] = {}
            
        # Update with our module info
        cache_data["metadata"]["module_dependencies"].update(self.module_info)
        
        # Save the updated cache
        save_cache(cache_data)


def load_cache() -> dict[str, Any]:
    """Load the existing cache if it exists."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:            
            # Remove the corrupted file
            try:
                os.remove(CACHE_FILE)
                print(f"Removed corrupted cache file: {CACHE_FILE}")
            except Exception as e:
                print(f"Failed to remove corrupted cache: {e}")
    
    # Return a fresh cache structure
    return {
        "metadata": {},
        "app": {
            "name": "nemo", 
            "help": "NeMo CLI", 
            "groups": {},
            "cli_structure": {}
        },
        "data": {"common_options": {}, "executors": [], "factories": {}}
    }


def save_cache(cache_data: dict[str, Any]) -> None:
    """Save the cache to the JSON file."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # Ensure the data is JSON serializable
    serializable_data = make_json_serializable(cache_data)
    
    with open(CACHE_FILE, "w") as f:
        json.dump(serializable_data, f)


def make_json_serializable(obj: Any) -> Any:
    """Convert an object to a JSON serializable form."""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return [make_json_serializable(item) for item in obj]
    elif callable(obj) and hasattr(obj, "__name__"):
        # For function objects, store their name and module
        return {
            "__type__": "function",
            "module": obj.__module__,
            "name": obj.__name__
        }
    elif hasattr(obj, "__dict__"):
        # For custom objects, convert to dict if possible
        return {
            "__type__": obj.__class__.__name__,
            "module": obj.__class__.__module__,
            "data": make_json_serializable(obj.__dict__)
        }
    elif inspect.isbuiltin(obj) or inspect.isfunction(obj) or inspect.ismethod(obj):
        # For any other callables
        return str(obj)
    else:
        return obj


def sync(verbose: bool = True, no_cache: bool = False) -> bool:
    """
    Synchronize the NeMo Run workspace and CLI cache.
    
    Checks if the cache is up to date, and rebuilds it if necessary.
    When verbose=True, shows detailed progress through rich visualizations.
    When verbose=False, runs silently with no output.
    
    Args:
        verbose: Whether to show detailed progress information
        no_cache: Force a full rebuild of the cache, ignoring any existing cache data
    
    Returns:
        bool: True if sync was successful
    """
    # Set up console only if verbose mode is enabled
    console = None
    if verbose:
        console = Console()
        console.print("[bold cyan]NeMo Run Workspace Sync[/bold cyan]")
    
    start_time = time.time()
    
    # Load existing cache
    cache_data = load_cache()
    
    # If no_cache is specified, skip validation and force a rebuild
    if no_cache:
        if verbose:
            console.print("[yellow]Forcing full cache rebuild...[/yellow]")
        success = _rebuild_cache(verbose, console)
    else:
        # Otherwise, run validation in a separate async function but call it synchronously
        is_valid = _run_async(validate_cache(cache_data, console))
        
        if is_valid:
            if verbose:
                console.print("[green]✓ Workspace cache is up-to-date[/green]")
                
                # Show the same detailed summary as we do after rebuilding
                _print_sync_summary(console)
                
            return True
        
        # If cache is invalid, rebuild it
        if verbose:
            console.print("Cache needs to be rebuilt...")
        
        # Rebuild the cache
        success = _rebuild_cache(verbose, console)
    
    # Show completion message if verbose
    if verbose and success:
        end_time = time.time()
        duration = end_time - start_time
        console.print(f"[green]✓ Workspace synced in {duration:.2f}s[/green]")
        
        # Show detailed summary of what was cached
        _print_sync_summary(console)
    
    return success


def _run_async(coro):
    """Run an async coroutine from a synchronous function."""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # If no event loop exists, create a new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(coro)


async def validate_cache(cache_data: dict[str, Any], console=None) -> bool:
    """
    Validate if the cache is up to date.
    
    Args:
        cache_data: The cache data to validate
        console: Optional console for verbose output
    
    Returns:
        bool: Whether the cache is valid and up to date
    """
    # Check if cache is empty
    if not cache_data or "metadata" not in cache_data:
        if console:
            console.print("[yellow]Cache is empty or missing metadata[/yellow]")
        return False
        
    metadata_info = cache_data.get("metadata", {})
    
    # Python version check
    current_python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if metadata_info.get("python_version") != current_python_version:
        if console:
            console.print(f"[yellow]Python version changed: {metadata_info.get('python_version')} → {current_python_version}[/yellow]")
        return False

    # Check library versions
    try:
        required_libs = {"nemo_run": metadata.version("nemo_run"), "typer": metadata.version("typer")}
        for lib, ver in metadata_info.get("library_versions", {}).items():
            if console:
                console.print(f"Checking {lib} version...", end="\r")
            
            try:
                current_ver = required_libs.get(lib)
                if current_ver != ver:
                    if console:
                        console.print(f"[yellow]{lib} version changed: {ver} → {current_ver}[/yellow]")
                    return False
            except metadata.PackageNotFoundError:
                if console:
                    console.print(f"[yellow]{lib} not found[/yellow]")
                return False
    except Exception as e:
        if console:
            console.print(f"[red]Error checking library versions: {e}[/red]")
        return False
    
    # Check workspace file
    stored_workspace_file = metadata_info.get("workspace_file")
    if stored_workspace_file:
        if console:
            console.print("Checking workspace file...", end="\r")
        
        current_workspace_file = _search_workspace_file()
        if current_workspace_file != stored_workspace_file:
            if console:
                console.print(f"[yellow]Workspace file changed: {stored_workspace_file} → {current_workspace_file}[/yellow]")
            return False
            
        if os.path.exists(stored_workspace_file):
            stats = os.stat(stored_workspace_file)
            if stats.st_mtime > metadata_info.get("workspace_metadata", {}).get("mtime", 0):
                if console:
                    console.print(f"[yellow]Workspace file has been modified[/yellow]")
                return False
    
    # Check editable packages if any are tracked
    if "editable_packages_metadata" in metadata_info:
        if console:
            console.print("Checking editable packages...", end="\r")
        
        for path, pkg_meta in metadata_info["editable_packages_metadata"].items():
            if not os.path.exists(path):
                if console:
                    console.print(f"[yellow]Editable package path no longer exists: {path}[/yellow]")
                return False
                
            # Check if any files were modified
            try:
                latest_mtime = pkg_meta.get("mtime", 0)
                current_mtime = os.path.getmtime(path)
                if current_mtime > latest_mtime:
                    if console:
                        console.print(f"[yellow]Editable package modified: {pkg_meta.get('package_name', path)}[/yellow]")
                    return False
            except Exception:
                # If we can't check the time, assume it's changed
                return False
    
    return True


def _rebuild_cache(verbose: bool = True, console=None) -> bool:
    """
    Rebuild the cache from scratch.
    
    Args:
        verbose: Whether to show detailed progress
        console: Optional console object for output
        
    Returns:
        bool: Whether the rebuild was successful
    """
    try:
        # Set up progress tracking if verbose
        if verbose:
            from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TextColumn("[green]{task.fields[status]}"),
                console=console
            )
            progress_context = progress
        else:
            # Use a dummy context manager when not verbose
            progress_context = nullcontext()
            progress = None
        
        # Track modules for metadata
        tracker = ImportTracker()
        sys.meta_path.insert(0, tracker)
        
        try:
            with progress_context:
                entrypoint_modules = []
                factory_modules = []
                
                # Step 1: CLI Structure
                if progress:
                    cli_task = progress.add_task("Caching CLI structure", status="loading entrypoints", total=3)
                
                # Load entrypoints
                entrypoints = metadata.entry_points().select(group="nemo_run.cli")
                loaded_count = 0
                
                for ep in entrypoints:
                    try:
                        if verbose and progress:
                            progress.update(cli_task, status=f"loading {ep.name}")
                        elif verbose:
                            console.print(f"Loading entrypoint: {ep.name}", end="\r")
                            
                        module = ep.load()
                        loaded_count += 1
                        
                        # Collect module name for metadata
                        if hasattr(module, "__module__"):
                            entrypoint_modules.append(module.__module__)
                    except Exception as e:
                        if verbose:
                            console.print(f"[yellow]Warning:[/yellow] Failed to load entrypoint {ep.name}: {e}")
                
                if progress:
                    progress.update(cli_task, advance=1, status=f"loaded {loaded_count} entrypoints")
                
                # Cache CLI structure
                if progress:
                    progress.update(cli_task, status="caching structure")
                cache_cli_structure()
                
                if progress:
                    progress.update(cli_task, advance=1, status="CLI structure cached")
                
                # Executor classes
                if progress:
                    progress.update(cli_task, status="caching executors")
                cache_executor_classes() 
                
                if progress:
                    progress.update(cli_task, advance=1, completed=True, status="executors cached")
                
                # Step 2: Workspace
                if progress:
                    workspace_task = progress.add_task("Processing workspace", status="searching", total=2)
                
                workspace_file_path = _search_workspace_file()
                if workspace_file_path:
                    if progress:
                        progress.update(workspace_task, advance=1, status=f"loading {os.path.basename(workspace_file_path)}")
                    elif verbose:
                        console.print(f"Loading workspace: {os.path.basename(workspace_file_path)}", end="\r")
                    
                    try:
                        # Load the workspace file
                        spec = importlib.util.spec_from_file_location("workspace", workspace_file_path)
                        assert spec
                        workspace_module = importlib.util.module_from_spec(spec)
                        assert spec.loader
                        spec.loader.exec_module(workspace_module)
                        
                        if progress:
                            progress.update(workspace_task, advance=1, completed=True, status="workspace loaded")
                    except Exception as e:
                        if progress:
                            progress.update(workspace_task, advance=1, completed=True, status=f"error: {str(e)}")
                        if verbose:
                            console.print(f"[yellow]Warning:[/yellow] Failed to load workspace file: {e}")
                else:
                    if progress:
                        progress.update(workspace_task, completed=True, status="no workspace file found")
                
                # Step 3: Update metadata
                if progress:
                    metadata_task = progress.add_task("Updating cache metadata", status="collecting", total=1)
                elif verbose:
                    console.print("Updating cache metadata...", end="\r")
                
                # Update the cache with collected dependencies
                tracker.update_cache()
                
                # Update metadata
                cache_data = load_cache()  # Reload after structure updates
                cache_data["metadata"] = update_cache_metadata(entrypoint_modules, factory_modules)
                save_cache(cache_data)
                
                if progress:
                    progress.update(metadata_task, completed=1, status="cache metadata updated")
            
            return True
            
        finally:
            # Clean up the import tracker
            if tracker in sys.meta_path:
                sys.meta_path.remove(tracker)
                
    except Exception as e:
        if verbose:
            console.print(f"[red]Error rebuilding cache: {e}[/red]")
        return False


def cache_entrypoint_metadata(fn: Callable, entrypoint: Any) -> None:
    """Cache metadata for an entrypoint function, including its place in the app structure."""
    cache_data = load_cache()
    full_namespace = f"{fn.__module__}.{fn.__name__}"
    signature = inspect.signature(fn)
    
    # Get the namespace directly from the entrypoint
    cli_namespace = getattr(entrypoint, "namespace", None)
    
    # Handle parameters, ensuring they're serializable
    parameters = []
    for param_name, param in signature.parameters.items():
        param_type = "Any"
        
        if param.annotation != inspect.Parameter.empty:
            try:
                # Get the full type namespace for this parameter
                underlying_types = get_underlying_types(param.annotation)
                if underlying_types:
                    # Get the namespace for the first type (main type)
                    typ = list(underlying_types)[0]
                    type_namespace = get_type_namespace(typ)
                    if type_namespace:
                        param_type = type_namespace
            except Exception:
                # If we can't process the type, use "Any"
                param_type = "Any"
        
        # Handle default value
        default_value = None
        if param.default != inspect.Parameter.empty:
            try:
                # Test if the default is JSON serializable
                json.dumps(param.default)
                default_value = param.default
            except (TypeError, OverflowError):
                # If not serializable, convert to string
                default_value = str(param.default)
        
        parameters.append({
            "name": param_name,
            "type": param_type,  # Full type namespace for factory matching
            "default": default_value,
        })

    # Determine the CLI path based on multiple sources of information
    cli_path = []
    module_parts = fn.__module__.split(".")
    
    # First, check if this module belongs to any registered entry points
    cli_structure = cache_data["app"].get("cli_structure", {})
    entry_point_match = None
    
    for ep_name, ep_data in cli_structure.items():
        ep_module = ep_data.get("module", "")
        # Check if this function is from a module that matches an entry point
        if fn.__module__ == ep_module or fn.__module__.startswith(f"{ep_module}."):
            entry_point_match = ep_name
            break
    
    if entry_point_match:
        # This function is part of a registered CLI entry point
        if cli_namespace:
            # If namespace is specified, use it as a subcommand
            # e.g., entrypoint(namespace="llm") in nemo.collections should be nemo llm
            cli_path = [entry_point_match]
            if cli_namespace != entry_point_match:
                # Only add if not redundant
                cli_path.extend(cli_namespace.split("."))
        else:
            # Otherwise use the entry point name and module path
            cli_path = [entry_point_match]
            
            # Add submodule parts after the entry point module
            if fn.__module__ != ep_module:
                # Get the parts after the entry point module
                ep_parts = ep_module.split(".")
                submodule_parts = module_parts[len(ep_parts):-1]  # Exclude the last part (file)
                # Add significant submodule parts (skip common non-descriptive parts)
                for part in submodule_parts:
                    if part not in ("api", "cli", "core", "entrypoints"):
                        cli_path.append(part)
    else:
        # Not part of a registered entry point, fall back to module-based path
        if cli_namespace:
            # Use explicit namespace
            cli_path = cli_namespace.split(".")
        else:
            # Derive from module, skipping common prefixes
            if module_parts[0] == "nemo_run":
                start_idx = 1
                for i, part in enumerate(module_parts[1:], 1):
                    if part not in ("cli", "api", "core"):
                        start_idx = i
                        break
                cli_path = module_parts[start_idx:-1]
            else:
                # For other modules, use significant parts
                for part in module_parts[:-1]:
                    if part not in ("api", "cli", "core", "entrypoints"):
                        cli_path.append(part)
    
    # Navigate and create the group structure
    current = cache_data["app"]["groups"]
    for i, part in enumerate(cli_path):
        if part not in current:
            # Create the group with appropriate metadata
            current[part] = {
                "name": part,
                "help": f"[Module] {part}",
                "commands": {},
                "module_path": ".".join(module_parts[:len(module_parts)-len(cli_path)+i+1]) if i < len(cli_path)-1 else fn.__module__,
                "is_entrypoint": i == 0 and part in cli_structure,
            }
        if "commands" not in current[part]:
            current[part]["commands"] = {}
        current = current[part]["commands"]

    # Store command metadata under the function name
    current[fn.__name__] = {
        "name": fn.__name__,
        "help": entrypoint.help_str or inspect.getdoc(fn) or "",
        "signature": parameters,
        "full_namespace": full_namespace,
        "cli_namespace": cli_namespace,
        "cli_path": cli_path,  # Store the full CLI path for reference
    }

    # Update metadata with module hash and ensure CLI structure is cached
    if not cache_data["metadata"].get("cli_structure_cached"):
        cache_cli_structure()
    
    cache_data["metadata"] = update_cache_metadata([fn.__module__], [])
    save_cache(cache_data)

def cache_factory_metadata(fn: Callable, registration: Any) -> None:
    """Cache metadata for a factory function under its registration namespace."""
    cache_data = load_cache()
    namespace = registration.namespace  # String like "pl.LightningModule" or "nemo_run.cli.entrypoints.llm.finetune.model"
    
    # Get the source line number if possible
    line_no = None
    try:
        line_no = str(inspect.getsourcelines(fn)[1])
    except (TypeError, OSError, IOError):
        # If we can't get the source line, just use an empty string
        line_no = ""
    
    if namespace not in cache_data["data"]["factories"]:
        cache_data["data"]["factories"][namespace] = []
    
    cache_data["data"]["factories"][namespace].append({
        "name": registration.name,
        "fn": f"{fn.__module__}.{fn.__name__}",
        "docstring": inspect.getdoc(fn),
        "module": fn.__module__,
        "line_no": line_no,  # Add the actual line number at cache creation time
    })
    
    # Update metadata with module hash
    cache_data["metadata"] = update_cache_metadata([], [fn.__module__])
    save_cache(cache_data)


def _print_sync_summary(console):
    """Print a detailed summary of the sync results with entrypoint breakdown."""
    from rich.panel import Panel
    from rich.table import Table
    from rich import box
    
    cache_data = load_cache()
    
    # Create the main summary table
    summary_table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
    summary_table.add_column("Component", style="blue")
    summary_table.add_column("Details", style="green")
    summary_table.add_column("Count", style="yellow", justify="right")
    
    # Process CLI entrypoints by group
    groups = cache_data["app"]["groups"]
    factories = cache_data["data"]["factories"]
    
    # Track totals for the final summary
    total_commands = 0
    total_factories = 0
    
    # Create the entrypoints breakdown table
    entrypoint_table = Table(box=None, show_header=True, header_style="bold")
    entrypoint_table.add_column("Entrypoint", style="cyan")
    entrypoint_table.add_column("Commands", justify="right")
    entrypoint_table.add_column("Factories", justify="right")
    
    # Process each CLI group
    for group_name, group_data in sorted(groups.items()):
        command_count = len(group_data.get("commands", {}))
        total_commands += command_count
        
        # Count factories associated with this entrypoint
        group_factories = 0
        
        # Check for factories in this group's namespace
        for namespace, factory_list in factories.items():
            # Match factory namespaces that start with the group's module_path
            module_path = group_data.get("module_path", "")
            if module_path and namespace.startswith(module_path):
                group_factories += len(factory_list)
        
        total_factories += group_factories
        
        # Add to the entrypoint breakdown table
        entrypoint_table.add_row(
            group_name,
            str(command_count),
            str(group_factories)
        )
    
    # Add the entrypoint breakdown to the main summary
    summary_table.add_row(
        "CLI Entrypoints",
        f"{len(groups)} groups, {total_commands} commands",
        str(len(groups))
    )
    
    # Add the factories information
    factory_namespaces = len(factories)
    summary_table.add_row(
        "Factory Functions",
        f"Across {factory_namespaces} namespaces",
        str(total_factories)
    )
    
    # Add workspace information if available
    if "workspace_file" in cache_data["metadata"] and cache_data["metadata"]["workspace_file"]:
        workspace_file = cache_data["metadata"]["workspace_file"]
        workspace_name = os.path.basename(workspace_file)
        
        # Count imports tracked from workspace if available
        workspace_modules = 0
        if "module_dependencies" in cache_data["metadata"]:
            # Estimate modules imported by workspace by looking for modules
            # whose path is not in site-packages
            for module_name, module_info in cache_data["metadata"]["module_dependencies"].items():
                if not module_info.get("file_path", "").startswith(SITE_PACKAGES):
                    workspace_modules += 1
        
        summary_table.add_row(
            "Workspace",
            f"{workspace_name} ({workspace_modules} modules)",
            "1"
        )
    
    # Add editable packages information
    if "editable_packages_metadata" in cache_data["metadata"]:
        editable_count = len(cache_data["metadata"]["editable_packages_metadata"])
        if editable_count > 0:
            editable_packages = list(cache_data["metadata"]["editable_packages_metadata"].values())
            packages = [meta.get("package_name", "unknown") for meta in editable_packages[:3]]
            package_str = ", ".join(packages)
            
            if editable_count > 3:
                package_str += f" and {editable_count - 3} more"
                
            summary_table.add_row(
                "Editable Packages",
                package_str,
                str(editable_count)
            )
    
    # Add metadata version information
    if "library_versions" in cache_data["metadata"]:
        lib_versions = []
        for lib, version in cache_data["metadata"]["library_versions"].items():
            lib_versions.append(f"{lib} {version}")
        
        summary_table.add_row(
            "Libraries",
            ", ".join(lib_versions),
            str(len(lib_versions))
        )
    
    # Create final panel with all information
    panel = Panel(
        summary_table,
        title="[bold cyan]NeMo Run Workspace Sync Summary[/bold cyan]",
        subtitle=f"[italic]Python {cache_data['metadata'].get('python_version', 'unknown')}[/italic]",
        expand=False,
        border_style="blue"
    )
    
    # Add the entrypoint breakdown as a second panel
    entrypoint_panel = Panel(
        entrypoint_table,
        title="[bold cyan]CLI Entrypoint Breakdown[/bold cyan]",
        expand=False,
        border_style="blue"
    )
    
    # Print the panels
    console.print(panel)
    console.print(entrypoint_panel)


def update_cache_metadata(entrypoint_modules: list[str], factory_modules: list[str]) -> dict[str, Any]:
    """
    Update the environment metadata for the cache.
    
    Args:
        entrypoint_modules: List of entrypoint module names
        factory_modules: List of factory module names
        
    Returns:
        Updated metadata dictionary
    """
    # Get module metadata
    module_metadata = {}
    for module_name in set(entrypoint_modules + factory_modules):
        module = sys.modules.get(module_name)
        if not module or not hasattr(module, "__file__") or not module.__file__:
            continue
            
        file_path = module.__file__
        if not os.path.exists(file_path):
            continue
            
        try:
            stats = os.stat(file_path)
            module_metadata[module_name] = {
                "file_path": file_path,
                "mtime": stats.st_mtime,
                "size": stats.st_size
            }
        except Exception:
            continue
    
    # Get workspace file metadata
    workspace_file_path = _search_workspace_file()
    workspace_metadata = None
    if workspace_file_path and os.path.exists(workspace_file_path):
        try:
            stats = os.stat(workspace_file_path)
            workspace_metadata = {
                "mtime": stats.st_mtime,
                "size": stats.st_size
            }
        except Exception:
            pass
    
    # Get editable packages metadata
    editable_metadata = {}
    for file in os.listdir(SITE_PACKAGES):
        if file.endswith(".egg-link"):
            try:
                with open(os.path.join(SITE_PACKAGES, file), "r") as f:
                    content = f.read()
                    path = content.strip().split('\n')[0]
                    if os.path.exists(path):
                        package_name = file[:-9] if file.endswith('.egg-link') else file
                        
                        # Get basic metadata without computing full hashes
                        editable_metadata[path] = {
                            "package_name": package_name,
                            "mtime": os.path.getmtime(path)
                        }
            except Exception:
                pass
    
    # Return updated metadata
    return {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "library_versions": {
            "nemo_run": metadata.version("nemo_run"),
            "typer": metadata.version("typer"),
        },
        "editable_packages_metadata": editable_metadata,
        "module_metadata": module_metadata,
        "workspace_file": workspace_file_path,
        "workspace_metadata": workspace_metadata,
    }


def cache_cli_structure() -> None:
    """Cache the CLI structure based on entry points."""
    cache_data = load_cache()
    
    try:
        entrypoints = metadata.entry_points().select(group="nemo_run.cli")
        cli_structure = {}
        
        for ep in entrypoints:
            # Store mapping from CLI command name to module
            cli_structure[ep.name] = {
                "module": ep.value,
                "name": ep.name,
            }
            
            # Create corresponding group structure in the app
            if ep.name not in cache_data["app"]["groups"]:
                cache_data["app"]["groups"][ep.name] = {
                    "name": ep.name,
                    "help": f"Commands from {ep.value}",
                    "commands": {},
                    "module_path": ep.value,
                    "is_entrypoint": True
                }
        
        # Store the CLI structure in the cache
        cache_data["app"]["cli_structure"] = cli_structure
        
        # Update metadata
        cache_data["metadata"]["cli_structure_cached"] = True
        save_cache(cache_data)
        
    except Exception as e:
        print(f"Warning: Failed to cache CLI structure: {e}")


def cache_executor_classes() -> None:
    """Cache metadata about executor classes to avoid hardcoding namespaces."""
    try:
        from nemo_run.cli.api import EXECUTOR_CLASSES
        
        cache_data = load_cache()
        
        # Extract namespaces for all executor classes
        executor_namespaces = []
        for cls in EXECUTOR_CLASSES:
            namespace = f"{cls.__module__}.{cls.__name__}"
            executor_namespaces.append(namespace)
        
        # Store in the cache
        cache_data["metadata"]["executor_namespaces"] = executor_namespaces
        save_cache(cache_data)
    except Exception as e:
        print(f"Warning: Failed to cache executor classes: {e}")


def get_editable_packages_metadata() -> dict[str, dict[str, Any]]:
    """Identify editable packages and compute their directory metadata."""
    editable_metadata = {}
    
    # Look for .egg-link files which indicate editable installs
    for file in os.listdir(SITE_PACKAGES):
        if file.endswith(".egg-link"):
            try:
                with open(os.path.join(SITE_PACKAGES, file), "r") as f:
                    content = f.read()
                    path = content.strip().split('\n')[0]
                    if os.path.exists(path):
                        # Get package name from the filename (remove .egg-link)
                        package_name = file[:-9] if file.endswith('.egg-link') else file
                        
                        # Compute basic metadata 
                        editable_metadata[path] = {
                            "package_name": package_name,
                            "mtime": os.path.getmtime(path)
                        }
            except Exception as e:
                print(f"Warning: Failed to process editable package {file}: {e}")
    
    return editable_metadata


@functools.cache
def load_entrypoints():
    """Load entrypoints from entry_points."""
    tracker = None
    try:
        # Create and use a tracker when we need to populate the cache
        if POPULATE_CACHE:
            tracker = ImportTracker()
            sys.meta_path.insert(0, tracker)
            
        entrypoints = metadata.entry_points().select(group="nemo_run.cli")
        for ep in entrypoints:
            try:
                ep.load()
            except Exception as e:
                print(f"Couldn't load entrypoint {ep.name}: {e}")
                
        # If we're tracking imports for cache population, update the cache
        if POPULATE_CACHE and tracker:
            tracker.update_cache()
                
    finally:
        # Make sure to clean up
        if tracker and tracker in sys.meta_path:
            sys.meta_path.remove(tracker)


@functools.cache
def load_workspace():
    """Load workspace file."""
    workspace_file_path = _search_workspace_file()

    if workspace_file_path:
        return _load_workspace_file(workspace_file_path)


def _search_workspace_file() -> str | None:
    """Search for workspace.py file in the current directory or parent directories."""
    if not INCLUDE_WORKSPACE_FILE:
        return None

    current_dir = os.getcwd()
    file_names = [
        "workspace_private.py",
        "workspace.py",
        os.path.join(NEMORUN_HOME, "workspace.py"),
    ]

    while True:
        for file_name in file_names:
            workspace_file_path = os.path.join(current_dir, file_name)
            if os.path.exists(workspace_file_path):
                return workspace_file_path

        # Go up one directory level
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:  # Root directory
            break
        current_dir = parent_dir

    return None


def _load_workspace_file(path):
    """Load workspace file."""
    tracker = None
    try:
        # Create and use a tracker when we need to populate the cache
        if POPULATE_CACHE:
            tracker = ImportTracker()
            sys.meta_path.insert(0, tracker)
            
        spec = importlib.util.spec_from_file_location("workspace", path)
        assert spec
        workspace_module = importlib.util.module_from_spec(spec)
        assert spec.loader
        spec.loader.exec_module(workspace_module)
        
        # If we're tracking imports for cache population, update the cache
        if POPULATE_CACHE and tracker:
            tracker.update_cache()
                
    finally:
        # Make sure to clean up
        if tracker and tracker in sys.meta_path:
            sys.meta_path.remove(tracker)


def delete_cache() -> bool:
    """
    Delete the CLI cache file if it exists.
    
    Returns:
        bool: True if cache was deleted or didn't exist, False if there was an error
    """
    from nemo_run.cli.cache import CACHE_FILE
    from nemo_run.core.frontend.console.api import CONSOLE
    
    if os.path.exists(CACHE_FILE):
        try:
            os.remove(CACHE_FILE)
            CONSOLE.print(f"Cache file removed: {CACHE_FILE}")
            return True
        except Exception as e:
            CONSOLE.print(f"Error removing cache file: {e}")
            return False
    return True  # Consider it a success if file doesn't exist


if __name__ == "__main__":
    sync(no_cache=True)