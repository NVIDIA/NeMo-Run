# cache.py
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import json
import hashlib
import site
import sys
import inspect
import multiprocessing
import tempfile
import importlib.util
import functools
import re
import asyncio
import aiofiles
from typing import Dict, Any, Callable, Optional, List, Tuple, Coroutine
from importlib import metadata

from typer import Typer
from typer.core import TyperCommand
from rich.console import Console
from rich.text import Text
from rich import box
from typer import rich_utils
from nemo_run.core.frontend.console.styles import BOX_STYLE, TABLE_STYLES
from nemo_run.config import get_type_namespace, get_underlying_types

# Constants
SITE_PACKAGES = site.getsitepackages()[0]
CACHE_DIR = os.path.join(SITE_PACKAGES, "nemo_run", "cache")
CACHE_FILE = os.path.join(CACHE_DIR, "cli_cache.json")
CONSOLE = Console()

# File hash cache to avoid recomputing hashes for the same file
_file_hash_cache = {}

class ImportTracker:
    """Tracks imported modules during execution and efficiently validates their freshness.
    
    Differentiates between site-packages modules (version check) and editable modules (metadata check).
    """

    def __init__(self):
        self.imported_modules = []
        self.module_info = {}  # Will store path and metadata info for each module
        
        # Get system temp directory paths
        self.temp_dirs = [
            os.path.realpath(tempfile.gettempdir()),  # Standard temp directory
            '/tmp',                                    # Common Unix temp
            '/var/tmp',                               # Another common Unix temp
            '/var/folders'                            # macOS temp location
        ]
        
        # Get stdlib paths for filtering out standard library modules
        self.stdlib_paths = self._get_stdlib_paths()

    def _get_stdlib_paths(self) -> List[str]:
        """Get paths to the standard library modules to filter them out."""
        stdlib_paths = []
        
        # Standard library path
        for path in sys.path:
            if path and os.path.exists(path):
                # Common patterns for standard library locations
                if any(pattern in path for pattern in [
                    "lib/python", 
                    "lib64/python",
                    "lib/Python",
                    "Python.framework"
                ]):
                    stdlib_paths.append(os.path.realpath(path))
                    
        # Also add the specific Python lib paths
        lib_paths = [p for p in sys.path if os.path.basename(p) == 'lib-dynload']
        stdlib_paths.extend([os.path.realpath(p) for p in lib_paths])
        
        return stdlib_paths

    def find_spec(self, fullname: str, path: Optional[str], target: Optional[Any] = None):
        """Record imported module and capture its location information."""
        self.imported_modules.append(fullname)
        
        # Immediately capture module info if it's already loaded
        if fullname in sys.modules:
            self._capture_module_info(fullname)
        
        return None

    def get_imported_modules(self) -> List[str]:
        """Return list of imported module names."""
        return self.imported_modules
    
    def _is_in_temp_dir(self, file_path: str) -> bool:
        """Check if a file is in a temporary directory."""
        if not file_path:
            return False
            
        real_path = os.path.realpath(file_path)
        return any(real_path.startswith(temp_dir) for temp_dir in self.temp_dirs)
    
    def _is_stdlib_module(self, file_path: str) -> bool:
        """Check if a module is part of the Python standard library."""
        if not file_path:
            return False
            
        real_path = os.path.realpath(file_path)
        
        # Check if the file is in a standard library path
        return any(real_path.startswith(stdlib_path) for stdlib_path in self.stdlib_paths)
    
    def _is_valid_python_module(self, file_path: str) -> bool:
        """Check if the path points to a valid Python module file."""
        if not file_path:
            return False
            
        # Check file extension for Python files
        if not (file_path.endswith('.py') or 
                file_path.endswith('.so') or 
                file_path.endswith('.pyd')):
            return False
            
        # Exclude common Python cache files
        if '__pycache__' in file_path or file_path.endswith('.pyc'):
            return False
            
        # Ensure it's a file that exists
        return os.path.isfile(file_path)
    
    def _is_transient_module(self, module_name: str, file_path: str) -> bool:
        """Check if a module appears to be transient or temporary."""
        if not module_name or not file_path:
            return True
            
        # Patterns that suggest a transient/non-package module
        patterns = [
            r'_remote_module_non_scriptab',  # Remote module patterns
            r'__temp__',                    # Temporary module pattern
            r'_anonymous_',                 # Anonymous module pattern
            r'dynamically_generated_',      # Dynamically generated modules
            r'ipykernel_launcher',         # IPython kernel modules
            r'xdist',                       # Pytest xdist temporary modules
        ]
        
        # Check module name patterns
        if any(re.search(pattern, module_name) for pattern in patterns):
            return True
            
        # Check file path patterns
        if any(re.search(pattern, file_path) for pattern in patterns):
            return True
            
        return False
    
    def _capture_module_info(self, module_name: str):
        """Capture metadata about a module for cache validation."""
        if module_name in self.module_info:
            return  # Already captured
            
        # Skip built-in modules
        if module_name in sys.builtin_module_names:
            return
            
        # Skip common Python standard library modules
        if module_name in ('sys', 'os', 're', 'time', 'json', 'math', 'random', 'datetime', 'collections'):
            return
            
        module = sys.modules.get(module_name)
        if not module:
            return  # Module not actually loaded
            
        # Skip modules without a file (built-ins, C extensions, etc.)
        if not hasattr(module, "__file__") or not module.__file__:
            return
            
        file_path = module.__file__
        
        # Skip modules in temp directories
        if self._is_in_temp_dir(file_path):
            return
            
        # Skip standard library modules
        if self._is_stdlib_module(file_path):
            return
            
        # Skip non-Python modules or invalid files
        if not self._is_valid_python_module(file_path):
            return
            
        # Skip modules that appear to be transient
        if self._is_transient_module(module_name, file_path):
            return
        
        # Determine if this is a site-packages module or editable mode
        is_site_package = file_path.startswith(SITE_PACKAGES)
        
        # For site packages, we'll track the version
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
            except (metadata.PackageNotFoundError, Exception):
                # Fallback to file metadata if we can't get version
                self._record_file_metadata(module_name, file_path)
        else:
            # For editable packages, track file metadata instead of full hash
            self._record_file_metadata(module_name, file_path)
    
    def _record_file_metadata(self, module_name: str, file_path: str):
        """Record file metadata (mtime) instead of full content hash."""
        try:
            # Get file modification time (much faster than hashing contents)
            mtime = os.path.getmtime(file_path)
            
            self.module_info[module_name] = {
                "type": "editable", 
                "file_path": file_path,
                "mtime": mtime
            }
        except (OSError, Exception) as e:
            print(f"Warning: Couldn't get metadata for {file_path}: {e}")
    
    def _get_package_name(self, module_name: str) -> Optional[str]:
        """Try to determine the package name from the module name."""
        # First try the base name
        if self._is_valid_package(module_name):
            return module_name
            
        # Try parent packages, starting from the top level
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
        # First collect info for any modules that were imported but not yet processed
        for module_name in self.imported_modules:
            if module_name not in self.module_info and module_name in sys.modules:
                self._capture_module_info(module_name)
        
        # Now update the cache with the collected info
        cache_data = load_cache()
        
        # Initialize the module dependencies section if it doesn't exist
        if "module_dependencies" not in cache_data["metadata"]:
            cache_data["metadata"]["module_dependencies"] = {}
            
        # Update with our module info
        cache_data["metadata"]["module_dependencies"].update(self.module_info)
        
        # Save the updated cache
        save_cache(cache_data)

    async def _capture_module_info_async(self, module_names: List[str]):
        """Capture metadata about multiple modules concurrently."""
        tasks = []
        for module_name in module_names:
            if module_name in self.module_info:
                continue  # Already captured
                
            # Skip built-in modules
            if module_name in sys.builtin_module_names:
                continue
                
            # Skip common Python standard library modules
            if module_name in ('sys', 'os', 're', 'time', 'json', 'math', 'random', 'datetime', 'collections'):
                continue
                
            module = sys.modules.get(module_name)
            if not module:
                continue  # Module not actually loaded
                
            # Skip modules without a file (built-ins, C extensions, etc.)
            if not hasattr(module, "__file__") or not module.__file__:
                continue
                
            file_path = module.__file__
            
            # Skip modules in temp directories
            if self._is_in_temp_dir(file_path):
                continue
                
            # Skip standard library modules
            if self._is_stdlib_module(file_path):
                continue
                
            # Skip non-Python modules or invalid files
            if not self._is_valid_python_module(file_path):
                continue
                
            # Skip modules that appear to be transient
            if self._is_transient_module(module_name, file_path):
                continue
            
            # Determine if this is a site-packages module or editable mode
            is_site_package = file_path.startswith(SITE_PACKAGES)
            
            # Create task for processing this module
            tasks.append(self._process_module_async(module_name, file_path, is_site_package))
        
        # Run all tasks concurrently
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _process_module_async(self, module_name: str, file_path: str, is_site_package: bool):
        """Process a single module asynchronously."""
        try:
            if is_site_package:
                # For site packages, we'll track the version
                package_name = self._get_package_name(module_name)
                version = metadata.version(package_name) if package_name else None
                
                self.module_info[module_name] = {
                    "type": "site-package",
                    "file_path": file_path,
                    "package_name": package_name,
                    "version": version
                }
            else:
                # For editable packages, track file metadata instead of full hash
                await self._record_file_metadata_async(module_name, file_path)
        except Exception as e:
            print(f"Warning: Error processing module {module_name}: {e}")
    
    async def _record_file_metadata_async(self, module_name: str, file_path: str):
        """Record file metadata (mtime) asynchronously."""
        try:
            # Get file modification time asynchronously
            # Note: os.path.getmtime is typically fast enough to not need async
            mtime = os.path.getmtime(file_path)
            
            self.module_info[module_name] = {
                "type": "editable", 
                "file_path": file_path,
                "mtime": mtime
            }
        except (OSError, Exception) as e:
            print(f"Warning: Couldn't get metadata for {file_path}: {e}")
    
    async def update_cache_async(self):
        """Update the cache with module dependency information asynchronously."""
        # First collect info for any modules that were imported but not yet processed
        modules_to_process = [
            module_name for module_name in self.imported_modules 
            if module_name not in self.module_info and module_name in sys.modules
        ]
        
        # Process all modules concurrently
        await self._capture_module_info_async(modules_to_process)
        
        # Now update the cache with the collected info
        cache_data = load_cache()
        
        # Initialize the module dependencies section if it doesn't exist
        if "module_dependencies" not in cache_data["metadata"]:
            cache_data["metadata"]["module_dependencies"] = {}
            
        # Update with our module info
        cache_data["metadata"]["module_dependencies"].update(self.module_info)
        
        # Save the updated cache
        save_cache(cache_data)
    
    def update_cache(self):
        """Synchronous wrapper for update_cache_async."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.update_cache_async())
        finally:
            loop.close()

async def _get_filehash_async(filepath, hasher_factory, chunk_size, cache=None):
    """Compute the hash of the given filepath asynchronously.

    Args:
        filepath: Path to the file to hash.
        hasher_factory: Callable that returns a hashlib hash object.
        chunk_size: The number of bytes to read in one go from files.
        cache: Optional dictionary to cache results by filepath.

    Returns:
        The hash/checksum as a string of hexadecimal digits.
    """
    if cache is not None:
        filehash = cache.get(filepath, None)
        if filehash is not None:
            return filehash

    hasher = hasher_factory()
    async with aiofiles.open(filepath, "rb") as f:
        while chunk := await f.read(chunk_size):
            hasher.update(chunk)

    filehash = hasher.hexdigest()
    
    if cache is not None:
        cache[filepath] = filehash
        
    return filehash

# Synchronous wrapper for backward compatibility
def _get_filehash(filepath, hasher_factory, chunk_size, cache=None):
    """Compute the hash of the given filepath (synchronous version).

    Args:
        filepath: Path to the file to hash.
        hasher_factory: Callable that returns a hashlib hash object.
        chunk_size: The number of bytes to read in one go from files.
        cache: Optional dictionary to cache results by filepath.

    Returns:
        The hash/checksum as a string of hexadecimal digits.
    """
    if cache is not None:
        filehash = cache.get(filepath, None)
        if filehash is not None:
            return filehash

    hasher = hasher_factory()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hasher.update(chunk)

    filehash = hasher.hexdigest()
    
    if cache is not None:
        cache[filepath] = filehash
        
    return filehash

async def compute_directory_metadata_async(path: str) -> Dict[str, Any]:
    """Compute metadata for a directory asynchronously using file modification times and sizes."""
    try:
        file_count = 0
        total_size = 0
        latest_mtime = 0
        path_hasher = hashlib.sha256()
        
        # Create a task to walk the directory
        async def walk_directory():
            nonlocal file_count, total_size, latest_mtime
            for root, _, files in os.walk(path):
                for file in sorted(files):  # Sort for deterministic ordering
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, path)
                    
                    # Skip common cache and temporary files
                    if (rel_path.startswith('.git') or 
                        rel_path.startswith('__pycache__') or 
                        rel_path.endswith('.pyc')):
                        continue
                    
                    try:
                        # Get file stats
                        stats = os.stat(file_path)
                        file_count += 1
                        total_size += stats.st_size
                        latest_mtime = max(latest_mtime, stats.st_mtime)
                        
                        # Add path to the path hash
                        path_hasher.update(rel_path.encode())
                    except (OSError, IOError):
                        # Skip files we can't access
                        continue
        
        await walk_directory()
        
        return {
            "file_count": file_count,
            "total_size": total_size,
            "latest_mtime": latest_mtime,
            "path_hash": path_hasher.hexdigest()
        }
    except Exception as e:
        print(f"Warning: Error computing directory metadata for {path}: {e}")
        return {
            "error": str(e),
            "path": path
        }

# Synchronous wrapper for backward compatibility
def compute_directory_metadata(path: str) -> Dict[str, Any]:
    """Synchronous wrapper for compute_directory_metadata_async."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(compute_directory_metadata_async(path))
    finally:
        loop.close()

async def get_editable_packages_metadata_async() -> Dict[str, Dict[str, Any]]:
    """Identify editable packages and compute their directory metadata asynchronously."""
    editable_metadata = {}
    
    # Look for .egg-link files which indicate editable installs
    metadata_tasks = []
    
    for file in os.listdir(SITE_PACKAGES):
        if file.endswith(".egg-link"):
            try:
                async with aiofiles.open(os.path.join(SITE_PACKAGES, file), "r") as f:
                    content = await f.read()
                    path = content.strip().split('\n')[0]
                    if os.path.exists(path):
                        # Get package name from the filename (remove .egg-link)
                        package_name = file[:-9] if file.endswith('.egg-link') else file
                        
                        # Create task to compute metadata asynchronously
                        metadata_tasks.append((path, package_name, compute_directory_metadata_async(path)))
            except Exception as e:
                print(f"Warning: Failed to process editable package {file}: {e}")
    
    # Wait for all metadata computation tasks to complete
    for path, package_name, task_coro in metadata_tasks:
        metadata = await task_coro
        metadata["package_name"] = package_name
        editable_metadata[path] = metadata
    
    return editable_metadata

# Synchronous wrapper for backward compatibility
def get_editable_packages_metadata() -> Dict[str, Dict[str, Any]]:
    """Synchronous wrapper for get_editable_packages_metadata_async."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(get_editable_packages_metadata_async())
    finally:
        loop.close()

def get_module_metadata(module_name: str) -> Dict[str, Any]:
    """Get metadata for a module file for cache validation."""
    module = sys.modules.get(module_name)
    if not module or not hasattr(module, "__file__") or not module.__file__:
        return {}  # Can't get metadata
        
    file_path = module.__file__
    if not os.path.exists(file_path):
        return {}
        
    try:
        stats = os.stat(file_path)
        return {
            "file_path": file_path,
            "mtime": stats.st_mtime,
            "size": stats.st_size
        }
    except (OSError, IOError) as e:
        print(f"Warning: Failed to get metadata for {file_path}: {e}")
        return {}

def load_cache() -> Dict[str, Any]:
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
            "cli_structure": {}  # New field to store CLI structure information
        },
        "data": {"common_options": {}, "executors": [], "factories": {}}
    }

def save_cache(cache_data: Dict[str, Any]) -> None:
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

async def validate_cache_async(cache_data: Dict[str, Any]) -> bool:
    """Asynchronous version of validate_cache that performs checks concurrently."""
    # Check if cache is empty or has no metadata (likely first run)
    if not cache_data or "metadata" not in cache_data:
        return False
        
    metadata_info = cache_data.get("metadata", {})
    
    # Python version check - this is fast, so keep it synchronous
    current_python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if metadata_info.get("python_version") != current_python_version:
        print(f"Cache invalidated: Python version changed from {metadata_info.get('python_version')} to {current_python_version}")
        return False

    # Create a list of validation coroutines to run concurrently
    validation_tasks = [
        _validate_libraries_async(metadata_info),
        _validate_editable_packages_async(metadata_info),
        _validate_module_dependencies_async(metadata_info),
        _validate_workspace_file_async(metadata_info)
    ]
    
    # Run all validation checks concurrently
    results = await asyncio.gather(*validation_tasks, return_exceptions=True)
    
    # Check if any validation failed (returned False or raised exception)
    for result in results:
        if isinstance(result, Exception):
            print(f"Cache validation error: {result}")
            return False
        if result is False:
            return False
    
    return True

async def _validate_libraries_async(metadata_info: Dict[str, Any]) -> bool:
    """Validate library versions asynchronously."""
    try:
        required_libs = {"nemo_run": metadata.version("nemo_run"), "typer": metadata.version("typer")}
        for lib, ver in metadata_info.get("library_versions", {}).items():
            try:
                current_ver = required_libs.get(lib)
                if current_ver != ver:
                    print(f"Cache invalidated: Library {lib} version changed from {ver} to {current_ver}")
                    return False
            except metadata.PackageNotFoundError:
                print(f"Cache invalidated: Library {lib} not found")
                return False
        return True
    except Exception as e:
        print(f"Cache validation error checking library versions: {e}")
        return False

async def _validate_editable_packages_async(metadata_info: Dict[str, Any]) -> bool:
    """Validate editable packages asynchronously."""
    try:
        # Get editable packages metadata concurrently
        current_editable_metadata = await get_editable_packages_metadata_async()
        stored_editable_metadata = metadata_info.get("editable_packages_metadata", {})
        
        # Only verify paths that exist in both sets
        common_paths = set(current_editable_metadata.keys()) & set(stored_editable_metadata.keys())
        
        # Create async tasks for each package validation
        validation_tasks = []
        for path in common_paths:
            current_meta = current_editable_metadata[path]
            stored_meta = stored_editable_metadata[path]
            
            # Check if file count, size, or path structure changed
            if (current_meta.get("file_count") != stored_meta.get("file_count") or
                current_meta.get("path_hash") != stored_meta.get("path_hash")):
                print(f"Cache invalidated: Editable package at {path} structure changed")
                return False
                
            # Check if any files were modified
            if current_meta.get("latest_mtime", 0) > stored_meta.get("latest_mtime", 0):
                print(f"Cache invalidated: Files in editable package at {path} were modified")
                return False
        
        return True
    except Exception as e:
        print(f"Cache validation error checking editable packages: {e}")
        return False

async def _validate_module_dependencies_async(metadata_info: Dict[str, Any]) -> bool:
    """Validate module dependencies asynchronously."""
    try:
        dependencies = metadata_info.get("module_dependencies", {})
        
        # Create tasks for validating each module dependency
        validation_tasks = []
        for module_name, dep_info in dependencies.items():
            validation_tasks.append(_validate_single_module_dependency(module_name, dep_info))
        
        # Run all module validations concurrently
        results = await asyncio.gather(*validation_tasks, return_exceptions=True)
        
        # Check if any validation failed
        for result in results:
            if isinstance(result, Exception):
                print(f"Cache validation error: {result}")
                return False
            if result is False:
                return False
        
        return True
    except Exception as e:
        print(f"Cache validation error checking module dependencies: {e}")
        return False

async def _validate_single_module_dependency(module_name: str, dep_info: Dict[str, Any]) -> bool:
    """Validate a single module dependency."""
    dependency_type = dep_info.get("type")
    
    if dependency_type == "site-package":
        # For site packages, just check the version
        package_name = dep_info.get("package_name")
        stored_version = dep_info.get("version")
        
        if package_name and stored_version:
            try:
                current_version = metadata.version(package_name)
                if current_version != stored_version:
                    print(f"Cache invalidated: Package {package_name} version changed from {stored_version} to {current_version}")
                    return False
            except metadata.PackageNotFoundError:
                print(f"Cache invalidated: Package {package_name} is no longer installed")
                return False
        else:
            # Fall back to file metadata check if no package version
            file_path = dep_info.get("file_path")
            if not file_path or not os.path.exists(file_path):
                print(f"Cache invalidated: Module file for {module_name} no longer exists")
                return False
            
            # If we have an mtime, check it
            if "mtime" in dep_info:
                current_mtime = os.path.getmtime(file_path)
                if current_mtime > dep_info["mtime"]:
                    print(f"Cache invalidated: Module file for {module_name} has changed (modified)")
                    return False
    
    elif dependency_type == "editable":
        # For editable packages, check file metadata
        file_path = dep_info.get("file_path")
        stored_mtime = dep_info.get("mtime")
        
        if not file_path or not os.path.exists(file_path):
            print(f"Cache invalidated: Editable module file {file_path} no longer exists")
            return False
        
        if stored_mtime:
            current_mtime = os.path.getmtime(file_path)
            if current_mtime > stored_mtime:
                print(f"Cache invalidated: Editable module file {file_path} has been modified")
                return False
    
    return True

async def _validate_workspace_file_async(metadata_info: Dict[str, Any]) -> bool:
    """Validate workspace file asynchronously."""
    try:
        stored_workspace_file = metadata_info.get("workspace_file")
        stored_workspace_metadata = metadata_info.get("workspace_metadata")
        
        if stored_workspace_file and stored_workspace_metadata:
            current_workspace_file = _search_workspace_file()
            
            # Check if workspace file path changed
            if current_workspace_file != stored_workspace_file:
                print(f"Cache invalidated: Workspace file changed from {stored_workspace_file} to {current_workspace_file}")
                return False
                
            # Check if workspace file content changed using metadata
            if os.path.exists(stored_workspace_file):
                stats = os.stat(stored_workspace_file)
                if stats.st_mtime > stored_workspace_metadata.get("mtime", 0):
                    print(f"Cache invalidated: Workspace file content changed (modified)")
                    return False
                if stats.st_size != stored_workspace_metadata.get("size", 0):
                    print(f"Cache invalidated: Workspace file size changed")
                    return False
        
        return True
    except Exception as e:
        print(f"Cache validation error checking workspace file: {e}")
        return False

# Add a synchronous wrapper for backward compatibility
def validate_cache(cache_data: Dict[str, Any]) -> bool:
    """Synchronous wrapper for validate_cache_async."""
    try:
        # Create a new event loop to run the async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(validate_cache_async(cache_data))
        finally:
            loop.close()
    except Exception as e:
        print(f"Error validating cache: {e}")
        return False

def update_cache_metadata(entrypoint_modules: list[str], factory_modules: list[str]) -> Dict[str, Any]:
    """Update the environment metadata for the cache using file metadata instead of hashes."""
    cache_data = load_cache()  # Load current cache data
    metadata_info = cache_data.get("metadata", {})  # Get existing metadata or empty dict
    
    # Instead of computing hashes, just store module metadata
    module_metadata = {}
    for module_name in set(entrypoint_modules + factory_modules):
        module_metadata[module_name] = get_module_metadata(module_name)
    
    # Capture entry point mappings from the package system
    entrypoint_mappings = {}
    try:
        entrypoints = metadata.entry_points().select(group="nemo_run.cli")
        for ep in entrypoints:
            entrypoint_mappings[ep.name] = ep.value
    except Exception as e:
        print(f"Warning: Failed to collect entry point mappings: {e}")
    
    # Check for workspace file and add its metadata if it exists
    workspace_file_path = _search_workspace_file()
    workspace_metadata = None
    if workspace_file_path and os.path.exists(workspace_file_path):
        try:
            stats = os.stat(workspace_file_path)
            workspace_metadata = {
                "mtime": stats.st_mtime,
                "size": stats.st_size
            }
        except Exception as e:
            print(f"Warning: Failed to get workspace file metadata: {e}")
    
    # Ensure executor classes are cached if not already
    if "executor_namespaces" not in metadata_info:
        cache_executor_classes()
        # Re-load the cache to get the updated metadata
        fresh_cache = load_cache()
        executor_namespaces = fresh_cache["metadata"].get("executor_namespaces", [])
    else:
        executor_namespaces = metadata_info.get("executor_namespaces", [])
    
    return {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "library_versions": {
            "nemo_run": metadata.version("nemo_run"),
            "typer": metadata.version("typer"),
        },
        "editable_packages_metadata": get_editable_packages_metadata(),
        "module_metadata": module_metadata,
        "workspace_file": workspace_file_path,
        "workspace_metadata": workspace_metadata,
        "entrypoint_mappings": entrypoint_mappings,  # Store CLI entry point mappings
        "executor_namespaces": executor_namespaces,  # Include executor namespaces in metadata
    }

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

class CachedEntrypointCommand(TyperCommand):
    """Custom command class that uses cached data to render help output with Rich formatting."""

    def __init__(self, *args, cached_data: Dict[str, Any], **kwargs):
        super().__init__(*args, **kwargs)
        self.cached_data = cached_data  # Contains command, executors, and factories

    def format_help(self, ctx, formatter):
        """Formats and displays the help output using cached data and Rich rendering."""
        from nemo_run.help import (
            render_entrypoint_factories, 
            render_arguments, 
            render_parameter_factories, 
            render_executors, 
        )
        
        cmd_data = self.cached_data["command"]
        executors = self.cached_data["executors"]
        factories = self.cached_data["factories"]
        console = Console()

        # Usage and Description
        command_path = " ".join(ctx.command_path.split()[1:])  # e.g., "llm finetune"
        usage_text = Text(f"Usage: nemo {command_path} [OPTIONS] [ARGUMENTS]", style="bold")
        description_text = Text(f"[Entrypoint] {cmd_data['name']}\n{cmd_data['help']}", style="cyan")
        console.print(usage_text)
        console.print(description_text)
        console.print()  # Add spacing

        # Pre-loaded Entrypoint Factories
        entrypoint_factories = factories.get(cmd_data["full_namespace"], [])
        if entrypoint_factories:
            factories_panel = render_entrypoint_factories(
                entrypoint_factories,
                title="Pre-loaded entrypoint factories, run with --factory"
            )
            console.print(factories_panel)

        # Arguments
        arguments_panel = render_arguments(cmd_data["signature"])
        console.print(arguments_panel)

        # Factories for Each Parameter
        for param in cmd_data["signature"]:
            arg_namespace = f"{cmd_data['full_namespace']}.{param['name']}"
            param_factories = factories.get(arg_namespace, [])
            if not param_factories:
                param_factories = factories.get(param["type"], [])
            if param_factories:
                param_factories_panel = render_parameter_factories(param["name"], param["type"], param_factories)
                console.print(param_factories_panel)

        # Registered Executors
        executors_panel = render_executors(executors)
        console.print(executors_panel)

        # Typer Default Options
        super().format_help(ctx, formatter)  # Let Typer handle default options

def reconstruct_help_from_cache(cache_data: Dict[str, Any], argv: list[str]) -> Typer:
    """
    Renders help information directly from the cache without trying to fully reconstruct
    the Typer app structure, then returns a minimal Typer app.

    Args:
        cache_data (Dict[str, Any]): Cached metadata containing app structure, commands, and factories.
        argv (list[str]): Command-line arguments (e.g., ["nemo", "llm", "finetune", "--help"]).

    Returns:
        Typer: A minimal Typer app after displaying the help information.
    """
    try:
        # Extract the command/group path from arguments
        help_path = [arg for arg in argv[1:] if arg not in ("--help", "-h")]
        
        if not help_path:
            # If no specific path is requested, render the root help
            render_root_help(cache_data)
        else:
            # Find the requested group or command
            render_path_help(cache_data, help_path)
        
        # After rendering the help, exit the program successfully
        sys.exit(0)
        
        # This part won't run due to sys.exit() above, but is required for the function signature
        app = Typer(name="nemo", help="NeMo CLI")
        return app
    
    except Exception as e:
        print(f"Error displaying help from cache: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Fall back to a minimal app, letting normal CLI handle things
        app = Typer(name="nemo", help="NeMo CLI (fallback)")
        return app

def render_root_help(cache_data: Dict[str, Any]) -> None:
    """Render the root help information from the cache."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    
    console = Console()
    
    # Display the app name and help
    app_data = cache_data["app"]
    title = Text("NeMo Run CLI", style="bold cyan")
    description = Text(app_data.get("help", "Command-line interface for NeMo Run."))
    
    console.print(title)
    console.print(description)
    console.print()
    
    # Display available groups
    console.print(Text("Available command groups:", style="bold yellow"))
    
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Name", style="green")
    table.add_column("Description", style="white")
    
    for group_name, group_data in app_data.get("groups", {}).items():
        help_text = group_data.get("help", f"Commands in the {group_name} module")
        table.add_row(group_name, help_text)
    
    console.print(table)
    console.print()
    
    # Show usage examples
    console.print(Text("Usage examples:", style="bold yellow"))
    console.print("  nemo <command-group> --help     Show help for a command group")
    console.print("  nemo <command-group> <command>  Run a specific command")
    console.print()
    
    # Show common options
    common_options = cache_data.get("data", {}).get("common_options", {})
    if common_options:
        console.print(Text("Common options:", style="bold yellow"))
        options_table = Table(show_header=False, box=None, padding=(0, 2))
        options_table.add_column("Option", style="green")
        options_table.add_column("Description", style="white")
        
        for option_name, option_data in common_options.items():
            options_table.add_row(option_name, option_data.get("help", ""))
        
        console.print(options_table)
        console.print()

def render_path_help(cache_data: Dict[str, Any], path: list[str]) -> None:
    """
    Render help for a specific command or group path.
    
    Args:
        cache_data (Dict[str, Any]): The cache data
        path (list[str]): The command/group path, e.g. ["llm"] or ["llm", "finetune"]
    """
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from nemo_run.help import render_arguments, render_entrypoint_factories, render_parameter_factories, render_executors
    
    console = Console()
    
    # Navigate the group structure to find the target
    current = cache_data["app"]["groups"]
    target = None
    found = False
    
    # Try direct path lookup
    current_level = current
    for i, part in enumerate(path):
        if part in current_level:
            if i == len(path) - 1:
                # This is the target
                target = current_level[part]
                found = True
                break
            elif "commands" in current_level[part]:
                current_level = current_level[part]["commands"]
            else:
                # Can't go deeper
                break
        else:
            # Path part not found at this level
            break
    
    # If we found the target, render it
    if found:
        if "signature" in target:
            # This is a command
            render_command_help(path[-1], target, path, cache_data)
        else:
            # This is a group
            render_group_help(path[-1], target, path, cache_data)
    else:
        # Show error with available options
        console.print(f"[bold red]Error:[/bold red] Command or group '[bold]{' '.join(path)}[/bold]' not found.")
        
        # Show what's available at the current level or root if we didn't match any part
        available_level = current
        available_path = []
        
        for i, part in enumerate(path):
            if part in available_level:
                available_path.append(part)
                if "commands" in available_level[part]:
                    available_level = available_level[part]["commands"]
                else:
                    break
            else:
                break
        
        console.print(f"\nAvailable options at [bold]nemo {' '.join(available_path)}[/bold]:")
        
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Name", style="green")
        table.add_column("Description", style="white")
        
        for name, data in available_level.items():
            help_text = data.get("help", "")
            table.add_row(name, help_text)
        
        console.print(table)

def render_command_help(cmd_name: str, cmd_data: Dict[str, Any], path_items: list[str], cache_data: Dict[str, Any]) -> None:
    """Render help for a command with styling that matches Typer exactly."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich import box
    from typer import rich_utils
    from nemo_run.help import render_arguments, render_entrypoint_factories, render_parameter_factories, render_executors
    
    # Create console with proper styling matching typer
    console = Console(
        highlighter=rich_utils.highlighter,
        color_system=rich_utils.COLOR_SYSTEM,
        force_terminal=rich_utils.FORCE_TERMINAL,
        width=rich_utils.MAX_WIDTH,
    )
    
    # Clean up path for display - remove duplicates and normalize
    clean_path = []
    for item in path_items:
        if item != "nemo" or not clean_path:  # Keep only first "nemo"
            clean_path.append(item)
    
    # Command header
    full_command = " ".join(clean_path)
    console.print(f"Usage: {full_command} [OPTIONS] [ARGUMENTS]", style=rich_utils.STYLE_USAGE)
    console.print()
    console.print(Text(f"[Entrypoint] {cmd_data.get('help', '')}", style="cyan"))
    console.print()
    
    # Get all available factories from the cache
    factories = cache_data["data"].get("factories", {})
    
    # Step 1: Render Options section first (to match the expected order)
    render_options_table(console)
    
    # Step 2: Arguments table using the existing render_arguments function
    if cmd_data.get("signature"):
        # Add the required line_no field if missing
        for param in cmd_data.get("signature", []):
            if "line_no" not in param:
                param["line_no"] = ""
                
        console.print(render_arguments(cmd_data["signature"]))
    
    # Step 3: Pre-loaded Entrypoint Factories
    full_namespace = cmd_data.get("full_namespace", "")
    if full_namespace and full_namespace in factories:
        # Ensure each factory has line_no data before rendering
        entrypoint_factories = factories[full_namespace]
        for factory in entrypoint_factories:
            if "line_no" not in factory:
                factory["line_no"] = ""
        
        panel = render_entrypoint_factories(
            entrypoint_factories,
            title="Pre-loaded entrypoint factories, run with --factory"
        )
        console.print(panel)
    
    # Step 4: For each parameter, render both parameter-specific and type-based factories
    # Track which types we've already rendered to avoid duplicates
    rendered_types = set()
    
    if cmd_data.get("signature"):
        for param in cmd_data.get("signature", []):
            param_name = param.get("name", "")
            param_type = param.get("type", "")  # Full type namespace for matching
            
            # First check for parameter-specific factories
            if full_namespace:
                param_namespace = f"{full_namespace}.{param_name}"
                if param_namespace in factories:
                    param_factories = factories[param_namespace]
                    
                    # Ensure each factory has line_no data
                    for factory in param_factories:
                        if "line_no" not in factory:
                            factory["line_no"] = ""
                    
                    panel = render_parameter_factories(param_name, param_type, param_factories)
                    console.print(panel)
            
            # Then check for type-based factories using the full type namespace
            if param_type and param_type != "Any" and param_type in factories and param_type not in rendered_types:
                type_factories = factories[param_type]
                
                # Ensure each factory has line_no data
                for factory in type_factories:
                    if "line_no" not in factory:
                        factory["line_no"] = ""
                
                panel = render_parameter_factories(param_name, param_type, type_factories)
                console.print(panel)
                rendered_types.add(param_type)
    
    # Step 5: Look for executor factories based on their namespace
    # Get executor namespaces from the metadata instead of hardcoding them
    executor_namespaces = cache_data["metadata"].get("executor_namespaces", [])
    
    # If we don't have executor namespaces in the cache (for backward compatibility),
    # populate them now
    if not executor_namespaces:
        cache_executor_classes()
        # Reload the cache to get the executor namespaces
        updated_cache = load_cache()
        executor_namespaces = updated_cache["metadata"].get("executor_namespaces", [])
    
    # Collect all executor factories, using a dictionary to deduplicate
    executor_factories_dict = {}
    for namespace in executor_namespaces:
        if namespace in factories:
            for factory in factories[namespace]:
                # Use factory name and function as the unique key
                key = (factory["name"], factory["fn"])
                if key not in executor_factories_dict:
                    if "line_no" not in factory:
                        factory["line_no"] = ""
                    executor_factories_dict[key] = factory

    # Convert to list for rendering
    executor_factories = list(executor_factories_dict.values())

    # Render the executors panel if we found any executor factories
    if executor_factories:
        panel = render_executors(executor_factories)
        console.print(panel)

def render_options_table(console: Console) -> None:
    """Render the options table using the standard entrypoint options with proper formatting."""
    from rich.panel import Panel
    from rich.table import Table
    from rich import box
    from typer import rich_utils
    
    # Create the options table with proper styling
    options_table = Table(
        show_header=False,
        box=None,  # Key difference - typer uses no internal boxes
        expand=True,
        show_lines=rich_utils.STYLE_OPTIONS_TABLE_SHOW_LINES,
        padding=rich_utils.STYLE_OPTIONS_TABLE_PADDING,
        pad_edge=rich_utils.STYLE_OPTIONS_TABLE_PAD_EDGE,
    )
    
    options_table.add_column("Option", style=rich_utils.STYLE_OPTION, no_wrap=True)
    options_table.add_column("Description", style="white")
    
    # Add all standard options from entrypoint_options with proper formatting
    from nemo_run.cli.api import entrypoint_options
    
    for option_name, option in entrypoint_options.items():
        # Format the option flags
        flag_str = ""
        if option.flags:
            flag_str = ", ".join(option.flags)
        else:
            flag_str = f"--{option_name}"
            
        # Special handling for toggle options that show as --opt/--no-opt
        if isinstance(option.default, bool):
            positive = f"--{option_name}"
            negative = f"--no-{option_name}"
            if option.flags:
                flags = [f for f in option.flags if f.startswith("--")]
                if len(flags) >= 2:
                    positive = flags[0]
                    negative = flags[1]
            flag_str = f"{positive}{' '*max(1, 20-len(positive))}{negative}"
        
        # Format the help text
        help_text = option.help or ""
        
        # Format default differently based on type
        if option.default is not None:
            # Format default value
            if isinstance(option.default, bool):
                default = "no-" + option_name if not option.default else option_name
                default_str = f"[default: {default}]"
            else:
                default_str = f"[default: {option.default}]"
            
            # Position the default value properly
            help_text = f"{help_text}\n{' '*8}{default_str}"
        
        options_table.add_row(flag_str, help_text)
    
    # Add standard Typer options
    options_table.add_row("--help", "Show this message and exit.")
    options_table.add_row("--version", "Show the version and exit.")
    
    console.print(
        Panel(
            options_table,
            title=rich_utils.OPTIONS_PANEL_TITLE,
            border_style=rich_utils.STYLE_OPTIONS_PANEL_BORDER,
            title_align=rich_utils.ALIGN_OPTIONS_PANEL,
            box=box.ROUNDED,
        )
    )

def render_group_help(group_name: str, group_data: Dict[str, Any], path_items: list[str], cache_data: Dict[str, Any]) -> None:
    """Render help for a group, matching Typer's styling exactly."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich import box
    from typer import rich_utils
    
    # Create console with proper styling matching typer
    console = Console(
        highlighter=rich_utils.highlighter,
        color_system=rich_utils.COLOR_SYSTEM,
        force_terminal=rich_utils.FORCE_TERMINAL,
        width=rich_utils.MAX_WIDTH,
    )
    
    # Clean up path for display - remove duplicates and normalize
    clean_path = []
    for item in path_items:
        if item != "nemo" or not clean_path:  # Keep only first "nemo"
            clean_path.append(item)
    
    # Format the help header similar to Typer
    full_path = " ".join(clean_path)
    console.print(f"Usage: {full_path} [OPTIONS] COMMAND [ARGS]...", style=rich_utils.STYLE_USAGE)
    console.print()
    console.print(Text(group_data.get("help", f"[Module] {group_name}"), style="cyan"))
    console.print()
    
    # Options section with proper box styling (exactly matching typer)
    options_table = Table(
        show_header=False,
        box=None,  # Key difference - typer uses no internal boxes
        expand=True,
        show_lines=rich_utils.STYLE_OPTIONS_TABLE_SHOW_LINES,
        padding=rich_utils.STYLE_OPTIONS_TABLE_PADDING,
        pad_edge=rich_utils.STYLE_OPTIONS_TABLE_PAD_EDGE,
    )
    options_table.add_column("Option", style=rich_utils.STYLE_OPTION, no_wrap=True)
    options_table.add_column("Description", style="white")
    options_table.add_row("--help", "Show this message and exit.")
    
    console.print(
        Panel(
            options_table,
            title=rich_utils.OPTIONS_PANEL_TITLE,
            border_style=rich_utils.STYLE_OPTIONS_PANEL_BORDER,
            title_align=rich_utils.ALIGN_OPTIONS_PANEL,
            box=box.ROUNDED,
        )
    )
    
    # Commands section with proper box styling
    commands = []
    for cmd_name, cmd_data in group_data.get("commands", {}).items():
        if isinstance(cmd_data, dict) and "signature" in cmd_data:
            # Extract just the first line of help for the command list
            help_text = cmd_data.get("help", "")
            first_line = help_text.split('\n')[0] if help_text else ""
            commands.append((cmd_name, f"[Entrypoint] {first_line}"))
    
    if commands:
        commands_table = Table(
            show_header=False,
            box=None,  # Key difference - typer uses no internal boxes
            expand=True,
            show_lines=rich_utils.STYLE_COMMANDS_TABLE_SHOW_LINES,
            padding=rich_utils.STYLE_COMMANDS_TABLE_PADDING,
            pad_edge=rich_utils.STYLE_COMMANDS_TABLE_PAD_EDGE,
        )
        commands_table.add_column("Command", style=rich_utils.STYLE_COMMANDS_TABLE_FIRST_COLUMN, no_wrap=True)
        commands_table.add_column("Description", style="white")
        
        for cmd_name, help_text in commands:
            commands_table.add_row(cmd_name, help_text)
        
        console.print(
            Panel(
                commands_table,
                title=rich_utils.COMMANDS_PANEL_TITLE,
                border_style=rich_utils.STYLE_COMMANDS_PANEL_BORDER, 
                title_align=rich_utils.ALIGN_COMMANDS_PANEL,
                box=box.ROUNDED,
            )
        )
    
    # Nested groups (if any)
    nested_groups = []
    for nested_name, nested_data in group_data.get("commands", {}).items():
        if isinstance(nested_data, dict) and "commands" in nested_data and "signature" not in nested_data:
            help_text = nested_data.get("help", f"Commands in the {nested_name} group")
            nested_groups.append((nested_name, help_text))
    
    if nested_groups:
        nested_table = Table(
            show_header=False,
            box=None,  # Key difference - typer uses no internal boxes
            expand=True,
            show_lines=rich_utils.STYLE_COMMANDS_TABLE_SHOW_LINES,
            padding=rich_utils.STYLE_COMMANDS_TABLE_PADDING,
            pad_edge=rich_utils.STYLE_COMMANDS_TABLE_PAD_EDGE,
        )
        nested_table.add_column("Group", style=rich_utils.STYLE_COMMANDS_TABLE_FIRST_COLUMN, no_wrap=True)
        nested_table.add_column("Description", style="white")
        
        for nested_name, help_text in nested_groups:
            nested_table.add_row(nested_name, help_text)
        
        console.print(
            Panel(
                nested_table,
                title="Nested Groups",
                border_style=rich_utils.STYLE_COMMANDS_PANEL_BORDER,
                title_align=rich_utils.ALIGN_COMMANDS_PANEL,
                box=box.ROUNDED,
            )
        )

def _search_workspace_file() -> str | None:
    """
    Search for workspace.py file in the current directory or parent directories.
    This is a simplified version that doesn't import anything but just finds the file path.
    """
    import os
    
    # Start from the current directory
    current_dir = os.getcwd()
    
    # Look for workspace.py in the current directory and parent directories
    while current_dir:
        workspace_path = os.path.join(current_dir, "workspace.py")
        if os.path.isfile(workspace_path):
            return workspace_path
            
        # Move up one directory
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:  # We've reached the root
            break
        current_dir = parent_dir
            
    return None

def cache_cli_structure() -> None:
    """Cache the CLI structure based on entry points."""
    cache_data = load_cache()
    
    # Get entry point mappings
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
    # Import the actual EXECUTOR_CLASSES list from the api module
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