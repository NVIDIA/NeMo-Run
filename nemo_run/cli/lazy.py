import ast
import builtins
import contextlib
import importlib
import os
import re
import shlex
import sys
from dataclasses import dataclass, field
import inspect
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterator, Optional, TYPE_CHECKING

from fiddle import Buildable, daglish
from fiddle._src import signatures
from fiddle._src.building import _state
from fiddle.experimental import serialization
from omegaconf import DictConfig, OmegaConf

from nemo_run.config import Partial

if TYPE_CHECKING:
    from nemo_run.cli.cli_parser import RunContext


@contextlib.contextmanager
def lazy_imports(fallback_to_lazy: bool = False) -> Iterator[None]:
    original_import = builtins.__import__
    lazy_modules: dict[str, LazyModule] = {}

    def lazy_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level != 0:
            raise ImportError("Relative imports are not supported in lazy imports")

        if fallback_to_lazy:
            try:
                return original_import(name, globals, locals, fromlist, level)
            except ImportError:
                pass

        if name not in lazy_modules:
            lazy_modules[name] = LazyModule(name)

        module = lazy_modules[name]

        if not fromlist:
            return module

        for attr in fromlist:
            if "." in attr:  # Handle nested attributes
                parts = attr.split(".")
                current = module
                for part in parts[:-1]:
                    if not hasattr(current, part):
                        setattr(current, part, LazyModule(f"{current.name}.{part}"))
                    current = getattr(current, part)
                setattr(current, parts[-1], LazyTarget(f"{current.name}.{parts[-1]}"))
            else:
                if not hasattr(module, attr):
                    setattr(module, attr, LazyModule(f"{name}.{attr}"))

        return module

    try:
        builtins.__import__ = lazy_import
        yield
    finally:
        builtins.__import__ = original_import


class LazyEntrypoint(Buildable):
    """
    A class for lazy initialization and configuration of entrypoints.

    This class allows for the creation of a configurable entrypoint that can be
    modified with overwrites, which are only applied when the `resolve` method is called.
    """

    def __init__(
        self,
        target: Callable | str,
        factory: Callable | str | None = None,
        yaml: str | DictConfig | Path | None = None,
        overwrites: list[str] | None = None,
    ):
        cmd_args = []
        if isinstance(target, str) and (" " in target or target.endswith(".py")):
            cmd = []
            for arg in shlex.split(target):
                # Skip the --lazy flag and export flags
                if arg == "--lazy" or arg.startswith("--to-"):
                    continue
                if "=" in arg:
                    cmd_args.append(arg)
                else:
                    cmd.append(arg)
            target = " ".join(cmd)
        elif isinstance(target, LazyModule):
            target = LazyTarget(target.name)

        if isinstance(factory, LazyModule):
            factory = factory.name
        # Handle @ syntax in factory parameter
        elif (
            isinstance(factory, str)
            and factory.startswith("@")
            and _is_config_file_path(factory[1:])
        ):
            try:
                factory_config = load_config_from_path(factory)

                # If the config has a _factory_ key, use that as the factory
                if "_factory_" in factory_config:
                    factory = factory_config["_factory_"]
                    # Remove _factory_ from the config so it's not processed twice
                    remaining_config = OmegaConf.create(
                        {k: v for k, v in factory_config.items() if k != "_factory_"}
                    )
                    # Convert remaining config to arguments and add them
                    factory_args = dictconfig_to_dot_list(remaining_config)
                    cmd_args.extend([f"{name}{op}{value}" for name, op, value in factory_args])
                else:
                    # If no _factory_ key, just add the config to arguments and clear the factory
                    factory_args = dictconfig_to_dot_list(factory_config)
                    cmd_args.extend([f"{name}{op}{value}" for name, op, value in factory_args])
                    factory = None

            except ValueError as e:
                print(f"Warning: Error loading factory from {factory}: {str(e)}")
                # Keep the original string if loading fails

        self._target_: Callable = LazyTarget(target) if isinstance(target, str) else target
        self._factory_ = factory
        self._args_ = []

        # Process command args first
        if cmd_args:
            self._add_overwrite(*cmd_args)

        # Process all configuration with the consolidated method
        # This handles the main config file and any @ references in the overwrites
        remaining_overwrites = self._parse_config(yaml, overwrites)

        # Process any remaining overwrites normally
        if remaining_overwrites:
            self._add_overwrite(*remaining_overwrites)

    def resolve(self, ctx: Optional["RunContext"] = None) -> Partial:
        from nemo_run.cli.cli_parser import parse_cli_args, parse_factory

        fn = self._target_
        if self._factory_:
            if isinstance(self._factory_, str):
                if isinstance(self._target_, LazyTarget):
                    target = self._target_.target
                else:
                    target = self._target_

                fn = parse_factory(target, "", target, self._factory_)
            else:
                fn = self._factory_()
        else:
            if isinstance(fn, LazyTarget):
                fn = fn.target

        _fn = fn
        if hasattr(fn, "__fn_or_cls__"):
            _fn = fn.__fn_or_cls__

        sig = inspect.signature(_fn)
        param_names = sig.parameters.keys()

        dotlist = dictconfig_to_dot_list(
            _args_to_dictconfig(self._args_), has_factory=self._factory_ is not None
        )
        _args = [f"{name}{op}{value}" for name, op, value in dotlist]

        out = parse_cli_args(fn, _args)

        if "ctx" in param_names:
            if not ctx:
                raise ValueError("ctx is required for this function")
            out.ctx = ctx

        return out

    def __getattr__(self, item: str) -> "LazyEntrypoint":
        """
        Handle attribute access by returning a new LazyFactory with an updated path.

        Args:
            item: The attribute name being accessed.

        Returns:
            A new LazyFactory instance with the updated path.
        """
        base, _, path = self._target_path_.partition(":")
        new_path = f"{path}.{item}" if path else item
        out = LazyEntrypoint(f"{base}:{new_path}")
        out._args_ = self._args_
        return out

    def __setattr__(self, item: str, value: Any):
        """
        Handle attribute assignment by storing the value in overwrites.

        Args:
            item: The attribute name being assigned.
            value: The value to assign to the attribute.
        """
        if item in {"_target_", "_factory_", "_args_"}:
            object.__setattr__(self, item, value)
        else:
            if isinstance(value, LazyEntrypoint):
                return

            _, _, path = self._target_path_.partition(":")
            full_path = f"{path}.{item}" if path else item
            self._args_.append((full_path, "=", value))

    def __add__(self, other: Any) -> "LazyEntrypoint":
        return self._record_operation("+=", other)

    def __sub__(self, other: Any) -> "LazyEntrypoint":
        return self._record_operation("-=", other)

    def __mul__(self, other: Any) -> "LazyEntrypoint":
        return self._record_operation("*=", other)

    def __iadd__(self, other: Any) -> "LazyEntrypoint":
        return self._record_operation("+=", other)

    def __isub__(self, other: Any) -> "LazyEntrypoint":
        return self._record_operation("-=", other)

    def __imul__(self, other: Any) -> "LazyEntrypoint":
        return self._record_operation("*=", other)

    def _record_operation(self, op: str, other: Any) -> "LazyEntrypoint":
        _, _, path = self._target_path_.partition(":")
        self._args_.append((path, op, other))
        return self

    def _add_overwrite(self, *overwrites: str):
        for overwrite in overwrites:
            # Skip CLI flags like --lazy, --to-yaml, etc.
            if overwrite.startswith("--"):
                continue

            # Split into key, op, value
            match = re.match(r"([^=]+)([*+-]?=)(.*)", overwrite)
            if not match:
                raise ValueError(f"Invalid overwrite format: {overwrite}")
            key, op, value = match.groups()
            self._args_.append((key, op, value))

    def _parse_config(
        self, config: str | DictConfig | Path | None = None, overwrites: list[str] | None = None
    ):
        """
        Parse configuration files and CLI overwrites, handling @ syntax references.

        This method handles loading and merging configurations from various sources:
        1. Main config file (YAML, JSON, or TOML)
        2. CLI overwrites that might contain @ syntax references to other config files

        Args:
            config: Path to config file or DictConfig object (optional)
            overwrites: List of CLI overwrites that might contain @ syntax (optional)

        Returns:
            Remaining overwrites that don't use @ syntax
        """
        from nemo_run.cli.config import ConfigSerializer

        # Start with empty config if none provided
        to_parse = OmegaConf.create({})

        # Load the main config file if provided
        if config is not None:
            if isinstance(config, DictConfig):
                to_parse = config
            elif isinstance(config, (str, Path)):
                try:
                    serializer = ConfigSerializer()
                    # Convert to Path object for consistent handling
                    path = Path(config) if isinstance(config, str) else config

                    # Load based on file extension
                    if path.suffix.lower() in (".yaml", ".yml", ".json", ".toml"):
                        # Load as raw dict first to avoid resolving references
                        config_data = serializer.load_dict(path)
                        to_parse = OmegaConf.create(config_data)
                    else:
                        # Handle as raw string (YAML format)
                        to_parse = OmegaConf.create(config)
                except Exception as e:
                    raise ValueError(f"Error loading config file {config}: {str(e)}")
            else:
                raise ValueError(f"Invalid config type: {type(config)}")

        # Extract factory if present
        if "_factory_" in to_parse:
            self._factory_ = to_parse["_factory_"]
        if "run" in to_parse and "factory" in to_parse["run"]:
            self._factory_ = to_parse["run"]["factory"]

        # Handle any @ syntax in the overwrites
        remaining_overwrites = []
        if overwrites:
            for overwrite in overwrites:
                # Skip CLI flags like --lazy, --to-yaml, etc.
                if overwrite.startswith("--"):
                    continue

                # Parse the overwrite to get key, op, value
                match = re.match(r"([^=]+)([*+-]?=)(.*)", overwrite)
                if not match:
                    raise ValueError(f"Invalid overwrite format: {overwrite}")

                key, op, value = match.groups()

                # If this is a @ syntax, load the config and merge it
                if (
                    isinstance(value, str)
                    and value.startswith("@")
                    and _is_config_file_path(value[1:])
                ):
                    try:
                        # Load the referenced config file
                        loaded_config = load_config_from_path(value)

                        # Update the main config with this loaded config
                        # If the key already exists in to_parse, we need special handling
                        if key in to_parse:
                            # If both are dictionaries, merge them
                            if isinstance(to_parse[key], DictConfig) and isinstance(
                                loaded_config, DictConfig
                            ):
                                to_parse[key] = OmegaConf.merge(to_parse[key], loaded_config)
                            else:
                                # Otherwise, the @ syntax takes precedence
                                to_parse[key] = loaded_config
                        else:
                            # Simple case: just add it to the config
                            to_parse[key] = loaded_config
                    except ValueError as e:
                        print(f"Warning: {str(e)}")
                        # Add to remaining overwrites if loading fails
                        remaining_overwrites.append(overwrite)
                else:
                    # This is not an @ syntax, keep it for normal processing
                    remaining_overwrites.append(overwrite)

        # Convert the merged config to args
        self._args_.extend(dictconfig_to_dot_list(to_parse, has_factory=self._factory_ is not None))

        # Return the remaining overwrites to be processed normally
        return remaining_overwrites

    @property
    def _target_path_(self) -> str:
        if isinstance(self._target_, LazyTarget):
            return self._target_.import_path

        return f"{self._target_.__module__}.{self._target_.__name__}"

    @property
    def is_lazy(self) -> bool:
        return True

    def __build__(self, *args, **kwargs):
        buildable = self.resolve()
        return buildable.__build__(*args, **kwargs)

    def __flatten__(self):
        if _state.in_build:
            buildable = self.resolve()
            return buildable.__flatten__()

        return _flatten_lazy_entrypoint(self)

    @classmethod
    def __unflatten__(cls, values, metadata):
        if _state.in_build:
            buildable = cls.resolve()
            return buildable.__unflatten__(values, metadata)

        return _unflatten_lazy_entrypoint(values, metadata)

    def __path_elements__(self):
        return (
            daglish.Attr("_target_"),
            daglish.Attr("_factory_"),
            daglish.Attr("_args_"),
        )

    @property
    def __fn_or_cls__(self):
        return _dummy_fn

    @property
    def __arguments__(self):
        return {
            "_target_": self._target_,
            "_factory_": self._factory_,
            "_args_": self._args_,
        }

    @property
    def __signature_info__(self):
        return signatures.SignatureInfo(signature=signatures.get_signature(_dummy_fn))

    @property
    def __argument_tags__(self):
        return {}

    @property
    def __argument_history__(self):
        return {}


@dataclass
class LazyTarget:
    import_path: str
    script: str = field(default="")

    def __post_init__(self):
        # Check if it's a CLI command
        if " " in self.import_path or self.import_path.endswith(".py"):
            cmd = shlex.split(self.import_path)
            if cmd[0].endswith(".py"):
                script_path = Path(cmd[0])
                if not script_path.exists():
                    raise FileNotFoundError(f"Script '{script_path}' not found.")

                self.script = script_path.read_text()
                if len(cmd) > 1:
                    self.import_path = " ".join(cmd[1:])
            if (
                cmd[0] in ("nemo", "nemo_run")
                or cmd[0].endswith("/nemo")
                or cmd[0].endswith("/nemo_run")
            ):
                self.import_path = " ".join(cmd[1:])

    def __call__(self, *args, **kwargs):
        return self.target(*args, **kwargs)

    def _load_real_object(self):
        module_name, _, object_name = self.import_path.rpartition(".")
        module = importlib.import_module(module_name)
        self._target_fn = getattr(module, object_name)

    def _load_entrypoint(self):
        from nemo_run.cli.api import list_entrypoints

        entrypoints = list_entrypoints()

        if self.script:
            entrypoint = _load_entrypoint_from_script(self.script)
            self._target_fn = entrypoint.fn
        else:
            parts = self.import_path.split(" ")
            if parts[0] not in entrypoints:
                available_cmds = ", ".join(sorted(entrypoints.keys()))
                raise ValueError(
                    f"Entrypoint '{parts[0]}' not found. Available top-level entrypoints: {available_cmds}"
                )
            output = entrypoints[parts[0]]

            # Re-key the nested entrypoint dict to include 'name' attribute as keys
            def rekey_entrypoints(entries):
                if not isinstance(entries, dict):
                    return entries

                result = {}
                for key, value in entries.items():
                    result[key] = value
                    if hasattr(value, "name") and value.name != key:
                        result[value.name] = value
                    elif isinstance(value, dict):
                        result[key] = rekey_entrypoints(value)
                return result

            # Only rekey if we're dealing with a dictionary
            if isinstance(output, dict):
                output = rekey_entrypoints(output)

            if len(parts) > 1:
                for part in parts[1:]:
                    # Skip args with - or -- prefix or containing = as they're parameters, not subcommands
                    if part.startswith("-") or "=" in part:
                        continue

                    if isinstance(output, dict):
                        if part in output:
                            output = output[part]
                        else:
                            # Collect available commands for error message
                            available_cmds = sorted(output.keys())
                            raise ValueError(
                                f"Subcommand '{part}' not found for entrypoint '{parts[0]}'. "
                                f"Available subcommands: {', '.join(available_cmds)}"
                            )
                    else:
                        # We've reached an entrypoint object but tried to access a subcommand
                        entrypoint_name = getattr(output, "name", parts[0])
                        raise ValueError(
                            f"'{entrypoint_name}' is a terminal entrypoint and does not have subcommand '{part}'. "
                            f"You may have provided an incorrect command structure."
                        )

            # If output is a dict, we need to get the default entrypoint
            if isinstance(output, dict):
                raise ValueError(
                    f"Incomplete command: '{self.import_path}'. Please specify a subcommand. "
                    f"Available subcommands: {', '.join(sorted(output.keys()))}"
                )

            self._target_fn = output.fn

    @property
    def target(self):
        if not hasattr(self, "_target_fn"):
            if " " in self.import_path or self.import_path.endswith(".py"):
                self._load_entrypoint()
            else:
                self._load_real_object()
        return self._target_fn

    @property
    def __is_lazy__(self):
        return True


class LazyModule(ModuleType):
    """
    A class representing a lazily loaded module.

    Attributes:
        name (str): The name of the module.
        _lazy_attrs (Dict[str, Union[LazyObject, 'LazyModule']]): A dictionary of lazy attributes.

    Example:
        >>> lazy_mod = LazyModule("nemo.collections.llm")
        >>> model = lazy_mod.GPTModel(config)  # The GPTModel class is loaded here
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self._lazy_attrs: dict[str, Any] = {}

    def __getattr__(self, name):
        if name not in self._lazy_attrs:
            full_name = f"{self.name}.{name}"
            if "." in self.name:  # This is a nested module
                self._lazy_attrs[name] = LazyModule(full_name)
            else:
                self._lazy_attrs[name] = LazyTarget(full_name)
        return self._lazy_attrs[name]

    def __call__(self, *args, **kwargs):
        real_object = self._import()
        return real_object(*args, **kwargs)

    def _import(self):
        return import_module(self.name)

    def __dir__(self):
        return list(self._lazy_attrs.keys())

    @property
    def __signature__(self):
        return None

    @property
    def __annotations__(self):
        return {}

    def __getstate__(self):
        return {"name": self.name, "_lazy_attrs": self._lazy_attrs}

    def __setstate__(self, state):
        self.name = state["name"]
        self._lazy_attrs = state["_lazy_attrs"]
        super().__init__(self.name)

    @property
    def __is_lazy__(self):
        return True


def dictconfig_to_dot_list(
    config: DictConfig,
    prefix: str = "",
    resolve: bool = True,
    has_factory: bool = False,
    has_target: bool = False,
) -> list[tuple[str, str, Any]]:
    """
    Convert a DictConfig to a list of dot notation overwrites with special handling for _factory_.

    Args:
        config (DictConfig): The DictConfig to convert.
        prefix (str): The current prefix for nested keys.

    Returns:
        List[tuple[str, str, Any]]: A list of dot notation overwrites.

    Examples:
        >>> cfg = OmegaConf.create({
        ...     "model": {
        ...         "_factory_": "llama3_70b(input_1=5)",
        ...         "hidden_size*=": 1024,
        ...         "num_layers": 12
        ...     },
        ...     "a": 1,
        ...     "b": {"c": 2, "d": [3, 4]},
        ...     "e": "test"
        ... })
        >>> dictconfig_to_dot_list(cfg)
        ['model=llama3_70b(input_1=5)', 'model.hidden_size*=1024', 'model.num_layers=12', 'a=1', 'b.c=2', 'b.d=[3, 4]', 'e="test"']
    """
    result: list[tuple[str, str, Any]] = []

    if not prefix and resolve:
        OmegaConf.resolve(config)

    if prefix and not has_target and not has_factory and "_target_" not in config:
        result.append((prefix, "=", "Config"))

    for key, value in config.items():
        op_match = re.match(r"(.+?)([*+-]?=)$", key)
        if op_match:
            clean_key, op = op_match.groups()
        else:
            clean_key, op = key, "="

        full_key = f"{prefix}.{clean_key}" if prefix else clean_key

        if isinstance(value, DictConfig):
            if not prefix and key == "run":
                continue
            if "_target_" in value:
                target = value["_target_"]
                if not target.startswith(("Config", "Partial")):
                    if value.pop("_partial_", False):
                        target = f"Partial[{target}]"
                    else:
                        target = f"Config[{target}]"
                result.append((full_key, "=", target))
                remaining_config = OmegaConf.create(
                    {k: v for k, v in value.items() if k != "_target_"}
                )
                result.extend(
                    dictconfig_to_dot_list(
                        remaining_config, full_key, has_target=True, has_factory=has_factory
                    )
                )
            elif "_factory_" in value:
                result.append((full_key, "=", value["_factory_"]))
                remaining_config = OmegaConf.create(
                    {k: v for k, v in value.items() if k != "_factory_"}
                )
                result.extend(dictconfig_to_dot_list(remaining_config, full_key, has_factory=True))
            else:
                result.extend(dictconfig_to_dot_list(value, full_key, has_factory=has_factory))
        elif isinstance(value, list):
            result.append((full_key, op, value))
        else:
            result.append((full_key, op, value))

    return result


def _args_to_dictconfig(args: list[tuple[str, str, Any]]) -> DictConfig:
    """Convert a list of (path, op, value) tuples to a DictConfig."""
    config = {}
    structure_args = []
    value_args = []

    # Separate structure-defining args from value assignments
    for path, op, value in args:
        if "." in path:
            value_args.append((path, op, value))
        else:
            structure_args.append((path, op, value))

    # Process top-level assignments first
    for path, op, value in structure_args:
        # Handle @ syntax in values here
        if isinstance(value, str) and value.startswith("@") and _is_config_file_path(value[1:]):
            try:
                value = load_config_from_path(value)
            except ValueError as e:
                print(f"Warning: {str(e)}")
                # Keep the original string if loading fails

        if op != "=":
            path = f"{path}{op}"
        config[path] = value

    # Then process all value assignments
    for path, op, value in value_args:
        # Handle @ syntax in values here
        if isinstance(value, str) and value.startswith("@") and _is_config_file_path(value[1:]):
            try:
                value = load_config_from_path(value)
            except ValueError as e:
                print(f"Warning: {str(e)}")
                # Keep the original string if loading fails

        current = config
        *parts, last = path.split(".")

        # Create nested structure
        for part in parts:
            if part not in current:
                current[part] = {}
            elif not isinstance(current[part], dict):
                if isinstance(current[part], str):
                    current[part] = {"_factory_": current[part]}
                else:
                    current[part] = {}  # Convert to dict if it wasn't already
            current = current[part]

        # Add the operator suffix if it's not a simple assignment
        if op != "=":
            last = f"{last}{op}"

        current[last] = value

    return OmegaConf.create(config)


def _is_config_file_path(path_str: str) -> bool:
    """
    Check if a string appears to be a path to a supported config file.

    Args:
        path_str (str): The string to check

    Returns:
        bool: True if the string appears to be a config file path, False otherwise
    """
    # Check if there's a section specifier
    if ":" in path_str:
        path_str = path_str.split(":", 1)[0]

    # Check for supported extensions
    SUPPORTED_EXTENSIONS = (".yaml", ".yml", ".json", ".toml")
    return any(path_str.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS)


def _dummy_fn(*args, **kwargs):
    """Dummy function to satisfy fdl.Buildable's requirements."""
    raise NotImplementedError("This function should never be called directly.")


def _flatten_lazy_entrypoint(instance):
    return (instance._target_, instance._factory_, instance._args_), None


def _unflatten_lazy_entrypoint(values, metadata):
    target, factory, args = list(values)
    instance = LazyEntrypoint(target, factory)
    instance._args_ = args
    return instance


def _flatten_lazy_target(instance):
    return (instance.import_path, instance.script), None


def _unflatten_lazy_target(values, metadata):
    import_path, script = values
    return LazyTarget(import_path, script=script)


serialization.register_node_traverser(
    LazyTarget,
    flatten_fn=_flatten_lazy_target,
    unflatten_fn=_unflatten_lazy_target,
    path_elements_fn=lambda x: (daglish.Attr("import_path"), daglish.Attr("script")),
)


def _load_entrypoint_from_script(script_content: str, module_name: str = "__dynamic_module__"):
    """
    Load the script as a module from a string and extract its __main__ block.

    Args:
        script_content (str): String containing the Python script content.
        module_name (str): Name for the dynamically created module.

    Returns:
        ModuleType: The loaded module.
    """
    # Create a new module
    module = ModuleType(module_name)
    sys.modules[module_name] = module

    # Set the LAZY_CLI environment variable
    os.environ["LAZY_CLI"] = "true"

    # Parse the script and extract the main block
    tree = ast.parse(script_content)

    # Execute the entire script content in the module's namespace
    exec(script_content, module.__dict__)

    main_block = None
    for node in tree.body:
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
            if (
                isinstance(node.test.left, ast.Name)
                and node.test.left.id == "__name__"
                and isinstance(node.test.comparators[0], ast.Str)
                and node.test.comparators[0].s == "__main__"
            ):
                main_block = node.body
                break

    if main_block:
        # Create a new function with the main block content
        main_func = ast.FunctionDef(
            name="__main__",
            args=ast.arguments(
                posonlyargs=[],
                args=[],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[],
            ),
            body=main_block,
            decorator_list=[],
        )

        # Wrap the function in a module
        wrapper_module = ast.Module(body=[main_func], type_ignores=[])

        # Compile and execute the wrapper module
        code = compile(ast.fix_missing_locations(wrapper_module), filename="<string>", mode="exec")
        exec(code, module.__dict__)

        # Execute the __main__ function
        if hasattr(module, "__main__"):
            module.__main__()

        # Reset the LAZY_CLI environment variable
        os.environ["LAZY_CLI"] = "false"

    from nemo_run.cli.api import MAIN_ENTRYPOINT

    if MAIN_ENTRYPOINT is None:
        raise ValueError("No entrypoint function found in script.")

    return MAIN_ENTRYPOINT


def import_module(qualname_str: str) -> Any:
    module_name, _, attr_name = qualname_str.rpartition(".")
    module = importlib.import_module(module_name)
    return getattr(module, attr_name) if attr_name else module


def load_config_from_path(path_with_syntax: str) -> Any:
    """
    Load a configuration file using the @ syntax.

    This function handles loading configuration files with the @ syntax, including:
    - Basic file loading: @path/to/config.yaml
    - Section extraction: @path/to/config.yaml:section
    - Automatic structure detection: Will handle both nested and flat configurations

    Examples:
        # Nested config (model.yaml):
        model:
            _target_: Model
            hidden_size: 256

        # Flat config (model.yaml):
        _target_: Model
        hidden_size: 256

        Both can be loaded with: model=@model.yaml

    Args:
        path_with_syntax (str): Path to the config file with @ syntax

    Returns:
        DictConfig: The loaded configuration as a DictConfig or specific section

    Raises:
        ValueError: If the file path is invalid or the file doesn't exist
    """
    from nemo_run.cli.config import ConfigSerializer
    from omegaconf import OmegaConf
    import os

    # Extract file path and optional section
    section_match = re.match(r"^@([\w\./\\-]+)(?::(\w+))?$", path_with_syntax)
    if not section_match:
        raise ValueError(f"Invalid config file format: {path_with_syntax}")

    config_path, section = section_match.groups()

    # Validate the path exists
    if not os.path.exists(config_path):
        raise ValueError(f"Config file not found: {config_path}")

    # Use the ConfigSerializer to load the file as a dictionary
    serializer = ConfigSerializer()
    try:
        config_data = serializer.load_dict(config_path)

        # If a section is specified, extract just that section
        if section:
            if section not in config_data:
                raise ValueError(f"Section '{section}' not found in config file {config_path}")
            config_data = config_data[section]
            return OmegaConf.create(config_data)

        # Check if this is a flat configuration (no top-level component name)
        # We consider it flat if it has any of these indicators:
        # 1. Has _target_ at root level
        # 2. Has _factory_ at root level
        # 3. All top-level keys are typical config keys (not component names)
        is_flat = (
            "_target_" in config_data
            or "_factory_" in config_data
            or all(not isinstance(v, dict) for v in config_data.values())
        )

        if is_flat:
            # For flat configs, we return as-is
            return OmegaConf.create(config_data)
        else:
            # For nested configs, check if there's a single component that matches a known parameter
            # If we have a file with structure like: model: {...}, we should extract just the model part
            if len(config_data) == 1:
                component_name = next(iter(config_data.keys()))
                # Return just the component configuration, not the wrapper
                return OmegaConf.create(config_data[component_name])
            else:
                # Return the entire config for multi-component nested configs
                return OmegaConf.create(config_data)

    except Exception as e:
        raise ValueError(f"Error loading config file {config_path}: {str(e)}")
