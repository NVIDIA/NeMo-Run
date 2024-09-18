import ast
import builtins
import contextlib
import importlib
import os
import re
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Callable, Iterator

from fiddle import Buildable, daglish
from fiddle._src import signatures
from fiddle._src.building import _state
from fiddle.experimental import serialization
from omegaconf import DictConfig, OmegaConf

from nemo_run.config import Partial

if TYPE_CHECKING:
    from nemo_run.cli.api import Entrypoint


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
    ):
        cmd_args = []
        if isinstance(target, str) and (" " in target or target.endswith(".py")):
            cmd = []
            for arg in shlex.split(target):
                if "=" in arg:
                    cmd_args.append(arg)
                else:
                    cmd.append(arg)
            target = " ".join(cmd)
        elif isinstance(target, LazyModule):
            target = LazyTarget(target.name)

        if isinstance(factory, LazyModule):
            factory = factory.name

        self._target_: Callable = LazyTarget(target) if isinstance(target, str) else target
        self._factory_ = factory
        self._args_ = []

        if cmd_args:
            self._add_overwrite(*cmd_args)

        if yaml is not None:
            self._parse_yaml(yaml)

    def resolve(self) -> Partial:
        from nemo_run.cli.cli_parser import parse_cli_args, parse_factory

        fn = self._target_
        if self._factory_:
            if isinstance(self._target_, LazyTarget):
                target = self._target_.target
            else:
                target = self._target_

            fn = parse_factory(target, "", target, self._factory_)
        else:
            if isinstance(fn, LazyTarget):
                fn = fn.target

        args = [f"{name}{op}{value}" for name, op, value in self._args_]

        return parse_cli_args(fn, args)

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
            # Split into key, op, value
            match = re.match(r"([^=]+)([*+-]?=)(.*)", overwrite)
            if not match:
                raise ValueError(f"Invalid overwrite format: {overwrite}")
            key, op, value = match.groups()
            self._args_.append((key, op, value))

    def _parse_yaml(self, yaml: str | DictConfig | Path):
        if isinstance(yaml, DictConfig):
            to_parse = yaml
        elif isinstance(yaml, Path):
            with open(yaml, "r") as f:
                to_parse = OmegaConf.load(f)
        elif isinstance(yaml, str):
            if yaml.endswith(".yaml") or yaml.endswith(".yml"):
                with open(yaml, "r") as f:
                    to_parse = OmegaConf.load(f)
            else:
                to_parse = OmegaConf.create(yaml)
        else:
            raise ValueError(f"Invalid yaml type: {type(yaml)}")

        if "_factory_" in to_parse:
            self._factory_ = to_parse["_factory_"]
        if "run" in to_parse and "factory" in to_parse["run"]:
            self._factory_ = to_parse["run"]["factory"]

        self._args_.extend(dictconfig_to_dot_list(to_parse, has_factory=self._factory_ is not None))

    @property
    def _target_path_(self) -> str:
        if isinstance(self._target_, LazyTarget):
            return self._target_.import_path

        return f"{self._target_.__module__}.{self._target_.__name__}"

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
            if cmd[0] in ("nemo", "nemo_run"):
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
                raise ValueError(
                    f"Entrypoint {parts[0]} not found. Available entrypoints: {list(entrypoints.keys())}"
                )
            output = entrypoints[parts[0]]
            if len(parts) > 1:
                for part in parts[1:]:
                    if part in output:
                        output = output[part]
                    else:
                        raise ValueError(
                            f"Entrypoint {self.import_path} not found. Available entrypoints: {list(entrypoints.keys())}"
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
            if isinstance(value, str):
                value = f'"{value}"'
            result.append((full_key, op, value))

    return result


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


def _load_entrypoint_from_script(
    script_content: str, module_name: str = "__dynamic_module__"
) -> "Entrypoint":
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


if __name__ == "__main__":
    task = LazyEntrypoint("nemo.collections.llm.pretrain", factory="llama3_8b")

    task.model = "llama3_70b(input_1=5)"
    task.trainer = "my_trainer"
    task.data = "my_data"

    a = 5
