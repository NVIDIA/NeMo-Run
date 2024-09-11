from typing import Any, Callable, Dict, Optional, Type, Union

from fiddle import Buildable, Config, Partial
from fiddle._src.config import NO_VALUE
from fiddle._src.signatures import get_signature
from omegaconf import OmegaConf


def omegaconf_to_buildable(
    cfg: Union[OmegaConf, Dict[str, Any]], target_type: Optional[Type] = None
) -> Buildable:
    """
    Convert an OmegaConf object or dictionary to a Fiddle Buildable.

    Args:
        cfg: OmegaConf object or dictionary to convert.
        target_type: Optional type to use for the Buildable.

    Returns:
        A Fiddle Buildable (Config or Partial) representing the input configuration.
    """
    if isinstance(cfg, dict):
        cfg = OmegaConf.create(cfg)

    if not OmegaConf.is_config(cfg):
        raise ValueError("Input must be an OmegaConf object or a dictionary")

    target = cfg.pop("_target_", None)
    factory = cfg.pop("_factory_", None)

    if factory is not None:
        # Use the factory function to create the Buildable
        buildable = parse_factory(target_type, "_factory_", target_type, factory)
        if not isinstance(buildable, (Config, Partial)):
            buildable = Config(buildable)
    elif target is not None or target_type is not None:
        buildable_target = target_type if target is None else target
        buildable_cls = Partial if target is None else Config

        # Get the signature of the target
        signature = get_signature(buildable_target)

        # Create a dictionary of required arguments with NO_VALUE
        required_args = {
            name: NO_VALUE
            for name, param in signature.parameters.items()
            if param.default == param.empty
            and param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD)
        }

        # Update required_args with values from cfg
        for name in required_args:
            if name in cfg:
                required_args[name] = cfg.pop(name)

        # Create the buildable with required arguments
        buildable = buildable_cls(buildable_target, **required_args)
    else:
        raise ValueError(
            "Either '_target_', '_factory_' must be specified in the config or 'target_type' must be provided"
        )

    # Set remaining values from cfg
    for key, value in cfg.items():
        if OmegaConf.is_config(value):
            setattr(buildable, key, omegaconf_to_buildable(value))
        elif isinstance(value, list):
            setattr(
                buildable,
                key,
                [
                    omegaconf_to_buildable(item) if OmegaConf.is_config(item) else item
                    for item in value
                ],
            )
        else:
            setattr(buildable, key, value)

    return buildable


def parse_yaml_to_buildable(fn: Callable, yaml_path: str) -> Buildable:
    """
    Parse a YAML file and create a Fiddle Buildable for the given function.

    Args:
        fn: The function to create a Buildable for.
        yaml_path: Path to the YAML file.

    Returns:
        A Fiddle Buildable representing the function with YAML config applied.
    """
    yaml_config = OmegaConf.load(yaml_path)
    return omegaconf_to_buildable(yaml_config, target_type=fn)
