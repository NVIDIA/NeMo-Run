import inspect
import sys
from pathlib import Path
from types import FrameType
from typing import Any, Callable, Dict, List, Optional, Set, Type


def process_args(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    func: Callable,
    get_fn: Callable[[Any], Any],
) -> Dict[str, Any]:
    from nemo_run.config import Config

    # Process positional arguments
    processed_args = [process_single_arg(arg, get_fn) for arg in args]

    # Process keyword arguments
    processed_kwargs = {k: process_single_arg(v, get_fn) for k, v in kwargs.items()}

    # Combine processed positional and keyword arguments
    result = dict(enumerate(processed_args))
    result.update(processed_kwargs)
    return result

def process_single_arg(v: Any, get_fn: Callable[[Any], Any]) -> Any:
    from nemo_run.config import Config

    if isinstance(v, (str, int, float, bool, type(None))):
        return v
    elif isinstance(v, Path):
        return Config(Path, str(v))
    elif isinstance(v, (list, tuple)):
        return [process_single_arg(item, get_fn) for item in v]
    elif isinstance(v, dict):
        return {key: process_single_arg(value, get_fn) for key, value in v.items()}
    elif callable(v) or isinstance(v, type) or (isinstance(v, set) and all(isinstance(item, type) for item in v)):
        return v
    else:
        try:
            return get_fn(v)
        except Exception:
            return v  # If we can't process it, return the original value

def get_full_signature(cls: Type) -> inspect.Signature:
    """Get the full signature of a class, including inherited parameters."""
    mro = inspect.getmro(cls)
    parameters: Dict[str, inspect.Parameter] = {}
    for c in reversed(mro):
        if hasattr(c, '__init__'):
            sig = inspect.signature(c.__init__)
            for name, param in sig.parameters.items():
                if name != 'self':
                    parameters[name] = param

    # Sort parameters based on their kind
    sorted_parameters = sorted(
        parameters.values(),
        key=lambda p: (
            p.kind == inspect.Parameter.POSITIONAL_ONLY,
            p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD,
            p.kind == inspect.Parameter.VAR_POSITIONAL,
            p.kind == inspect.Parameter.KEYWORD_ONLY,
            p.kind == inspect.Parameter.VAR_KEYWORD
        )
    )

    return inspect.Signature(sorted_parameters)

def wrap_init(frame: FrameType, get_fn: Callable, register_fn: Callable, cls_to_ignore: Set[Type]):
    cls = frame.f_locals.get("self").__class__
    if cls not in cls_to_ignore:
        full_signature = get_full_signature(cls)
        args = inspect.getargvalues(frame)

        # Get all arguments passed to the constructor
        all_args = args.args[1:]  # Exclude 'self'
        all_kwargs = {k: v for k, v in args.locals.items() if k not in ('self', '__class__')}

        # Process all arguments
        processed_args = {}
        for name, param in full_signature.parameters.items():
            if name in all_kwargs:
                processed_args[name] = process_single_arg(all_kwargs[name], get_fn)
            elif param.default is not param.empty:
                processed_args[name] = param.default

        from nemo_run.config import Config

        cfg = Config(cls, **processed_args)
        if register_fn:
            register_fn(frame.f_locals.get("self"), cfg)


class _CaptureContext:
    def __init__(
        self, get_fn: Callable, register_fn: Callable, cls_to_ignore: Optional[Set[Type]] = None
    ):
        self.get = get_fn
        self.register = register_fn
        self.cls_to_ignore = cls_to_ignore or set()
        self.old_profile = None

    def __enter__(self):
        self.old_profile = sys.getprofile()
        sys.setprofile(self._profile_func)

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.setprofile(self.old_profile)

    def _profile_func(self, frame: FrameType, event: str, arg: Any):
        if event == "call" and frame.f_code.co_name == "__init__":
            wrap_init(frame, self.get, self.register, self.cls_to_ignore)
        return self.old_profile
