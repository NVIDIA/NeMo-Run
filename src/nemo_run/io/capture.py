import inspect
import sys
from pathlib import Path
from types import FrameType
from typing import Any, Callable, Dict, Optional, Set, Type


def process_args(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    func: Callable,
    get_fn: Callable[[Any], Any],
) -> Dict[str, Any]:
    from nemo_run.config import Config

    arg_names = inspect.getfullargspec(func).args
    if inspect.ismethod(func) or arg_names[0] == "self":
        arg_names = arg_names[1:]  # Remove 'self' for methods and class __init__
        args = args[1:]  # Remove 'self' from args as well

    processed_args = dict(zip(arg_names, args))
    processed_args.update(kwargs)

    for k, v in list(processed_args.items()):
        if isinstance(v, (str, int, float, bool, type(None))):
            continue
        elif isinstance(v, Path):
            processed_args[k] = Config(Path, str(v))
        elif isinstance(v, (list, tuple)):
            processed_args[k] = [
                process_args((), {"item": item}, lambda item: None, get_fn)["item"] for item in v
            ]
        elif isinstance(v, dict):
            processed_args[k] = {
                key: process_args((), {"value": value}, lambda value: None, get_fn)["value"]
                for key, value in v.items()
            }
        elif (
            callable(v)
            or isinstance(v, type)
            or (isinstance(v, set) and all(isinstance(item, type) for item in v))
        ):
            continue
        else:
            try:
                processed_args[k] = get_fn(v)
            except Exception as e:
                raise ValueError(
                    f"Unable to convert object of type {type(v)} for argument '{k}'. "
                    f"Consider using the @capture decorator or capture() context manager "
                    f"to capture the instantiation of this object. Error: {str(e)}"
                ) from e

    return processed_args


def wrap_init(frame: FrameType, get_fn: Callable, register_fn: Callable, cls_to_ignore: Set[Type]):
    cls = frame.f_locals.get("self").__class__
    if cls not in cls_to_ignore:
        args = inspect.getargvalues(frame)
        processed_args = process_args(
            args.args[1:],
            {k: v for k, v in args.locals.items() if k != "self"},
            cls.__init__,
            get_fn,
        )
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
