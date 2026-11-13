"""
A record of everything reachable from an object, to check that a call left it unchanged.
"""

from types import BuiltinFunctionType, FunctionType, MethodType, ModuleType
from typing import Dict

import torch
from torch import Tensor

_VALUES = (type(None), bool, int, float, complex, str, bytes, torch.dtype, torch.device)
#: kept by identity only: their contents are code, not state
_OPAQUE = (type, ModuleType, FunctionType, BuiltinFunctionType, MethodType)


class _Same:
    """Equal only to a record of the same object. Holds a reference, so the id cannot be reused."""

    def __init__(self, obj: object):
        self.obj = obj

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Same) and other.obj is self.obj

    def __repr__(self) -> str:
        return f"<{type(self.obj).__name__} at {id(self.obj):#x}>"


def snapshot(**roots: object) -> Dict[str, object]:
    """
    Maps the path of every object reachable from the ``roots`` through attributes, dicts, lists and
    tuples to its identity, and the path of every plain value to the value. Tensors also record their
    ``requires_grad``, their ``.grad`` and the bytes of their values.
    Two snapshots compare equal if nothing reachable was replaced, added, removed or changed in
    place, and pytest's diff then names the paths that differ.
    """
    out: Dict[str, object] = {}
    seen: Dict[int, str] = {}
    for name, root in roots.items():
        _walk(root, name, out, seen)
    return out


def _walk(obj: object, path: str, out: Dict[str, object], seen: Dict[int, str]) -> None:
    if isinstance(obj, _VALUES):
        out[path] = obj
        return
    if id(obj) in seen:  # shared references and cycles
        out[path] = f"same as {seen[id(obj)]!r}"
        return
    seen[id(obj)] = path
    out[path] = _Same(obj)

    if isinstance(obj, Tensor):
        data = obj.detach().cpu().contiguous().reshape(-1)
        out[f"{path}.<values>"] = (
            obj.dtype,
            tuple(obj.shape),
            data.view(torch.uint8).numpy().tobytes(),
        )
        out[f"{path}.requires_grad"] = obj.requires_grad
        _walk(obj.grad if obj.is_leaf else None, f"{path}.grad", out, seen)
    if isinstance(obj, dict):
        for key, value in obj.items():
            _walk(value, f"{path}[{key!r}]", out, seen)
    elif isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            _walk(value, f"{path}[{i}]", out, seen)
    elif isinstance(obj, (set, frozenset)):
        out[f"{path}.<items>"] = frozenset(obj)
    elif hasattr(obj, "__dict__") and not isinstance(obj, _OPAQUE):
        for key, value in vars(obj).items():
            _walk(value, f"{path}.{key}", out, seen)
