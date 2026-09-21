"""
Strict JSON serialization for scientific evidence.

Python's json.dumps() defaults to allow_nan=True, which silently emits
the non-standard tokens NaN/Infinity/-Infinity into "JSON" output - a
statistic that legitimately came out NaN (e.g. a degenerate zero-
variance computation) would otherwise be serialized without any
signal that something numerically degenerate happened. This module
makes that fail loudly and precisely instead.
"""

from __future__ import annotations

import json
import math
from typing import Any


class StrictJSONError(ValueError):
    pass


def find_non_finite_paths(obj: Any, path: str = "$") -> list[str]:
    """
    Walk a JSON-serializable structure and return the paths (JSONPath-
    like strings) of every float that is NaN, +Infinity, or -Infinity.
    Used to produce an informative error before attempting strict
    serialization, rather than a bare "Out of range float" exception.
    """
    problems: list[str] = []
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            problems.append(path)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            problems.extend(find_non_finite_paths(value, f"{path}.{key}"))
    elif isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            problems.extend(find_non_finite_paths(value, f"{path}[{i}]"))
    return problems


def dumps_strict(obj: Any, **kwargs: Any) -> str:
    """
    Serialize to JSON, refusing NaN/Infinity/-Infinity anywhere in the
    structure. Raises StrictJSONError naming every offending path
    (not just the first one json.dumps would otherwise choke on).
    """
    problems = find_non_finite_paths(obj)
    if problems:
        raise StrictJSONError(
            f"Refusing to serialize non-finite scientific value(s) at: {problems}. "
            "A NaN/Infinity value in evidence output must be investigated and "
            "explicitly handled (e.g. reported as a failure/limitation), not "
            "silently written to a certificate as if it were a normal number."
        )
    kwargs.setdefault("allow_nan", False)
    return json.dumps(obj, **kwargs)


def dump_strict(obj: Any, fp, **kwargs: Any) -> None:
    """File-writing counterpart to dumps_strict, with the same guarantee."""
    fp.write(dumps_strict(obj, **kwargs))
