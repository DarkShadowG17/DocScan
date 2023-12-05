from typing import Dict, TypeVar

K = TypeVar("K")
V = TypeVar("V")


def intersect_dict(a: Dict[K, V], b: Dict[K, V]) -> Dict[K, V]:
    """Intersects a dictionary by key."""
    return {k: a[k] for k in a if k in b}
