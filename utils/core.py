from typing import Any


def exists(x: Any) -> bool:
    return x is not None


def default(x: Any, default: Any) -> Any:
    return default if not exists(x) else x
