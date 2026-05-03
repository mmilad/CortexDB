from typing import Any


class RegistryState:
    """In-memory registry; replace with a database-backed implementation later."""

    def __init__(self) -> None:
        self.datasets: dict[str, dict[str, Any]] = {}
        self.tools: dict[str, dict[str, Any]] = {}


_registry = RegistryState()


def get_registry() -> RegistryState:
    return _registry
