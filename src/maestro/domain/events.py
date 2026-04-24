from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class DomainEvent:
    name: str
    timestamp_ms: int
    payload: dict[str, object]
