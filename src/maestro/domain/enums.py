from enum import StrEnum


class RuntimeMode(StrEnum):
    IDLE = "idle"
    ARMED = "armed"
    COOLDOWN = "cooldown"
