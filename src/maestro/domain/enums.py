from enum import StrEnum


class RuntimeMode(StrEnum):
    IDLE = "IDLE"
    ARMED = "ARMED"
    COOLDOWN = "COOLDOWN"
