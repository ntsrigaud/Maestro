from enum import Enum

try:
    from enum import StrEnum
except ImportError:  # Python < 3.11

    class StrEnum(str, Enum):
        pass


class RuntimeMode(StrEnum):
    ARMED = "armed"
    COOLDOWN = "cooldown"
