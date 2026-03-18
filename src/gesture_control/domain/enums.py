from enum import StrEnum


class RuntimeMode(StrEnum):
    IDLE = "IDLE"
    ARMED = "ARMED"
    COOLDOWN = "COOLDOWN"


class CommandID(StrEnum):
    NEXT_SLIDE = "next_slide"
    PREVIOUS_SLIDE = "previous_slide"
    MUTE_TOGGLE = "mute_toggle"
    RAISE_HAND = "raise_hand"
    START_PRESENTATION = "start_presentation"
    TOGGLE_PRESENTER_TOOLS = "toggle_presenter_tools"
