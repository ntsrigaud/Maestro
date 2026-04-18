from enum import StrEnum


class RuntimeMode(StrEnum):
    IDLE = "IDLE"
    ARMED = "ARMED"
    COOLDOWN = "COOLDOWN"


class CommandID(StrEnum):
    NEXT_SLIDE = "next_slide"
    PREVIOUS_SLIDE = "previous_slide"
    RAISE_HAND = "raise_hand"
    START_PRESENTATION = "start_presentation"
    TOGGLE_POINTER_MODE = "toggle_pointer_mode"
    EXIT_PRESENTATION = "exit_presentation"
