"""
Gesture Semantic Mappings for HaGRID Dataset

This file contains text descriptions and semantic meanings for each gesture class
in the HaGRID dataset. These descriptions will be used to train the contrastive
learning model for semantic gesture understanding and zero-shot learning.

Each gesture is mapped to multiple text descriptions that capture:
1. Physical appearance (what the gesture looks like)
2. Semantic meaning (what the gesture means)
3. Action implications (what the gesture could trigger)
4. Cultural variations (different interpretations)

Author: Neil Taison Rigaud
Date: December 12, 2025
"""

# HaGRID 18-class gesture semantic mappings
GESTURE_SEMANTICS = {
    "call": {
        "descriptions": [
            "phone gesture",
            "call me",
            "make a phone call",
            "telephone sign",
            "talk on the phone",
            "dial a number",
            "answer the phone",
            "communication gesture",
            "hold phone to ear",
            "calling someone",
            "phone call hand sign",
            "telephone gesture",
            "ring me up",
            "give me a ring",
            "contact me by phone",
            "cellular call gesture",
            "mobile phone sign",
            "speaking on telephone",
            "handheld phone posture",
            "simulate phone conversation"
        ],
        "actions": [
            "initiate phone call",
            "answer incoming call",
            "open phone app",
            "voice command mode",
            "start voice call",
            "dial contact",
            "begin telecommunication",
            "activate phone interface",
            "launch dialer application",
            "respond to call"
        ],
        "semantic_group": "communication"
    },
    
    "dislike": {
        "descriptions": [
            "thumbs down",
            "disapprove",
            "I dislike this",
            "negative gesture",
            "bad",
            "no",
            "reject",
            "vote down",
            "downvote",
            "not good",
            "thumb pointing downward",
            "disapproval sign",
            "negative response",
            "rejection gesture",
            "disagree with this",
            "do not approve",
            "this is unacceptable",
            "poor quality indicator",
            "express dissatisfaction",
            "unfavorable rating",
            "decline this option",
            "give negative review",
            "show displeasure"
        ],
        "actions": [
            "decrease volume",
            "reject suggestion",
            "dislike content",
            "negative feedback",
            "scroll down",
            "minimize window",
            "reduce brightness",
            "lower setting",
            "move down in list",
            "dismiss notification",
            "rate negatively",
            "decrease value"
        ],
        "semantic_group": "negative_feedback"
    },
    
    "fist": {
        "descriptions": [
            "closed fist",
            "solidarity",
            "strength",
            "power",
            "determination",
            "resistance",
            "unity",
            "protest gesture",
            "clenched hand",
            "fist bump ready",
            "show strength",
            "tightly closed hand",
            "power gesture",
            "demonstrate resolve",
            "express determination",
            "solidarity sign",
            "united front",
            "firm decision",
            "hold tight",
            "grip closed",
            "assertive posture"
        ],
        "actions": [
            "confirm action",
            "select item",
            "grab object",
            "activate mode",
            "close hand",
            "execute command firmly",
            "hold position",
            "grip control",
            "lock selection",
            "finalize choice"
        ],
        "semantic_group": "confirmation"
    },
    
    "four": {
        "descriptions": [
            "number four",
            "four fingers",
            "fourth",
            "count to four",
            "four items",
            "quantity four",
            "show four",
            "quad digits",
            "fourth choice",
            "quartet of fingers",
            "four units",
            "quaternary option",
            "quadruple count",
            "four-way selection",
            "tetrad gesture"
        ],
        "actions": [
            "select fourth option",
            "jump to page four",
            "set value to four",
            "display four items",
            "fourth choice",
            "quad selection",
            "choose quartet"
        ],
        "semantic_group": "numeric"
    },
    
    "like": {
        "descriptions": [
            "thumbs up",
            "approve",
            "I like this",
            "positive gesture",
            "good",
            "yes",
            "agreement",
            "upvote",
            "vote up",
            "excellent",
            "great job",
            "well done",
            "I agree",
            "confirmed",
            "thumb pointing upward",
            "approval sign",
            "positive response",
            "acceptance gesture",
            "express satisfaction",
            "favorable rating",
            "this is great",
            "show appreciation",
            "endorse this",
            "support this idea",
            "give positive feedback",
            "recommend this",
            "quality content",
            "affirmative signal",
            "express agreement"
        ],
        "actions": [
            "increase volume",
            "approve selection",
            "like content",
            "positive feedback",
            "scroll up",
            "maximize window",
            "thumbs up reaction",
            "boost signal",
            "raise brightness",
            "increase setting",
            "move up in list",
            "promote content",
            "rate positively",
            "increment value"
        ],
        "semantic_group": "positive_feedback"
    },
    
    "mute": {
        "descriptions": [
            "shush gesture",
            "be quiet",
            "silence",
            "mute audio",
            "no sound",
            "quiet please",
            "finger to lips",
            "ssh",
            "hush",
            "silence gesture",
            "quiet down",
            "keep silent",
            "lower your voice",
            "stop making noise",
            "whisper sign",
            "need silence",
            "mute your microphone",
            "turn off audio",
            "disable sound output",
            "silence is needed",
            "reduce noise",
            "quiet mode activated"
        ],
        "actions": [
            "mute audio",
            "disable microphone",
            "silence notifications",
            "pause audio",
            "turn off sound",
            "deactivate speakers",
            "stop audio playback",
            "suppress volume",
            "cut audio feed",
            "enable silent mode",
            "quiet system sounds"
        ],
        "semantic_group": "audio_control"
    },
    
    "ok": {
        "descriptions": [
            "okay sign",
            "everything is fine",
            "perfect",
            "all good",
            "confirmed",
            "OK gesture",
            "approval",
            "circle with fingers",
            "zero sign",
            "it's fine",
            "acknowledged",
            "fingers forming circle",
            "acceptance sign",
            "things are perfect",
            "all systems go",
            "ready to proceed",
            "affirmative response",
            "situation is good",
            "status okay",
            "confirm understanding",
            "signal approval",
            "everything checks out",
            "satisfactory result",
            "acceptable condition"
        ],
        "actions": [
            "confirm selection",
            "approve action",
            "acknowledge message",
            "everything OK",
            "proceed with action",
            "validate input",
            "accept proposal",
            "green light command",
            "authorize process",
            "finalize decision",
            "complete transaction"
        ],
        "semantic_group": "confirmation"
    },
    
    "one": {
        "descriptions": [
            "number one",
            "single finger",
            "first",
            "index finger",
            "point up",
            "count to one",
            "one item",
            "quantity one",
            "show one",
            "one finger raised",
            "single digit",
            "first place",
            "primary option",
            "solo finger",
            "singular selection",
            "number uno",
            "we are number one",
            "top choice",
            "first priority",
            "one unit",
            "single count"
        ],
        "actions": [
            "select first option",
            "jump to page one",
            "set value to one",
            "display one item",
            "number one",
            "choose primary",
            "first selection",
            "initial choice",
            "top option"
        ],
        "semantic_group": "numeric"
    },
    
    "palm": {
        "descriptions": [
            "open hand",
            "stop",
            "show palm",
            "hand raised",
            "halt",
            "wait",
            "open palm",
            "hand forward",
            "stop sign",
            "hold on",
            "palm facing outward",
            "hand up gesture",
            "pause signal",
            "wait a moment",
            "hold position",
            "cease movement",
            "suspend activity",
            "give me a second",
            "hold that thought",
            "interrupt gesture",
            "attention please",
            "request to pause",
            "break in action"
        ],
        "actions": [
            "stop action",
            "pause playback",
            "halt process",
            "wait command",
            "show palm",
            "open hand gesture",
            "suspend operation",
            "freeze frame",
            "temporary stop",
            "interrupt sequence",
            "hold execution"
        ],
        "semantic_group": "stop_control"
    },
    
    "peace": {
        "descriptions": [
            "peace sign",
            "victory",
            "two fingers up",
            "V sign",
            "peace gesture",
            "victory sign",
            "V for victory",
            "two finger salute",
            "peace symbol with hand",
            "hippie gesture",
            "tranquility sign",
            "harmony gesture",
            "victory celebration",
            "win indicator",
            "peaceful expression",
            "serenity signal",
            "success sign",
            "triumph gesture"
        ],
        "actions": [
            "select second option",
            "peace gesture",
            "victory celebration",
            "two items",
            "number two",
            "celebrate win",
            "show peace",
            "express harmony"
        ],
        "semantic_group": "symbolic"
    },
    
    "peace_inverted": {
        "descriptions": [
            "inverted peace sign",
            "reverse V sign",
            "two fingers down",
            "British V sign",
            "inverted victory",
            "upside down peace"
        ],
        "actions": [
            "alternative peace sign",
            "inverted selection",
            "reverse command"
        ],
        "semantic_group": "symbolic"
    },
    
    "rock": {
        "descriptions": [
            "rock and roll",
            "horns gesture",
            "metal sign",
            "rock on",
            "devil horns",
            "rock hand",
            "heavy metal",
            "horn fingers",
            "rock sign",
            "metal horns",
            "rock music gesture",
            "headbanger sign",
            "mosh pit gesture",
            "concert salute",
            "rock concert sign",
            "metalhead greeting",
            "horns up",
            "rock symbol",
            "metal culture sign"
        ],
        "actions": [
            "rock gesture",
            "celebration",
            "excitement",
            "music playback",
            "play rock music",
            "activate party mode",
            "energize system",
            "show enthusiasm"
        ],
        "semantic_group": "symbolic"
    },
    
    "stop": {
        "descriptions": [
            "stop sign",
            "halt",
            "don't proceed",
            "wait",
            "stop hand",
            "stop gesture",
            "stop action",
            "cease",
            "end",
            "terminate",
            "halt immediately",
            "stop right now",
            "cease all activity",
            "discontinue process",
            "abort operation",
            "end this action",
            "do not continue",
            "full stop",
            "complete cessation",
            "terminate execution",
            "cancel everything",
            "emergency halt",
            "immediate termination"
        ],
        "actions": [
            "stop playback",
            "halt action",
            "cancel operation",
            "stop scrolling",
            "emergency stop",
            "abort process",
            "terminate program",
            "cease activity",
            "end session",
            "kill process",
            "force quit"
        ],
        "semantic_group": "stop_control"
    },
    
    "stop_inverted": {
        "descriptions": [
            "inverted stop sign",
            "upside down stop",
            "reverse stop",
            "palm up stop",
            "stop hand inverted"
        ],
        "actions": [
            "alternative stop",
            "pause with palm up",
            "reverse halt"
        ],
        "semantic_group": "stop_control"
    },
    
    "three": {
        "descriptions": [
            "number three",
            "three fingers",
            "third",
            "count to three",
            "three items",
            "quantity three",
            "show three",
            "three finger salute",
            "triple digits",
            "third choice",
            "trio of fingers",
            "three units",
            "tertiary option",
            "threesome gesture",
            "triple count",
            "three-way selection"
        ],
        "actions": [
            "select third option",
            "jump to page three",
            "set value to three",
            "display three items",
            "third choice",
            "triple selection",
            "choose trilogy"
        ],
        "semantic_group": "numeric"
    },
    
    "three2": {
        "descriptions": [
            "alternative three",
            "three fingers variant",
            "different three gesture",
            "three with thumb",
            "three fingers alternative"
        ],
        "actions": [
            "select third option alternative",
            "three variant",
            "alternative three gesture"
        ],
        "semantic_group": "numeric"
    },
    
    "two_up": {
        "descriptions": [
            "number two",
            "two fingers",
            "second",
            "count to two",
            "two items",
            "quantity two",
            "show two",
            "two fingers up",
            "dual digits",
            "second choice",
            "pair of fingers",
            "double count",
            "twin fingers",
            "two units",
            "secondary option",
            "deuce sign",
            "couple gesture",
            "both options"
        ],
        "actions": [
            "select second option",
            "jump to page two",
            "set value to two",
            "display two items",
            "second choice",
            "dual selection",
            "choose pair",
            "two-item display"
        ],
        "semantic_group": "numeric"
    },
    
    "two_up_inverted": {
        "descriptions": [
            "inverted two",
            "two fingers inverted",
            "upside down two",
            "reverse two fingers",
            "two fingers down"
        ],
        "actions": [
            "alternative two",
            "inverted second option",
            "reverse two gesture"
        ],
        "semantic_group": "numeric"
    },
    
    # Additional classes that may be in HaGRID
    "grabbing": {
        "descriptions": [
            "grabbing gesture",
            "reach and grab",
            "hand closing",
            "grasping",
            "catch object",
            "hold item",
            "grip motion"
        ],
        "actions": [
            "grab object",
            "drag item",
            "start drag operation",
            "hold and move"
        ],
        "semantic_group": "manipulation"
    },
    
    "point": {
        "descriptions": [
            "pointing",
            "indicate direction",
            "show direction",
            "point finger",
            "index finger pointing",
            "direct attention",
            "that way",
            "look there",
            "finger indicating",
            "select this object",
            "focus here",
            "aim at target",
            "guide attention",
            "show location",
            "refer to this",
            "designate item",
            "direct gaze",
            "highlight position",
            "indicate choice",
            "choose this one",
            "target selection",
            "pointer gesture"
        ],
        "actions": [
            "select item",
            "indicate direction",
            "cursor control",
            "point at object",
            "mouse click",
            "choose option",
            "focus element",
            "highlight item",
            "navigate to location",
            "select target",
            "activate pointed item"
        ],
        "semantic_group": "navigation"
    },
    
    "no_gesture": {
        "descriptions": [
            "no hand detected",
            "empty frame",
            "no gesture",
            "idle state",
            "rest position",
            "neutral state",
            "no action"
        ],
        "actions": [
            "idle",
            "no action",
            "wait",
            "neutral"
        ],
        "semantic_group": "idle"
    }
}

# Semantic group to action mapping (for extensibility)
SEMANTIC_GROUP_ACTIONS = {
    "positive_feedback": [
        "increase volume", "scroll up", "like content", "approve", 
        "upvote", "raise value", "brighten", "maximize"
    ],
    "negative_feedback": [
        "decrease volume", "scroll down", "dislike content", "reject",
        "downvote", "lower value", "dim", "minimize"
    ],
    "stop_control": [
        "stop playback", "pause", "halt", "freeze", "stop scrolling",
        "cancel", "emergency stop"
    ],
    "confirmation": [
        "confirm selection", "approve action", "acknowledge", "proceed",
        "submit", "accept", "yes", "agree"
    ],
    "numeric": [
        "select option 1", "select option 2", "select option 3", "select option 4",
        "jump to page", "set value", "count", "enumerate"
    ],
    "navigation": [
        "point at", "select", "move cursor", "indicate", "direct",
        "focus", "highlight"
    ],
    "communication": [
        "call", "answer", "dial", "voice mode", "talk", "communicate"
    ],
    "audio_control": [
        "mute", "unmute", "silence", "volume control", "audio off", "audio on"
    ],
    "manipulation": [
        "grab", "drag", "move", "hold", "release", "drop", "pick up"
    ],
    "symbolic": [
        "celebrate", "peace", "victory", "rock on", "express emotion"
    ],
    "idle": [
        "no action", "wait", "idle", "neutral", "rest"
    ]
}

# Zero-shot test commands (unseen during training)
ZERO_SHOT_TEST_COMMANDS = {
    # Should map to "like" gesture
    "positive_feedback": [
        "I appreciate this",
        "this is great",
        "wonderful work",
        "increase the volume",
        "turn it up",
        "make it louder",
        "I agree with this",
        "yes please",
        "boost the audio",
        "raise the level",
        "approve this content",
        "give positive rating",
        "this is excellent",
        "show appreciation",
        "endorse this item",
        "support this decision",
        "favorable response"
    ],
    
    # Should map to "dislike" gesture
    "negative_feedback": [
        "I don't like this",
        "this is bad",
        "decrease the volume",
        "turn it down",
        "make it quieter",
        "I disagree",
        "no thanks",
        "lower the audio",
        "reduce the level",
        "reject this content",
        "unfavorable rating",
        "this is poor quality",
        "express dissatisfaction",
        "decline this option",
        "disapprove this choice"
    ],
    
    # Should map to "stop" gesture
    "stop_action": [
        "stop the music",
        "halt the process",
        "pause everything",
        "stop scrolling",
        "freeze the screen",
        "cancel this action",
        "terminate execution",
        "abort operation",
        "cease activity",
        "end this immediately",
        "kill the process",
        "emergency stop",
        "discontinue playback"
    ],
    
    # Should map to "ok" gesture
    "confirm_action": [
        "confirm my selection",
        "proceed with this",
        "everything looks good",
        "approved",
        "go ahead",
        "I accept",
        "validate this input",
        "authorize action",
        "all systems are go",
        "affirmative response",
        "ready to continue",
        "things check out",
        "finalize decision"
    ],
    
    # Should map to "point" gesture
    "selection": [
        "select this item",
        "click here",
        "choose this option",
        "I want that one",
        "focus on this",
        "indicate this target",
        "highlight this element",
        "pick this choice",
        "designate this object",
        "direct attention here",
        "show me that"
    ],
    
    # Should map to "mute" gesture
    "audio_control": [
        "turn off the sound",
        "silence the audio",
        "disable microphone",
        "quiet mode",
        "no sound please",
        "cut the volume",
        "suppress audio",
        "deactivate speakers",
        "enable silent mode",
        "mute all sounds"
    ],
    
    # Should map to "call" gesture
    "communication": [
        "make a phone call",
        "dial this number",
        "start voice chat",
        "answer the call",
        "open communication",
        "initiate phone conversation",
        "ring someone up",
        "place a call",
        "contact via phone",
        "begin telecommunication"
    ],
    
    # Should map to "palm" or "stop"
    "pause_request": [
        "hold on a moment",
        "wait a second",
        "give me time",
        "pause for now",
        "hold that thought",
        "interrupt briefly",
        "temporary break",
        "suspend activity"
    ],
    
    # Should map to "fist"
    "strength_action": [
        "show determination",
        "demonstrate strength",
        "express solidarity",
        "confirm firmly",
        "hold tight",
        "grip control",
        "assert position"
    ],
    
    # Should map to "peace"
    "victory_celebration": [
        "celebrate victory",
        "show peace",
        "express harmony",
        "win gesture",
        "peaceful sign",
        "success indicator"
    ],
    
    # Should map to "rock"
    "excitement_music": [
        "rock and roll",
        "party time",
        "music excitement",
        "show enthusiasm",
        "concert energy",
        "metal music"
    ]
}

# Helper function to get all text descriptions for a gesture
def get_gesture_descriptions(gesture_class):
    """
    Get all text descriptions for a given gesture class.
    
    Args:
        gesture_class (str): Gesture class name (e.g., 'like', 'dislike')
        
    Returns:
        list: All text descriptions for the gesture
    """
    if gesture_class not in GESTURE_SEMANTICS:
        return []
    return GESTURE_SEMANTICS[gesture_class]["descriptions"]

# Helper function to get action descriptions for a gesture
def get_gesture_actions(gesture_class):
    """
    Get all action descriptions for a given gesture class.
    
    Args:
        gesture_class (str): Gesture class name (e.g., 'like', 'dislike')
        
    Returns:
        list: All action descriptions for the gesture
    """
    if gesture_class not in GESTURE_SEMANTICS:
        return []
    return GESTURE_SEMANTICS[gesture_class]["actions"]

# Helper function to get all text for a gesture (descriptions + actions)
def get_all_gesture_text(gesture_class):
    """
    Get all text (descriptions + actions) for a given gesture class.
    
    Args:
        gesture_class (str): Gesture class name (e.g., 'like', 'dislike')
        
    Returns:
        list: All text descriptions and actions for the gesture
    """
    if gesture_class not in GESTURE_SEMANTICS:
        return []
    return (GESTURE_SEMANTICS[gesture_class]["descriptions"] + 
            GESTURE_SEMANTICS[gesture_class]["actions"])

# Helper function to get semantic group
def get_semantic_group(gesture_class):
    """
    Get the semantic group for a given gesture class.
    
    Args:
        gesture_class (str): Gesture class name (e.g., 'like', 'dislike')
        
    Returns:
        str: Semantic group name
    """
    if gesture_class not in GESTURE_SEMANTICS:
        return "unknown"
    return GESTURE_SEMANTICS[gesture_class]["semantic_group"]

# Statistics
def print_dataset_stats():
    """Print statistics about the gesture semantic mappings."""
    print("=" * 70)
    print("GESTURE SEMANTIC MAPPINGS - STATISTICS")
    print("=" * 70)
    print(f"Total gestures: {len(GESTURE_SEMANTICS)}")
    
    total_descriptions = sum(len(v["descriptions"]) for v in GESTURE_SEMANTICS.values())
    total_actions = sum(len(v["actions"]) for v in GESTURE_SEMANTICS.values())
    
    print(f"Total descriptions: {total_descriptions}")
    print(f"Total actions: {total_actions}")
    print(f"Total text samples: {total_descriptions + total_actions}")
    print(f"Average descriptions per gesture: {total_descriptions / len(GESTURE_SEMANTICS):.1f}")
    print(f"Average actions per gesture: {total_actions / len(GESTURE_SEMANTICS):.1f}")
    
    # Group statistics
    groups = {}
    for gesture_data in GESTURE_SEMANTICS.values():
        group = gesture_data["semantic_group"]
        groups[group] = groups.get(group, 0) + 1
    
    print(f"\nSemantic groups: {len(groups)}")
    for group, count in sorted(groups.items(), key=lambda x: x[1], reverse=True):
        print(f"  {group}: {count} gestures")
    
    print(f"\nZero-shot test commands: {len(ZERO_SHOT_TEST_COMMANDS)} categories")
    total_zero_shot = sum(len(v) for v in ZERO_SHOT_TEST_COMMANDS.values())
    print(f"Total zero-shot test samples: {total_zero_shot}")
    print("=" * 70)

if __name__ == "__main__":
    print_dataset_stats()
    
    # Example usage
    print("\nExample: 'like' gesture")
    print("-" * 70)
    print("Descriptions:", get_gesture_descriptions("like"))
    print("Actions:", get_gesture_actions("like"))
    print("Semantic group:", get_semantic_group("like"))
