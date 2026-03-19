"""Protocol interfaces for dependency inversion."""

from gesture_control.interfaces.activation_controller import IActivationController
from gesture_control.interfaces.command_dispatcher import ICommandDispatcher
from gesture_control.interfaces.feature_transformer import IFeatureTransformer
from gesture_control.interfaces.feedback_renderer import IFeedbackRenderer
from gesture_control.interfaces.frame_source import IFrameSource
from gesture_control.interfaces.gesture_model import IGestureModel
from gesture_control.interfaces.landmark_extractor import IHandLandmarkExtractor
from gesture_control.interfaces.prediction_stabilizer import IPredictionStabilizer
from gesture_control.interfaces.sequence_buffer import ISequenceBuffer

__all__ = [
    "IActivationController",
    "ICommandDispatcher",
    "IFeatureTransformer",
    "IFrameSource",
    "IGestureModel",
    "IHandLandmarkExtractor",
    "IPredictionStabilizer",
    "IFeedbackRenderer",
    "ISequenceBuffer",
]
