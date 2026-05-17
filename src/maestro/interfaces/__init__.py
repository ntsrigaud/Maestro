"""Protocol interfaces for dependency inversion."""

from maestro.interfaces.activation_controller import IActivationController
from maestro.interfaces.command_dispatcher import ICommandDispatcher
from maestro.interfaces.feature_transformer import IFeatureTransformer
from maestro.interfaces.feedback_renderer import IFeedbackRenderer
from maestro.interfaces.frame_source import IFrameSource
from maestro.interfaces.gesture_model import IGestureModel
from maestro.interfaces.landmark_extractor import IHandLandmarkExtractor
from maestro.interfaces.prediction_stabilizer import IPredictionStabilizer
from maestro.interfaces.sequence_buffer import ISequenceBuffer

__all__ = [
    "IActivationController",
    "ICommandDispatcher",
    "IFeatureTransformer",
    "IFeedbackRenderer",
    "IFrameSource",
    "IGestureModel",
    "IHandLandmarkExtractor",
    "IPredictionStabilizer",
    "ISequenceBuffer",
]
