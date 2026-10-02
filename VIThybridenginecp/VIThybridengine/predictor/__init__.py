"""Speed prediction neural network package for hybrid vehicle EMS."""
from predictor.speed_predictor import (
    train_model,
    export_tflite_quantized,
    generate_presentation_summary,
    generate_training_dataset,
    build_mlp_model,
    build_cnn_model,
    LOOKBACK_SECONDS,
    HORIZON_SECONDS,
    SPEED_NORMALIZATION,
)

__all__ = [
    "train_model",
    "export_tflite_quantized",
    "generate_presentation_summary",
    "generate_training_dataset",
    "build_mlp_model",
    "build_cnn_model",
    "LOOKBACK_SECONDS",
    "HORIZON_SECONDS",
    "SPEED_NORMALIZATION",
]
