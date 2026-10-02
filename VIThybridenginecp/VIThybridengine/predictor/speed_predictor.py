"""
Predictive Speed Forecasting Neural Network Module.

Builds, trains, and exports a lightweight speed prediction model that forecasts
the next 15 seconds of vehicle speed given the past 10 seconds.

Architecture Options:
  1. MLP  — Multi-Layer Perceptron (default, smallest footprint)
  2. CNN  — 1D Convolutional Neural Network (better temporal feature extraction)

Designed for deployment on automotive-grade microcontrollers via quantized TFLite export.

Input  : Past 10 seconds of vehicle speed  (shape: [batch, 10, 1])
Output : Next 15 seconds of vehicle speed   (shape: [batch, 15])
Horizon: 10-30 s prediction window (15 s satisfies the constraint)
"""

import os
import json
import datetime
from typing import Tuple, Optional, Dict, Any

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TF INFO/WARNING logs
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
LOOKBACK_SECONDS = 10     # Past window fed as input
HORIZON_SECONDS  = 15     # Future window to predict  (within 10-30 s constraint)
SPEED_NORMALIZATION = 140.0  # Max plausible speed in km/h for [0, 1] normalization

MODEL_TYPES = {"mlp", "cnn"}


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Synthetic WLTP Training Dataset Generator
# ──────────────────────────────────────────────────────────────────────────────

def _generate_single_wltp_variant(
    rng: np.random.Generator,
    base_duration: int = 1800,
) -> np.ndarray:
    """
    Creates one synthetic WLTP-like speed profile (km/h, 1 Hz) with
    controlled randomisation so the network learns generalised dynamics.
    """
    # Characteristic speed nodes for each WLTP-class-3 phase
    # Each tuple: (target_speed_kmh, hold_duration_range)
    phase_templates = [
        # Low phase – urban stop-and-go
        [(0, (8, 20)), (18, (6, 14)), (0, (8, 16)), (32, (10, 18)),
         (16, (6, 12)), (35, (8, 16)), (0, (10, 20)), (45, (10, 20)),
         (56.5, (12, 22)), (38, (8, 16)), (20, (6, 12)), (0, (14, 26)),
         (25, (8, 14)), (0, (10, 18)), (48, (12, 20)), (15, (6, 10)),
         (0, (10, 22))],
        # Medium phase – suburban
        [(25, (8, 14)), (45, (10, 18)), (0, (8, 16)), (55, (12, 20)),
         (28, (6, 12)), (62, (14, 22)), (0, (10, 18)), (68, (14, 24)),
         (50, (8, 16)), (76.6, (16, 28)), (45, (10, 18)), (20, (6, 12)),
         (0, (10, 20))],
        # High phase – rural
        [(35, (8, 14)), (60, (12, 20)), (80, (16, 26)), (55, (8, 16)),
         (88, (18, 30)), (70, (10, 18)), (97.4, (20, 34)), (82, (14, 24)),
         (65, (10, 18)), (30, (6, 12)), (0, (8, 16))],
        # Extra-high phase – motorway
        [(45, (8, 14)), (80, (10, 18)), (110, (16, 28)), (95, (10, 20)),
         (125, (20, 36)), (131.3, (24, 42)), (115, (16, 28)),
         (90, (12, 22)), (50, (8, 16)), (0, (6, 14))],
    ]

    speed_profile = []
    current_v = 0.0

    for phase_nodes in phase_templates:
        for target_v, (dur_lo, dur_hi) in phase_nodes:
            # Apply small random perturbation to target speed (+/- 8 %)
            perturbed_v = max(0.0, target_v * rng.uniform(0.92, 1.08))
            # Random transition + hold duration
            total_dur = int(rng.integers(dur_lo, dur_hi + 1))
            trans_dur = max(2, int(total_dur * rng.uniform(0.35, 0.55)))
            hold_dur = max(1, total_dur - trans_dur)

            # Smooth cosine transition
            t_arr = np.linspace(0, np.pi, trans_dur)
            transition = current_v + 0.5 * (perturbed_v - current_v) * (1.0 - np.cos(t_arr))
            speed_profile.extend(transition.tolist())

            # Constant-speed hold with micro-noise
            hold = np.full(hold_dur, perturbed_v) + rng.normal(0, 0.3, hold_dur)
            speed_profile.extend(np.clip(hold, 0.0, 140.0).tolist())

            current_v = perturbed_v

    # Resample to exactly base_duration seconds + 1 point
    raw = np.array(speed_profile, dtype=np.float64)
    x_old = np.linspace(0, 1, len(raw))
    x_new = np.linspace(0, 1, base_duration + 1)
    resampled = np.interp(x_new, x_old, raw)
    return np.clip(resampled, 0.0, 140.0)


def generate_training_dataset(
    num_cycles: int = 80,
    lookback: int = LOOKBACK_SECONDS,
    horizon: int = HORIZON_SECONDS,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a windowed (X, Y) dataset from multiple synthetic WLTP drive-cycle
    variants for supervised training of the speed predictor.

    Returns
    -------
    X_train, Y_train, X_val, Y_val
        Normalised arrays ready for model.fit().
        X shape: (N, lookback, 1)
        Y shape: (N, horizon)
    """
    rng = np.random.default_rng(seed)
    all_X, all_Y = [], []

    for i in range(num_cycles):
        cycle = _generate_single_wltp_variant(rng)
        # Normalise to [0, 1]
        cycle_norm = cycle / SPEED_NORMALIZATION

        window_size = lookback + horizon
        n_windows = len(cycle_norm) - window_size
        if n_windows <= 0:
            continue

        for j in range(n_windows):
            window = cycle_norm[j : j + window_size]
            x = window[:lookback].reshape(lookback, 1)
            y = window[lookback:]
            all_X.append(x)
            all_Y.append(y)

    X = np.array(all_X, dtype=np.float32)
    Y = np.array(all_Y, dtype=np.float32)

    # Shuffle deterministically
    idx = rng.permutation(len(X))
    X, Y = X[idx], Y[idx]

    # 85 / 15 split
    split = int(0.85 * len(X))
    return X[:split], Y[:split], X[split:], Y[split:]


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Model Architecture Builders
# ──────────────────────────────────────────────────────────────────────────────

def build_mlp_model(
    lookback: int = LOOKBACK_SECONDS,
    horizon: int = HORIZON_SECONDS,
) -> keras.Model:
    """
    Lightweight Multi-Layer Perceptron for speed prediction.
    ~5 K parameters — feasible on Cortex-M / Renesas RH850 class MCUs.
    """
    inp = layers.Input(shape=(lookback, 1), name="speed_history")
    x = layers.Flatten()(inp)
    x = layers.Dense(64, activation="relu", name="hidden_1")(x)
    x = layers.Dropout(0.15)(x)
    x = layers.Dense(48, activation="relu", name="hidden_2")(x)
    x = layers.Dropout(0.10)(x)
    x = layers.Dense(32, activation="relu", name="hidden_3")(x)
    out = layers.Dense(horizon, activation="sigmoid", name="speed_forecast")(x)

    model = keras.Model(inputs=inp, outputs=out, name="SpeedPredictor_MLP")
    return model


def build_cnn_model(
    lookback: int = LOOKBACK_SECONDS,
    horizon: int = HORIZON_SECONDS,
) -> keras.Model:
    """
    Lightweight 1D-CNN for speed prediction.
    Temporal convolutions capture local acceleration / deceleration patterns.
    ~8 K parameters.
    """
    inp = layers.Input(shape=(lookback, 1), name="speed_history")

    x = layers.Conv1D(32, kernel_size=3, padding="causal", activation="relu", name="conv1")(inp)
    x = layers.Conv1D(32, kernel_size=3, padding="causal", activation="relu", name="conv2")(x)
    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(48, activation="relu", name="fc1")(x)
    x = layers.Dropout(0.15)(x)
    out = layers.Dense(horizon, activation="sigmoid", name="speed_forecast")(x)

    model = keras.Model(inputs=inp, outputs=out, name="SpeedPredictor_1DCNN")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Training Loop
# ──────────────────────────────────────────────────────────────────────────────

def train_model(
    model_type: str = "mlp",
    num_cycles: int = 80,
    epochs: int = 40,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    seed: int = 42,
    verbose: int = 1,
) -> Tuple[keras.Model, dict, Dict[str, Any]]:
    """
    End-to-end: generate data → build model → compile → train → return.

    Parameters
    ----------
    model_type : str
        'mlp' or 'cnn'.
    num_cycles : int
        Number of synthetic WLTP drive-cycle variants to generate for training.
    epochs : int
        Training epochs.
    batch_size : int
        Mini-batch size.
    learning_rate : float
        Adam optimiser learning rate.
    seed : int
        Random seed for reproducibility.
    verbose : int
        Keras verbosity level (0=silent, 1=progress bar, 2=one line per epoch).

    Returns
    -------
    model : keras.Model
        Trained Keras model.
    history : dict
        Training history dictionary.
    metadata : dict
        Summary metadata (architecture, data stats, performance).
    """
    if model_type not in MODEL_TYPES:
        raise ValueError(f"model_type must be one of {MODEL_TYPES}, got '{model_type}'")

    tf.random.set_seed(seed)
    np.random.seed(seed)

    # 1. Generate dataset
    print(f"[SpeedPredictor] Generating {num_cycles} synthetic WLTP cycle variants …")
    X_train, Y_train, X_val, Y_val = generate_training_dataset(
        num_cycles=num_cycles, seed=seed
    )
    print(f"[SpeedPredictor] Training samples: {len(X_train):,}  |  Validation samples: {len(X_val):,}")

    # 2. Build model
    if model_type == "mlp":
        model = build_mlp_model()
    else:
        model = build_cnn_model()

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )

    if verbose > 0:
        model.summary()

    # 3. Train
    cb_list = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    hist = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cb_list,
        verbose=verbose,
    )

    # 4. Evaluate
    val_loss, val_mae = model.evaluate(X_val, Y_val, verbose=0)
    # Convert normalised MAE back to km/h
    val_mae_kmh = val_mae * SPEED_NORMALIZATION

    metadata = {
        "model_name": model.name,
        "model_type": model_type.upper(),
        "architecture": "Multi-Layer Perceptron (MLP)" if model_type == "mlp" else "1D Convolutional Neural Network (1D-CNN)",
        "input_description": f"Past {LOOKBACK_SECONDS} seconds of vehicle speed (1 Hz sampling)",
        "input_shape": f"({LOOKBACK_SECONDS}, 1)",
        "output_description": f"Next {HORIZON_SECONDS} seconds of predicted vehicle speed",
        "output_shape": f"({HORIZON_SECONDS},)",
        "prediction_horizon_s": HORIZON_SECONDS,
        "lookback_window_s": LOOKBACK_SECONDS,
        "horizon_constraint": "10 – 30 s (satisfied: 15 s)",
        "total_parameters": int(model.count_params()),
        "training_data_source": f"{num_cycles} synthetic WLTP Class 3 drive-cycle variants (randomised speed perturbations)",
        "training_samples": int(len(X_train)),
        "validation_samples": int(len(X_val)),
        "final_val_loss_mse": float(val_loss),
        "final_val_mae_normalised": float(val_mae),
        "final_val_mae_kmh": float(val_mae_kmh),
        "epochs_trained": len(hist.history["loss"]),
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "speed_normalisation_factor": SPEED_NORMALIZATION,
        "timestamp": datetime.datetime.now().isoformat(),
    }

    print(f"\n[SpeedPredictor] Training complete.")
    print(f"  Model Type       : {metadata['architecture']}")
    print(f"  Total Parameters : {metadata['total_parameters']:,}")
    print(f"  Val MSE Loss     : {val_loss:.6f}")
    print(f"  Val MAE          : {val_mae_kmh:.2f} km/h")
    print(f"  Epochs Trained   : {metadata['epochs_trained']}")

    return model, hist.history, metadata


# ──────────────────────────────────────────────────────────────────────────────
# 4.  TFLite Export with INT8 Quantization
# ──────────────────────────────────────────────────────────────────────────────

def export_tflite_quantized(
    model: keras.Model,
    output_dir: str,
    metadata: Dict[str, Any],
    num_calibration_cycles: int = 10,
    seed: int = 42,
) -> str:
    """
    Exports the trained Keras model as a fully INT8-quantized TFLite file
    suitable for deployment on automotive microcontrollers (e.g. ARM Cortex-M,
    Renesas RH850, Infineon AURIX).

    Parameters
    ----------
    model : keras.Model
        Trained speed prediction model.
    output_dir : str
        Directory to write the .tflite and metadata JSON.
    metadata : dict
        Training metadata to persist alongside the model.
    num_calibration_cycles : int
        Number of WLTP variants used for INT8 calibration.
    seed : int
        Random seed for calibration data generation.

    Returns
    -------
    tflite_path : str
        Absolute path to the exported .tflite file.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Generate representative calibration dataset for INT8 quantization
    print("[TFLite] Generating calibration dataset …")
    X_cal, _, _, _ = generate_training_dataset(
        num_cycles=num_calibration_cycles, seed=seed + 100
    )
    # Subsample for efficiency
    cal_subset = X_cal[:500]

    def representative_dataset_gen():
        for sample in cal_subset:
            yield [sample[np.newaxis, ...]]

    # Convert to TFLite with full INT8 quantization
    print("[TFLite] Converting with INT8 post-training quantization …")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    try:
        tflite_quant = converter.convert()
        quant_type = "INT8 (Full Integer)"
    except Exception:
        # Fallback: dynamic-range quantization if full INT8 fails
        print("[TFLite] Full INT8 quantisation failed, falling back to dynamic-range …")
        converter2 = tf.lite.TFLiteConverter.from_keras_model(model)
        converter2.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_quant = converter2.convert()
        quant_type = "Dynamic Range (Float16/INT8 hybrid)"

    # Save .tflite
    model_name = model.name.lower()
    tflite_filename = f"{model_name}_quantized.tflite"
    tflite_path = os.path.join(output_dir, tflite_filename)
    with open(tflite_path, "wb") as f:
        f.write(tflite_quant)

    tflite_size_kb = len(tflite_quant) / 1024.0

    # Also export a float32 version for comparison
    converter_f32 = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_f32 = converter_f32.convert()
    f32_path = os.path.join(output_dir, f"{model_name}_float32.tflite")
    with open(f32_path, "wb") as f:
        f.write(tflite_f32)
    f32_size_kb = len(tflite_f32) / 1024.0

    # Update metadata with export info
    metadata["tflite_quantized_path"] = tflite_path
    metadata["tflite_float32_path"] = f32_path
    metadata["quantization_type"] = quant_type
    metadata["tflite_quantized_size_kb"] = round(tflite_size_kb, 2)
    metadata["tflite_float32_size_kb"] = round(f32_size_kb, 2)
    metadata["size_reduction_pct"] = round((1.0 - tflite_size_kb / f32_size_kb) * 100.0, 1)

    # Save metadata JSON
    meta_path = os.path.join(output_dir, f"{model_name}_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"[TFLite] Export complete:")
    print(f"  Quantized  : {tflite_path}  ({tflite_size_kb:.1f} KB)")
    print(f"  Float32    : {f32_path}  ({f32_size_kb:.1f} KB)")
    print(f"  Reduction  : {metadata['size_reduction_pct']:.1f}%")
    print(f"  Metadata   : {meta_path}")

    return tflite_path


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Presentation-Ready Summary Generator
# ──────────────────────────────────────────────────────────────────────────────

def generate_presentation_summary(metadata: Dict[str, Any]) -> str:
    """
    Produces a markdown-formatted summary block that can be directly
    copied into a presentation slide or report.
    """
    summary = f"""## Predictive Speed Forecasting Neural Network

### Model Specification

| Property | Value |
| :--- | :--- |
| **Model Type** | {metadata['architecture']} |
| **Model Name** | `{metadata['model_name']}` |
| **Total Parameters** | {metadata['total_parameters']:,} |
| **Input** | Past **{metadata['lookback_window_s']} seconds** of vehicle speed (1 Hz, shape `{metadata['input_shape']}`) |
| **Output** | Next **{metadata['prediction_horizon_s']} seconds** of predicted vehicle speed (shape `{metadata['output_shape']}`) |
| **Prediction Horizon** | **{metadata['prediction_horizon_s']} s** (constraint: {metadata['horizon_constraint']}) |
| **Training Data Source** | {metadata['training_data_source']} |
| **Training Samples** | {metadata['training_samples']:,} |
| **Validation Samples** | {metadata['validation_samples']:,} |

### Performance

| Metric | Value |
| :--- | :--- |
| **Validation MSE Loss** | {metadata['final_val_loss_mse']:.6f} (normalised) |
| **Validation MAE** | **{metadata['final_val_mae_kmh']:.2f} km/h** |
| **Epochs Trained** | {metadata['epochs_trained']} |
| **Batch Size** | {metadata['batch_size']} |
| **Learning Rate** | {metadata['learning_rate']} |

### Deployment

| Property | Value |
| :--- | :--- |
| **Quantization** | {metadata.get('quantization_type', 'N/A')} |
| **Quantized Model Size** | {metadata.get('tflite_quantized_size_kb', 'N/A')} KB |
| **Float32 Model Size** | {metadata.get('tflite_float32_size_kb', 'N/A')} KB |
| **Size Reduction** | {metadata.get('size_reduction_pct', 'N/A')}% |
| **Target Platform** | Automotive MCU (ARM Cortex-M / Renesas RH850 / Infineon AURIX) |
| **Export Format** | TensorFlow Lite (`.tflite`) |
"""
    return summary


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Standalone Execution Entry Point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train & export HEV speed prediction model")
    parser.add_argument("--model-type", choices=["mlp", "cnn"], default="mlp",
                        help="Architecture: 'mlp' or 'cnn'")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--cycles", type=int, default=80,
                        help="Number of synthetic WLTP variants to generate")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output-dir", type=str, default="predictor/exported_model")
    args = parser.parse_args()

    # Train
    model, history, metadata = train_model(
        model_type=args.model_type,
        num_cycles=args.cycles,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    # Export quantized TFLite
    tflite_path = export_tflite_quantized(model, args.output_dir, metadata)

    # Generate and save presentation summary
    summary_md = generate_presentation_summary(metadata)
    summary_path = os.path.join(args.output_dir, "model_summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_md)
    print(f"\n[Summary] Presentation-ready markdown saved to: {summary_path}")
    print("\n" + summary_md)
