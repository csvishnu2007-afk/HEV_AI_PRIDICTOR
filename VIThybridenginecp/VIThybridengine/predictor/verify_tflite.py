"""Verify quantized TFLite model loads, runs inference, and has correct shapes."""
import numpy as np
import tensorflow as tf

interp = tf.lite.Interpreter(
    model_path=r"predictor/exported_model/speedpredictor_mlp_quantized.tflite"
)
interp.allocate_tensors()

inp_det = interp.get_input_details()[0]
out_det = interp.get_output_details()[0]

print("=== Quantized TFLite Model Verification ===")
print(f"Input  shape : {inp_det['shape']}  dtype: {inp_det['dtype']}")
print(f"Output shape : {out_det['shape']}  dtype: {out_det['dtype']}")

# Test inference with a synthetic 10-second speed ramp (30 -> 50 km/h)
speed_history_kmh = np.linspace(30.0, 50.0, 10)
speed_norm = (speed_history_kmh / 140.0).astype(np.float32)

scale_in = inp_det["quantization_parameters"]["scales"][0]
zp_in = inp_det["quantization_parameters"]["zero_points"][0]
input_int8 = np.clip(np.round(speed_norm / scale_in + zp_in), -128, 127).astype(np.int8)
input_tensor = input_int8.reshape(1, 10, 1)

interp.set_tensor(inp_det["index"], input_tensor)
interp.invoke()
output_int8 = interp.get_tensor(out_det["index"])[0]

scale_out = out_det["quantization_parameters"]["scales"][0]
zp_out = out_det["quantization_parameters"]["zero_points"][0]
output_norm = (output_int8.astype(np.float32) - zp_out) * scale_out
output_kmh = output_norm * 140.0

print(f"\nTest Input (past 10s, km/h)     : {np.round(speed_history_kmh, 1)}")
print(f"Predicted (next 15s, km/h)      : {np.round(output_kmh, 1)}")
print(f"\nPrediction length: {len(output_kmh)} seconds  (constraint: 10-30s -> 15s SATISFIED)")

assert inp_det["shape"].tolist() == [1, 10, 1], "Input shape mismatch!"
assert out_det["shape"].tolist() == [1, 15], "Output shape mismatch!"
assert inp_det["dtype"] == np.int8, "Input must be INT8 for quantized model!"
assert out_det["dtype"] == np.int8, "Output must be INT8 for quantized model!"
assert len(output_kmh) == 15, "Must predict exactly 15 seconds!"

print("\nAll assertions PASSED.")
