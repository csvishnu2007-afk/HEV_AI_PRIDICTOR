## Predictive Speed Forecasting Neural Network

### Model Specification

| Property | Value |
| :--- | :--- |
| **Model Type** | Multi-Layer Perceptron (MLP) |
| **Model Name** | `SpeedPredictor_MLP` |
| **Total Parameters** | 5,887 |
| **Input** | Past **10 seconds** of vehicle speed (1 Hz, shape `(10, 1)`) |
| **Output** | Next **15 seconds** of predicted vehicle speed (shape `(15,)`) |
| **Prediction Horizon** | **15 s** (constraint: 10 – 30 s (satisfied: 15 s)) |
| **Training Data Source** | 80 synthetic WLTP Class 3 drive-cycle variants (randomised speed perturbations) |
| **Training Samples** | 120,768 |
| **Validation Samples** | 21,312 |

### Performance

| Metric | Value |
| :--- | :--- |
| **Validation MSE Loss** | 0.004656 (normalised) |
| **Validation MAE** | **6.82 km/h** |
| **Epochs Trained** | 15 |
| **Batch Size** | 128 |
| **Learning Rate** | 0.001 |

### Deployment

| Property | Value |
| :--- | :--- |
| **Quantization** | INT8 (Full Integer) |
| **Quantized Model Size** | 14.16 KB |
| **Float32 Model Size** | 26.81 KB |
| **Size Reduction** | 47.2% |
| **Target Platform** | Automotive MCU (ARM Cortex-M / Renesas RH850 / Infineon AURIX) |
| **Export Format** | TensorFlow Lite (`.tflite`) |
