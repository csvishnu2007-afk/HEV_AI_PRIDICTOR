"""
Predictive Adaptive Equivalent Consumption Minimization Strategy (A-ECMS).

Inherits from the baseline AECMSStrategy and uses the trained TFLite MLP model
to forecast the vehicle speed 15 seconds into the future, dynamically adjusting
the equivalence factor s(t).
"""

import collections
import numpy as np
import tensorflow as tf

from ems.a_ecms_preview import AECMSStrategy
from ems.base import EMSInput, EMSOutput
from models.vehicle import VehicleConfig
from predictor.speed_predictor import LOOKBACK_SECONDS, HORIZON_SECONDS, SPEED_NORMALIZATION


class PredictiveAECMSStrategy(AECMSStrategy):
    """
    ML-Augmented A-ECMS that uses a TFLite neural network to forecast speed
    and opportunistically adjust the battery equivalence factor.
    """

    def __init__(
        self,
        model_path: str = None,
        s_initial: float = 2.70,
        kp_soc: float = 0.08,
        ki_soc: float = 0.001,
        num_candidates: int = 25,
    ):
        super().__init__(
            s_initial=s_initial,
            kp_soc=kp_soc,
            ki_soc=ki_soc,
            num_candidates=num_candidates
        )
        self.name = "Predictive ML A-ECMS"
        self.description = "A-ECMS augmented with a 15-second Neural Network Speed Forecast"

        import os
        from pathlib import Path
        if model_path is None or not os.path.exists(model_path):
            base_dir = Path(__file__).resolve().parent.parent
            default_path = base_dir / "predictor" / "exported_model" / "speedpredictor_mlp_quantized.tflite"
            model_path = str(default_path)

        # Load TFLite Model
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.inp_det = self.interpreter.get_input_details()[0]
        self.out_det = self.interpreter.get_output_details()[0]

        # Extract quantization params for fast inference
        self.scale_in = self.inp_det["quantization_parameters"]["scales"][0]
        self.zp_in = self.inp_det["quantization_parameters"]["zero_points"][0]
        self.scale_out = self.out_det["quantization_parameters"]["scales"][0]
        self.zp_out = self.out_det["quantization_parameters"]["zero_points"][0]

        # State memory for ML input
        self.speed_history = collections.deque(maxlen=LOOKBACK_SECONDS)
        self.predictive_s_modifier = 0.0
        self.last_forecast = np.zeros(HORIZON_SECONDS)

    def reset(self) -> None:
        super().reset()
        self.speed_history.clear()
        self.predictive_s_modifier = 0.0
        self.last_forecast = np.zeros(HORIZON_SECONDS)

    def decide_power_split(
        self,
        ems_input: EMSInput,
        vehicle: VehicleConfig,
    ) -> EMSOutput:
        
        # 1. Update 10-second sliding window
        self.speed_history.append(ems_input.speed_kmh)

        # 2. ML Inference (only when buffer is fully populated)
        if len(self.speed_history) == LOOKBACK_SECONDS:
            history_arr = np.array(self.speed_history, dtype=np.float32)
            
            # Normalize and Quantize Input
            speed_norm = history_arr / SPEED_NORMALIZATION
            input_int8 = np.clip(np.round(speed_norm / self.scale_in + self.zp_in), -128, 127).astype(np.int8)
            input_tensor = input_int8.reshape(1, LOOKBACK_SECONDS, 1)

            # TFLite Invoke
            self.interpreter.set_tensor(self.inp_det["index"], input_tensor)
            self.interpreter.invoke()
            output_int8 = self.interpreter.get_tensor(self.out_det["index"])[0]

            # Dequantize Output
            output_norm = (output_int8.astype(np.float32) - self.zp_out) * self.scale_out
            predicted_speeds_kmh = output_norm * SPEED_NORMALIZATION
            self.last_forecast = predicted_speeds_kmh

            # 3. Dynamic Strategy Adjustment based on 15s forecast
            current_speed = ems_input.speed_kmh
            avg_future_speed = np.mean(predicted_speeds_kmh)

            if avg_future_speed < current_speed - 5.0:
                # Imminent deceleration: electricity is cheap because regen is coming.
                # Drop s(t) significantly to use battery now and make room for regen.
                self.predictive_s_modifier = -0.5
            elif avg_future_speed > current_speed + 5.0:
                # Imminent acceleration: conserve battery for the upcoming high power demand.
                self.predictive_s_modifier = +0.2
            else:
                # Steady state
                self.predictive_s_modifier = 0.0
        else:
            self.predictive_s_modifier = 0.0
            self.last_forecast = np.full(HORIZON_SECONDS, ems_input.speed_kmh)

        # 4. Delegate to parent class A-ECMS optimization with the modified s_factor
        decision = super().decide_power_split(ems_input, vehicle)

        # Attach prediction diagnostics for telemetry/plots if needed
        if decision.info is None:
            decision.info = {}
        decision.info["ml_s_modifier"] = self.predictive_s_modifier
        decision.info["forecast_avg_speed"] = np.mean(self.last_forecast)
        
        return decision
