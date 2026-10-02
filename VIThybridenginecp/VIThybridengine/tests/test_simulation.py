"""
Unit tests for vehicle configuration, longitudinal kinematics, baseline rule-based EMS,
battery SoC continuity, and full WLTP simulation execution.
"""

import pytest
import numpy as np
import pandas as pd

from models.vehicle import VehicleConfig
from physics.kinematics import calculate_traction_power
from cycle.wltp import generate_synthetic_wltp
from ems.base import EMSInput
from ems.rule_based import RuleBasedEMS
from ems.a_ecms_preview import AECMSStrategy
from simulation.engine import HybridSimulationEngine


def test_strict_vehicle_parameters():
    """Verify strictly specified vehicle parameters."""
    v = VehicleConfig()
    assert v.mass == 1500.0, f"Vehicle mass must be 1500 kg, got {v.mass}"
    assert v.ice_power_max == 100.0, f"ICE power must be 100 kW, got {v.ice_power_max}"
    assert v.em_power_max == 45.0, f"EM power must be 45 kW, got {v.em_power_max}"
    assert v.battery_capacity_kwh == 1.2, f"Battery capacity must be 1.2 kWh, got {v.battery_capacity_kwh}"
    assert abs(v.battery_capacity_joules - 4_320_000.0) < 1e-3


def test_kinematics_standstill_and_cruise():
    """Test longitudinal physics calculations at standstill and at steady cruise."""
    v = VehicleConfig()
    t = np.array([0.0, 1.0, 2.0])

    # Standstill test: v = 0 km/h
    res_standstill = calculate_traction_power(t, np.array([0.0, 0.0, 0.0]), v)
    assert np.allclose(res_standstill.f_aero_n, 0.0)
    assert np.allclose(res_standstill.p_demand_kw, 0.0)

    # Steady cruise at 100 km/h (27.78 m/s, zero acceleration)
    res_cruise = calculate_traction_power(t, np.array([100.0, 100.0, 100.0]), v)
    speed_ms = 100.0 / 3.6
    expected_f_aero = 0.5 * v.air_density * v.cd * v.frontal_area * (speed_ms ** 2)
    expected_f_roll = v.mass * v.gravity * v.crr
    expected_p_wheel = (expected_f_aero + expected_f_roll) * speed_ms / 1000.0
    expected_p_demand = expected_p_wheel / v.transmission_efficiency

    # Midpoint checks
    assert abs(res_cruise.f_aero_n[1] - expected_f_aero) < 1.0
    assert abs(res_cruise.f_roll_n[1] - expected_f_roll) < 5.0
    assert abs(res_cruise.p_demand_kw[1] - expected_p_demand) < 0.5
    assert res_cruise.p_demand_kw[1] > 0.0


def test_rule_based_ems_logic():
    """
    Test baseline rule logic:
    - Speed < 40 km/h & normal accel -> EV Mode (ICE OFF, EM ON)
    - Speed >= 40 km/h -> ICE engaged (ICE ON)
    - High accel demand (>= 1.2 m/s^2) -> ICE engaged even under 40 km/h
    - Deceleration -> Regenerative braking (ICE OFF, EM negative)
    """
    v = VehicleConfig()
    ems = RuleBasedEMS(speed_threshold_kmh=40.0, high_accel_threshold_ms2=1.2)

    # 1. Under 40 km/h, moderate acceleration (e.g. 30 km/h, a = 0.5 m/s^2, demand = 15 kW)
    inp_ev = EMSInput(
        time_s=10.0, speed_kmh=30.0, speed_ms=30.0/3.6,
        acceleration_ms2=0.5, p_demand_kw=15.0, soc=0.60, distance_km=0.1
    )
    out_ev = ems.decide_power_split(inp_ev, v)
    assert not out_ev.ice_on, "ICE must be OFF under 40 km/h during normal acceleration"
    assert out_ev.p_ice_kw == 0.0
    assert out_ev.p_em_kw == 15.0
    assert "EV" in out_ev.mode

    # 2. At or above 40 km/h (e.g. 50 km/h, a = 0.3 m/s^2, demand = 20 kW)
    inp_ice = EMSInput(
        time_s=20.0, speed_kmh=50.0, speed_ms=50.0/3.6,
        acceleration_ms2=0.3, p_demand_kw=20.0, soc=0.60, distance_km=0.3
    )
    out_ice = ems.decide_power_split(inp_ice, v)
    assert out_ice.ice_on, "ICE must be ON at speeds >= 40 km/h"
    assert out_ice.p_ice_kw > 0.0

    # 3. Low speed (25 km/h) but HIGH acceleration (1.8 m/s^2)
    inp_high_accel = EMSInput(
        time_s=30.0, speed_kmh=25.0, speed_ms=25.0/3.6,
        acceleration_ms2=1.8, p_demand_kw=35.0, soc=0.60, distance_km=0.4
    )
    out_high_accel = ems.decide_power_split(inp_high_accel, v)
    assert out_high_accel.ice_on, "ICE must engage during high acceleration demand even below 40 km/h"

    # 4. Deceleration / Braking (demand = -25 kW)
    inp_brake = EMSInput(
        time_s=40.0, speed_kmh=60.0, speed_ms=60.0/3.6,
        acceleration_ms2=-1.2, p_demand_kw=-25.0, soc=0.60, distance_km=0.6
    )
    out_brake = ems.decide_power_split(inp_brake, v)
    assert not out_brake.ice_on, "ICE must be OFF during braking/deceleration"
    assert out_brake.p_ice_kw == 0.0
    assert out_brake.p_em_kw == -25.0
    assert out_brake.p_brake_mech_kw == 0.0
    assert "Regenerative Braking" in out_brake.mode

    # 5. Heavy Deceleration exceeding EM rating (demand = -60 kW, EM max = 45 kW)
    inp_heavy_brake = EMSInput(
        time_s=45.0, speed_kmh=80.0, speed_ms=80.0/3.6,
        acceleration_ms2=-2.5, p_demand_kw=-60.0, soc=0.60, distance_km=0.7
    )
    out_heavy_brake = ems.decide_power_split(inp_heavy_brake, v)
    assert out_heavy_brake.p_em_kw == -45.0, "EM regen must saturate at -45 kW"
    assert out_heavy_brake.p_brake_mech_kw == 15.0, "Friction brakes must absorb excess 15 kW"


def test_full_simulation_wltp():
    """Verify complete simulation execution over synthetic WLTP cycle."""
    v = VehicleConfig()
    cycle = generate_synthetic_wltp("Full WLTP")
    assert cycle.duration_s == 1800.0
    assert cycle.total_distance_km > 20.0
    assert cycle.max_speed_kmh > 125.0

    ems = RuleBasedEMS(speed_threshold_kmh=40.0)
    engine = HybridSimulationEngine(v)
    df, metrics = engine.run(cycle, ems, initial_soc=0.60)

    # Check dataframe length and column integrity
    assert len(df) == 1801
    assert "fuel_consumption_l_100km" in df.columns
    assert "soc_pct" in df.columns
    assert "ice_on" in df.columns
    assert "p_ice_kw" in df.columns
    assert "p_em_kw" in df.columns

    # Verify physical plausibility
    assert 3.5 <= metrics.fuel_consumption_l_per_100km <= 7.0, (
        f"Expected realistic HEV fuel consumption (3.5-7.0 L/100km), got {metrics.fuel_consumption_l_per_100km:.2f}"
    )
    assert 20.0 <= metrics.final_soc_pct <= 85.0, "SoC must remain within safe operational buffer"
    assert metrics.total_regen_energy_kwh > 0.1, "Simulation must recuperate energy via regen braking"
    assert metrics.ice_active_time_pct > 20.0, "ICE must be active during higher speed WLTP phases"
