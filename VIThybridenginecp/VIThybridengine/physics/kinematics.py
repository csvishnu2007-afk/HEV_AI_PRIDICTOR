"""
Longitudinal kinematic physics engine for vehicle traction power calculation.
Models aerodynamic drag, tire rolling resistance, rotating inertial acceleration,
and driveline transmission losses over time-speed drive cycles.
"""

from dataclasses import dataclass
import numpy as np
from models.vehicle import VehicleConfig


@dataclass
class KinematicsResult:
    """Complete kinematic and longitudinal dynamic forces/powers result."""
    time_s: np.ndarray             # Time vector [s]
    speed_kmh: np.ndarray          # Vehicle speed [km/h]
    speed_ms: np.ndarray           # Vehicle speed [m/s]
    acceleration_ms2: np.ndarray   # Longitudinal acceleration [m/s^2]
    distance_m: np.ndarray         # Cumulative distance [m]
    distance_km: np.ndarray        # Cumulative distance [km]
    f_aero_n: np.ndarray           # Aerodynamic drag resistance force [N]
    f_roll_n: np.ndarray           # Tire rolling resistance force [N]
    f_acc_n: np.ndarray            # Inertial acceleration force (including rotating mass) [N]
    f_traction_n: np.ndarray       # Total net traction force at wheels [N]
    p_wheel_kw: np.ndarray         # Power required at the wheels [kW]
    p_demand_kw: np.ndarray        # Power demanded at the powertrain shaft (transmission adjusted) [kW]


def calculate_traction_power(
    time_s: np.ndarray,
    speed_kmh: np.ndarray,
    vehicle: VehicleConfig = None,
    road_grade_rad: float = 0.0,
) -> KinematicsResult:
    """
    Computes longitudinal forces and required traction power over a drive cycle array.

    Parameters
    ----------
    time_s : np.ndarray
        Time series vector in seconds (e.g. 0, 1, 2, ..., T).
    speed_kmh : np.ndarray
        Vehicle velocity profile in km/h.
    vehicle : VehicleConfig, optional
        Vehicle physical configuration (defaults to standard 1500kg, 100kW ICE, 45kW EM, 1.2kWh HEV).
    road_grade_rad : float, optional
        Road grade slope angle in radians (default 0.0 for flat road).

    Returns
    -------
    KinematicsResult
        Full array of speeds, accelerations, forces, and powertrain power demands.
    """
    if vehicle is None:
        vehicle = VehicleConfig()

    time_s = np.asarray(time_s, dtype=np.float64)
    speed_kmh = np.asarray(speed_kmh, dtype=np.float64)

    # Convert velocity from km/h to m/s
    speed_ms = speed_kmh / 3.6

    # Time step delta (handling potential varying dt, though standard WLTP is 1.0s)
    dt = np.diff(time_s, prepend=time_s[0] - (time_s[1] - time_s[0] if len(time_s) > 1 else 1.0))
    dt = np.where(dt <= 0.0, 1.0, dt)

    # Calculate longitudinal acceleration (dv/dt)
    # Using central gradient with edge preservation
    if len(speed_ms) > 1:
        acc_ms2 = np.gradient(speed_ms, time_s)
    else:
        acc_ms2 = np.zeros_like(speed_ms)

    # Cumulative distance traveled
    distance_m = np.cumsum(speed_ms * dt)
    distance_km = distance_m / 1000.0

    # 1. Aerodynamic Drag Force: F_aero = 0.5 * rho * Cd * Af * v^2
    f_aero_n = 0.5 * vehicle.air_density * vehicle.cd * vehicle.frontal_area * (speed_ms ** 2)

    # 2. Tire Rolling Resistance: F_roll = m * g * Crr * cos(theta)
    # Smooth step to ensure zero rolling resistance when completely stopped
    # tanh transition over 0.2 m/s avoids numerical chatter at v=0
    moving_factor = np.tanh(speed_ms / 0.2)
    f_roll_n = vehicle.mass * vehicle.gravity * vehicle.crr * np.cos(road_grade_rad) * moving_factor

    # 3. Gravitational Hill Climbing Force: F_grade = m * g * sin(theta)
    f_grade_n = vehicle.mass * vehicle.gravity * np.sin(road_grade_rad)

    # 4. Inertial Acceleration Force (with rotating inertia mass factor):
    # F_acc = m * delta * a
    effective_mass = vehicle.mass * vehicle.rotating_mass_factor
    f_acc_n = effective_mass * acc_ms2

    # 5. Total Net Traction Force required at the wheel contact patch:
    f_traction_n = f_aero_n + f_roll_n + f_grade_n + f_acc_n

    # 6. Power required at wheels: P_wheel = F_traction * v (Watts -> kW)
    p_wheel_kw = (f_traction_n * speed_ms) / 1000.0

    # 7. Powertrain Shaft Power Demand:
    # Accounting for mechanical transmission efficiency (gearbox, differential, bearings)
    # Motoring (P_wheel >= 0): Shaft must supply MORE power than wheel (P_wheel / eta)
    # Braking (P_wheel < 0): Driveline absorbs small friction, so motor receives LESS power (P_wheel * eta)
    eta = vehicle.transmission_efficiency
    p_demand_kw = np.where(
        p_wheel_kw >= 0.0,
        p_wheel_kw / eta,
        p_wheel_kw * eta
    )

    # Standstill cleanup: when vehicle is stopped and no acceleration, demand is 0
    standstill_mask = (speed_ms < 0.05) & (np.abs(acc_ms2) < 0.05)
    p_demand_kw[standstill_mask] = 0.0
    p_wheel_kw[standstill_mask] = 0.0
    f_traction_n[standstill_mask] = 0.0

    return KinematicsResult(
        time_s=time_s,
        speed_kmh=speed_kmh,
        speed_ms=speed_ms,
        acceleration_ms2=acc_ms2,
        distance_m=distance_m,
        distance_km=distance_km,
        f_aero_n=f_aero_n,
        f_roll_n=f_roll_n,
        f_acc_n=f_acc_n,
        f_traction_n=f_traction_n,
        p_wheel_kw=p_wheel_kw,
        p_demand_kw=p_demand_kw,
    )
