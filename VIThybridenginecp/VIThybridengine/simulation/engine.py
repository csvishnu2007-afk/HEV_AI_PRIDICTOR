"""
Simulation engine coordinating drive cycle kinematics, powertrain power split,
battery electrochemistry/SoC integration, and cumulative fuel economy calculations.
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

from models.vehicle import VehicleConfig
from cycle.wltp import DriveCycle
from physics.kinematics import calculate_traction_power, KinematicsResult
from ems.base import BaseEMS, EMSInput, EMSOutput


@dataclass
class SimulationMetrics:
    """High-level Key Performance Indicators (KPIs) for the completed simulation."""
    total_time_s: float
    total_distance_km: float
    avg_speed_kmh: float
    max_speed_kmh: float

    # Fuel Economy
    total_fuel_consumed_liters: float
    fuel_consumption_l_per_100km: float
    total_fuel_energy_mj: float

    # Battery & Electrical
    initial_soc_pct: float
    final_soc_pct: float
    delta_soc_pct: float
    net_battery_energy_kwh: float
    total_regen_energy_kwh: float
    total_traction_energy_kwh: float

    # Powertrain Utilization
    ice_active_time_pct: float
    ev_mode_time_pct: float
    regen_braking_time_pct: float
    standstill_time_pct: float
    max_ice_power_kw: float
    max_em_power_kw: float
    max_regen_power_kw: float
    friction_brake_energy_kwh: float


class HybridSimulationEngine:
    """
    Simulates a parallel hybrid vehicle over a drive cycle using a given EMS strategy.
    """

    def __init__(self, vehicle: VehicleConfig = None):
        self.vehicle = vehicle if vehicle is not None else VehicleConfig()

    def run(
        self,
        cycle: DriveCycle,
        ems_strategy: BaseEMS,
        initial_soc: float = None,
    ) -> Tuple[pd.DataFrame, SimulationMetrics]:
        """
        Executes the full drive cycle simulation.

        Parameters
        ----------
        cycle : DriveCycle
            The target drive cycle.
        ems_strategy : BaseEMS
            The active energy management system controller.
        initial_soc : float, optional
            Starting battery SoC (0.0 to 1.0). Defaults to vehicle.initial_soc.

        Returns
        -------
        Tuple[pd.DataFrame, SimulationMetrics]
            DataFrame with per-second simulation logs and summary metrics.
        """
        vehicle = self.vehicle
        if initial_soc is None:
            initial_soc = vehicle.initial_soc

        # 1. Compute longitudinal kinematics & required traction power
        kinematics: KinematicsResult = calculate_traction_power(
            time_s=cycle.time_s,
            speed_kmh=cycle.speed_kmh,
            vehicle=vehicle,
        )

        n_steps = len(kinematics.time_s)
        time_s = kinematics.time_s
        speed_kmh = kinematics.speed_kmh
        speed_ms = kinematics.speed_ms
        acc_ms2 = kinematics.acceleration_ms2
        distance_km = kinematics.distance_km
        p_demand_kw = kinematics.p_demand_kw

        # Prepare telemetry vectors
        p_ice_kw = np.zeros(n_steps, dtype=np.float64)
        p_em_kw = np.zeros(n_steps, dtype=np.float64)
        p_brake_mech_kw = np.zeros(n_steps, dtype=np.float64)
        ice_on = np.zeros(n_steps, dtype=bool)
        soc = np.zeros(n_steps, dtype=np.float64)
        mode = np.empty(n_steps, dtype=object)

        fuel_rate_g_s = np.zeros(n_steps, dtype=np.float64)
        fuel_consumed_step_liters = np.zeros(n_steps, dtype=np.float64)
        fuel_consumed_cum_liters = np.zeros(n_steps, dtype=np.float64)
        fuel_consumption_l_100km = np.zeros(n_steps, dtype=np.float64)
        bsfc_g_kwh = np.zeros(n_steps, dtype=np.float64)
        p_bat_electrical_kw = np.zeros(n_steps, dtype=np.float64)

        # Reset strategy internal state
        ems_strategy.reset()
        current_soc = float(initial_soc)

        cum_fuel_liters = 0.0
        total_regen_joules = 0.0
        total_traction_joules = 0.0
        total_brake_joules = 0.0

        for t_idx in range(n_steps):
            dt = 1.0 if t_idx == 0 else max(0.01, time_s[t_idx] - time_s[t_idx - 1])

            # Prepare EMS input state
            ems_in = EMSInput(
                time_s=float(time_s[t_idx]),
                speed_kmh=float(speed_kmh[t_idx]),
                speed_ms=float(speed_ms[t_idx]),
                acceleration_ms2=float(acc_ms2[t_idx]),
                p_demand_kw=float(p_demand_kw[t_idx]),
                soc=float(current_soc),
                distance_km=float(distance_km[t_idx]),
            )

            # Query the Energy Management Strategy
            decision: EMSOutput = ems_strategy.decide_power_split(ems_in, vehicle)

            # Record decisions
            p_ice = max(0.0, min(vehicle.ice_power_max, decision.p_ice_kw))
            p_em = max(-vehicle.em_power_max, min(vehicle.em_power_max, decision.p_em_kw))
            p_brake = max(0.0, decision.p_brake_mech_kw)
            is_ice_on = bool(decision.ice_on and p_ice > 0.05)

            p_ice_kw[t_idx] = p_ice
            p_em_kw[t_idx] = p_em
            p_brake_mech_kw[t_idx] = p_brake
            ice_on[t_idx] = is_ice_on
            mode[t_idx] = decision.mode
            soc[t_idx] = current_soc

            # ---------------------------------------------------------
            # Battery Dynamics & State of Charge Update
            # ---------------------------------------------------------
            # Motoring (p_em > 0): Battery discharges -> P_bat > 0
            # Generating (p_em < 0): Battery charges -> P_bat < 0
            if p_em >= 0.0:
                p_bat = p_em / (vehicle.em_efficiency * vehicle.battery_coulombic_efficiency)
            else:
                p_bat = p_em * (vehicle.em_efficiency * vehicle.battery_coulombic_efficiency)
                total_regen_joules += abs(p_em) * 1000.0 * dt

            p_bat_electrical_kw[t_idx] = p_bat

            # Energy change: Joules = kW * 1000 * dt
            delta_e_joules = p_bat * 1000.0 * dt
            next_soc = current_soc - (delta_e_joules / vehicle.battery_capacity_joules)
            next_soc = float(np.clip(next_soc, 0.05, 0.98))

            # ---------------------------------------------------------
            # Fuel Consumption Dynamics
            # ---------------------------------------------------------
            if is_ice_on:
                curr_bsfc = vehicle.compute_bsfc(p_ice)
                # Fuel mass flow (g/s) = (P_ice_kW * BSFC_g/kWh) / 3600
                m_dot = (p_ice * curr_bsfc) / 3600.0
            elif decision.mode == "Stationary Battery Charge":
                curr_bsfc = 300.0
                m_dot = (p_ice * curr_bsfc) / 3600.0
            else:
                curr_bsfc = 0.0
                m_dot = 0.0

            step_fuel_liters = (m_dot * dt) / vehicle.fuel_density_g_per_liter
            cum_fuel_liters += step_fuel_liters

            fuel_rate_g_s[t_idx] = m_dot
            bsfc_g_kwh[t_idx] = curr_bsfc
            fuel_consumed_step_liters[t_idx] = step_fuel_liters
            fuel_consumed_cum_liters[t_idx] = cum_fuel_liters

            # Distance-normalized L/100km trajectory
            curr_dist_km = distance_km[t_idx]
            if curr_dist_km > 0.05:
                fuel_consumption_l_100km[t_idx] = (cum_fuel_liters / curr_dist_km) * 100.0
            else:
                # Smooth initial steps before meaningful distance is accumulated
                fuel_consumption_l_100km[t_idx] = 0.0

            # Energy integrals
            if p_demand_kw[t_idx] > 0.0:
                total_traction_joules += p_demand_kw[t_idx] * 1000.0 * dt
            total_brake_joules += p_brake * 1000.0 * dt

            current_soc = next_soc

        # Handle beginning of fuel consumption trajectory for seamless plotting
        first_valid_idx = np.where(distance_km > 0.05)[0]
        if len(first_valid_idx) > 0:
            first_idx = first_valid_idx[0]
            fuel_consumption_l_100km[:first_idx] = fuel_consumption_l_100km[first_idx]

        # Assemble DataFrame
        df = pd.DataFrame({
            "time_s": time_s,
            "speed_kmh": speed_kmh,
            "speed_ms": speed_ms,
            "acceleration_ms2": acc_ms2,
            "distance_km": distance_km,
            "phase": cycle.phases,
            "p_demand_kw": p_demand_kw,
            "p_ice_kw": p_ice_kw,
            "p_em_kw": p_em_kw,
            "p_bat_electrical_kw": p_bat_electrical_kw,
            "p_brake_mech_kw": p_brake_mech_kw,
            "ice_on": ice_on,
            "mode": mode,
            "soc": soc,
            "soc_pct": soc * 100.0,
            "fuel_rate_g_s": fuel_rate_g_s,
            "fuel_cum_liters": fuel_consumed_cum_liters,
            "fuel_consumption_l_100km": fuel_consumption_l_100km,
            "bsfc_g_kwh": bsfc_g_kwh,
        })

        # Calculate high-level summary metrics
        total_dist_km = float(distance_km[-1])
        final_l_100km = (
            (cum_fuel_liters / total_dist_km * 100.0) if total_dist_km > 0.01 else 0.0
        )
        fuel_mass_kg = (cum_fuel_liters * vehicle.fuel_density_g_per_liter) / 1000.0
        total_fuel_energy_mj = fuel_mass_kg * vehicle.fuel_lhv_mj_per_kg

        delta_soc = (soc[-1] - initial_soc) * 100.0
        net_battery_kwh = (soc[-1] - initial_soc) * vehicle.battery_capacity_kwh

        ice_active_pct = float(np.mean(ice_on) * 100.0)
        ev_mode_pct = float(np.mean(df["mode"].str.contains("EV", na=False)) * 100.0)
        regen_time_pct = float(np.mean(p_em_kw < -0.5) * 100.0)
        standstill_pct = float(np.mean(speed_kmh < 0.5) * 100.0)

        metrics = SimulationMetrics(
            total_time_s=float(time_s[-1]),
            total_distance_km=total_dist_km,
            avg_speed_kmh=float(np.mean(speed_kmh)),
            max_speed_kmh=float(np.max(speed_kmh)),
            total_fuel_consumed_liters=float(cum_fuel_liters),
            fuel_consumption_l_per_100km=float(final_l_100km),
            total_fuel_energy_mj=float(total_fuel_energy_mj),
            initial_soc_pct=float(initial_soc * 100.0),
            final_soc_pct=float(soc[-1] * 100.0),
            delta_soc_pct=float(delta_soc),
            net_battery_energy_kwh=float(net_battery_kwh),
            total_regen_energy_kwh=float(total_regen_joules / (3600.0 * 1000.0)),
            total_traction_energy_kwh=float(total_traction_joules / (3600.0 * 1000.0)),
            ice_active_time_pct=ice_active_pct,
            ev_mode_time_pct=ev_mode_pct,
            regen_braking_time_pct=regen_time_pct,
            standstill_time_pct=standstill_pct,
            max_ice_power_kw=float(np.max(p_ice_kw)),
            max_em_power_kw=float(np.max(p_em_kw)),
            max_regen_power_kw=float(abs(np.min(p_em_kw))),
            friction_brake_energy_kwh=float(total_brake_joules / (3600.0 * 1000.0)),
        )

        return df, metrics
