"""
Adaptive Equivalent Consumption Minimization Strategy (A-ECMS) Preview / Modular Extension.

Demonstrates seamless modular swap with BaseEMS:
Calculates the instantaneous Hamiltonian cost function:
  H(P_ice, P_em) = m_dot_fuel(P_ice) + s(t) * (P_bat(P_em) / LHV_fuel)
where s(t) is the adaptive equivalence factor regulated by a PI controller on battery SoC error.
In subsequent phases, this module can be directly replaced by an offline-trained Deep RL policy
or predictive ML neural network.
"""

from ems.base import BaseEMS, EMSInput, EMSOutput
from models.vehicle import VehicleConfig
import numpy as np


class AECMSStrategy(BaseEMS):
    """
    Adaptive Equivalent Consumption Minimization Strategy (A-ECMS).
    Modularly inherits from BaseEMS.
    """

    def __init__(
        self,
        s_initial: float = 2.70,
        kp_soc: float = 0.08,
        ki_soc: float = 0.001,
        num_candidates: int = 25,
    ):
        super().__init__(
            name="Adaptive ECMS (Preview)",
            description=(
                "Pontryagin's Minimum Principle based instantaneous optimization "
                "with adaptive equivalence factor s(t)."
            )
        )
        self.s_initial = s_initial
        self.kp_soc = kp_soc
        self.ki_soc = ki_soc
        self.num_candidates = num_candidates

        # Internal state
        self.s_factor = s_initial
        self.soc_integral_error = 0.0

    def reset(self) -> None:
        """Resets the adaptive state between simulations."""
        self.s_factor = self.s_initial
        self.soc_integral_error = 0.0

    def decide_power_split(
        self,
        ems_input: EMSInput,
        vehicle: VehicleConfig,
    ) -> EMSOutput:
        """Evaluates Hamiltonian cost over permissible power splits and selects minimum."""
        p_demand = ems_input.p_demand_kw
        soc = ems_input.soc
        target_soc = vehicle.soc_target

        # Update adaptive equivalence factor s(t) via PI controller
        # If SoC < target, s increases (electricity is expensive -> preserve battery, use ICE)
        # If SoC > target, s decreases (electricity is cheap -> use EV motor)
        soc_error = target_soc - soc
        self.soc_integral_error += soc_error
        # Clamp integral to prevent windup
        self.soc_integral_error = max(-50.0, min(50.0, self.soc_integral_error))

        self.s_factor = self.s_initial + self.kp_soc * soc_error + self.ki_soc * self.soc_integral_error
        # Apply optional predictive modifier from ML sub-classes
        self.s_factor += getattr(self, 'predictive_s_modifier', 0.0)
        self.s_factor = max(1.2, min(5.5, self.s_factor))

        # 1. Braking / Deceleration (P_demand <= 0)
        if p_demand <= 0.0:
            p_ice = 0.0
            ice_on = False
            if soc >= vehicle.soc_max:
                p_em = 0.0
                p_brake_mech = abs(p_demand)
                mode = "Friction Braking (SoC Max)"
            else:
                p_em = max(-vehicle.em_power_max, p_demand)
                p_brake_mech = max(0.0, abs(p_demand) - abs(p_em))
                mode = "Regenerative Braking (A-ECMS)"
            return EMSOutput(
                p_ice_kw=p_ice,
                p_em_kw=p_em,
                p_brake_mech_kw=p_brake_mech,
                ice_on=ice_on,
                mode=mode,
                info={"s_factor": self.s_factor},
            )

        # 2. Standstill
        if ems_input.speed_kmh < 0.5 and p_demand < 0.5:
            return EMSOutput(
                p_ice_kw=0.0,
                p_em_kw=0.0,
                p_brake_mech_kw=0.0,
                ice_on=False,
                mode="Standstill (A-ECMS Stop)",
                info={"s_factor": self.s_factor},
            )

        # 3. Propulsion: Evaluate candidate P_ice candidates and pick argmin H
        # Candidate range: from 0 (pure EV) to min(p_ice_max, p_demand + 20kW)
        max_ice_candidate = min(vehicle.ice_power_max, p_demand + 15.0)
        p_ice_grid = np.linspace(0.0, max_ice_candidate, self.num_candidates)

        best_cost = float("inf")
        best_p_ice = 0.0
        best_p_em = p_demand

        for p_ice in p_ice_grid:
            p_em = p_demand - p_ice

            # Check EM limits [-45 kW, 45 kW]
            if p_em < -vehicle.em_power_max or p_em > vehicle.em_power_max:
                continue

            # Battery boundary penalties
            if p_em > 0 and soc <= vehicle.soc_min:
                continue
            if p_em < 0 and soc >= vehicle.soc_max:
                continue

            # 1. Fuel mass flow rate (g/s)
            if p_ice > 0.01:
                bsfc = vehicle.compute_bsfc(p_ice) # g/kWh
                fuel_rate_g_s = (p_ice * bsfc) / 3600.0
            else:
                fuel_rate_g_s = 0.0

            # 2. Virtual equivalent fuel rate from battery power (g/s)
            # P_bat in kW
            if p_em >= 0.0:
                p_bat_kw = p_em / (vehicle.em_efficiency * vehicle.battery_coulombic_efficiency)
            else:
                p_bat_kw = p_em * (vehicle.em_efficiency * vehicle.battery_coulombic_efficiency)

            # Equivalent fuel mass rate = (s * P_bat) / (LHV * 1000)
            lhv_kw_s_per_g = (vehicle.fuel_lhv_mj_per_kg * 1000.0) / 1000.0 # MJ/kg -> kJ/g = kW*s/g
            equiv_fuel_rate_g_s = (self.s_factor * p_bat_kw) / lhv_kw_s_per_g

            # Total Hamiltonian
            h_cost = fuel_rate_g_s + equiv_fuel_rate_g_s

            if h_cost < best_cost:
                best_cost = h_cost
                best_p_ice = p_ice
                best_p_em = p_em

        ice_on = best_p_ice > 0.5
        if best_p_ice <= 0.5:
            mode = "A-ECMS Pure EV Mode"
        elif best_p_em < -0.5:
            mode = "A-ECMS Optimal Load Point Charging"
        elif best_p_em > 0.5:
            mode = "A-ECMS Hybrid Boost Assist"
        else:
            mode = "A-ECMS Direct Engine Drive"

        return EMSOutput(
            p_ice_kw=float(best_p_ice),
            p_em_kw=float(best_p_em),
            p_brake_mech_kw=0.0,
            ice_on=ice_on,
            mode=mode,
            info={"s_factor": self.s_factor, "h_cost": best_cost},
        )
