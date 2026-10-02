"""
Baseline Rule-Based Energy Management System (EMS) for Parallel Hybrid Electric Vehicle.

Core Rule Logic:
1. Speed < 40 km/h: Use Electric Motor (EV mode), with pure electric propulsion.
2. Speed >= 40 km/h OR High Acceleration Demand: Engage Internal Combustion Engine (ICE).
3. Deceleration / Braking: ICE cutoff, maximize regenerative braking via EM (up to 45 kW limit),
   dissipating excess deceleration via mechanical friction brakes.
4. Battery Health & SoC Sustainment: If SoC drops near lower boundary (20-25%), engage ICE
   to charge battery; if demand exceeds ICE 100 kW, electric motor provides torque boost.
"""

from ems.base import BaseEMS, EMSInput, EMSOutput
from models.vehicle import VehicleConfig


class RuleBasedEMS(BaseEMS):
    """
    Deterministic rule-based state machine EMS.
    """

    def __init__(
        self,
        speed_threshold_kmh: float = 40.0,
        high_accel_threshold_ms2: float = 1.20,
        enable_load_point_shifting: bool = True,
        enable_torque_boost: bool = True,
    ):
        super().__init__(
            name="Baseline Rule-Based EMS",
            description=(
                f"EV Mode under {speed_threshold_kmh:.0f} km/h; "
                f"ICE engaged at >= {speed_threshold_kmh:.0f} km/h or high acceleration (> {high_accel_threshold_ms2:.2f} m/s²)."
            )
        )
        self.speed_threshold_kmh = speed_threshold_kmh
        self.high_accel_threshold_ms2 = high_accel_threshold_ms2
        self.enable_load_point_shifting = enable_load_point_shifting
        self.enable_torque_boost = enable_torque_boost

    def decide_power_split(
        self,
        ems_input: EMSInput,
        vehicle: VehicleConfig,
    ) -> EMSOutput:
        """Executes deterministic rule-based powertrain power split."""
        speed = ems_input.speed_kmh
        accel = ems_input.acceleration_ms2
        p_demand = ems_input.p_demand_kw
        soc = ems_input.soc

        p_ice_max = vehicle.ice_power_max      # 100 kW
        p_em_max = vehicle.em_power_max        # 45 kW
        soc_min = vehicle.soc_min              # 0.20
        soc_target = vehicle.soc_target        # 0.60
        soc_max = vehicle.soc_max              # 0.85

        # -------------------------------------------------------------
        # 1. BRAKING / DECELERATION REGIME (P_demand < 0)
        # -------------------------------------------------------------
        if p_demand <= 0.0:
            # Engine is cut off during deceleration to save fuel
            p_ice = 0.0
            ice_on = False

            # Check battery overcharge limit
            if soc >= soc_max:
                # Battery full, cannot accept regen
                p_em = 0.0
                p_brake_mech = abs(p_demand)
                mode = "Friction Braking (Battery Full)"
            else:
                # Regenerative braking through EM up to motor rating (45 kW)
                # Note: p_demand is negative, p_em is negative for regeneration
                # Max regen capacity scaled down if very close to soc_max
                regen_capacity = p_em_max
                if soc > 0.80:
                    regen_capacity *= (soc_max - soc) / (soc_max - 0.80)

                p_em = max(-regen_capacity, p_demand)
                # Any leftover braking demand is handled by mechanical friction brakes
                p_brake_mech = max(0.0, abs(p_demand) - abs(p_em))
                mode = "Regenerative Braking" if abs(p_em) > 0.1 else "Standstill"

            return EMSOutput(
                p_ice_kw=p_ice,
                p_em_kw=p_em,
                p_brake_mech_kw=p_brake_mech,
                ice_on=ice_on,
                mode=mode,
                info={"rule_trigger": "Braking/Decel"},
            )

        # -------------------------------------------------------------
        # 2. STANDSTILL REGIME (v < 0.5 km/h and P_demand == 0)
        # -------------------------------------------------------------
        if speed < 0.5 and p_demand < 0.5:
            if soc < (soc_min + 0.05):
                # Critical battery protection: Engine idle-charges battery
                p_ice = min(12.0, p_ice_max)
                p_em = -p_ice
                return EMSOutput(
                    p_ice_kw=p_ice,
                    p_em_kw=p_em,
                    p_brake_mech_kw=0.0,
                    ice_on=True,
                    mode="Stationary Battery Charge",
                    info={"rule_trigger": "Critical Low SoC at Standstill"},
                )
            else:
                # Start-Stop active: Engine is off, zero emissions
                return EMSOutput(
                    p_ice_kw=0.0,
                    p_em_kw=0.0,
                    p_brake_mech_kw=0.0,
                    ice_on=False,
                    mode="Standstill (Start-Stop)",
                    info={"rule_trigger": "Idle Start-Stop"},
                )

        # -------------------------------------------------------------
        # 3. PROPULSION REGIME (P_demand > 0)
        # Check primary rule conditions:
        # Condition A: Under speed threshold (e.g. < 40 km/h)
        # Condition B: High acceleration demand (a >= threshold)
        # -------------------------------------------------------------
        is_low_speed = speed < self.speed_threshold_kmh
        is_high_accel = accel >= self.high_accel_threshold_ms2
        is_battery_depleted = soc <= (soc_min + 0.02)
        exceeds_em_capacity = p_demand > p_em_max

        # Rule Branch 1: EV Mode
        # Speeds under 40 km/h, normal acceleration, battery healthy, and within EM power limit (45 kW)
        if is_low_speed and not is_high_accel and not is_battery_depleted and not exceeds_em_capacity:
            p_ice = 0.0
            p_em = p_demand
            ice_on = False
            mode = "EV Mode (Pure Electric)"
            trigger = f"Speed < {self.speed_threshold_kmh:.0f} km/h & Normal Accel"

            return EMSOutput(
                p_ice_kw=p_ice,
                p_em_kw=p_em,
                p_brake_mech_kw=0.0,
                ice_on=ice_on,
                mode=mode,
                info={"rule_trigger": trigger},
            )

        # Rule Branch 2: ICE Engaged
        # Triggered by: Speed >= 40 km/h, OR High Acceleration, OR Low Battery, OR Demand > 45 kW
        ice_on = True

        if is_high_accel:
            trigger_reason = f"High Accel Demand ({accel:.2f} m/s² >= {self.high_accel_threshold_ms2:.2f})"
        elif not is_low_speed:
            trigger_reason = f"Speed ({speed:.1f} km/h >= {self.speed_threshold_kmh:.0f})"
        elif exceeds_em_capacity:
            trigger_reason = f"Demand ({p_demand:.1f} kW > EM max {p_em_max:.0f} kW)"
        else:
            trigger_reason = "Low Battery Protection"

        # Determine power split under ICE engagement:
        # Case 2a: Demand exceeds maximum ICE power (100 kW) -> Boost Assist
        if p_demand > p_ice_max:
            p_ice = p_ice_max
            deficit = p_demand - p_ice
            if self.enable_torque_boost and soc > soc_min:
                p_em = min(deficit, p_em_max)
            else:
                p_em = 0.0
            mode = "Torque Boost (ICE Max + EM Assist)"

        # Case 2b: Battery SoC is low (below target) -> Load Point Shifting (Charge battery)
        elif self.enable_load_point_shifting and soc < (soc_target - 0.03):
            # Shift ICE load up into its optimal BSFC efficiency zone (~35-50 kW)
            # and use surplus generation to recharge the battery via EM generator mode
            # Desired charging power proportional to SoC deficit
            soc_deficit = max(0.0, (soc_target - soc) / (soc_target - soc_min))
            desired_charge = min(15.0, p_em_max, soc_deficit * 18.0)

            # Do not exceed ICE 100 kW ceiling
            charge_power = min(desired_charge, p_ice_max - p_demand)
            p_ice = p_demand + charge_power
            p_em = -charge_power
            mode = "Load Point Shift (ICE Drive + Battery Charge)"

        # Case 2c: Battery SoC is high (above target + buffer) -> Motor Assist (Discharge surplus)
        elif soc > (soc_target + 0.12) and p_demand > 18.0:
            # Relieve engine load by assisting with electric motor
            assist_power = min(15.0, p_em_max, 0.40 * p_demand)
            p_ice = max(5.0, p_demand - assist_power)
            p_em = p_demand - p_ice
            mode = "Motor Assist (ICE + EM Hybrid)"

        # Case 2d: ICE Direct Drive
        else:
            p_ice = min(p_ice_max, p_demand)
            p_em = 0.0
            mode = "ICE Traction"

        return EMSOutput(
            p_ice_kw=float(p_ice),
            p_em_kw=float(p_em),
            p_brake_mech_kw=0.0,
            ice_on=ice_on,
            mode=mode,
            info={"rule_trigger": trigger_reason},
        )
