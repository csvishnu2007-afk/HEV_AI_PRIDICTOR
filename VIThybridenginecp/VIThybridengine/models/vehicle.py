"""
Vehicle configuration and physical parameter models for the parallel hybrid powertrain.
Strict parameters:
- Vehicle Mass: 1500 kg
- ICE Max Power: 100 kW
- Electric Motor Max Power: 45 kW
- Battery Capacity: 1.2 kWh
"""

from dataclasses import dataclass


@dataclass
class VehicleConfig:
    # Strict Primary Parameters
    mass: float = 1500.0              # kg: Total vehicle curb + test mass
    ice_power_max: float = 100.0      # kW: Maximum internal combustion engine power
    em_power_max: float = 45.0        # kW: Maximum electric motor power (motoring & regen)
    battery_capacity_kwh: float = 1.2 # kWh: Nominal high-voltage traction battery capacity

    # Longitudinal Aerodynamics & Rolling Physics
    cd: float = 0.28                  # Aerodynamic drag coefficient
    frontal_area: float = 2.22        # m^2: Frontal projected area
    air_density: float = 1.225        # kg/m^3: Standard ambient air density (sea level, 15°C)
    crr: float = 0.0095               # Rolling resistance coefficient (modern low-RR tires)
    gravity: float = 9.80665          # m/s^2: Gravitational acceleration
    rotating_mass_factor: float = 1.05 # delta: Factor accounting for rotational inertia of wheels & drivetrain

    # Powertrain & Transmission Efficiencies
    transmission_efficiency: float = 0.95 # Mechanical drivetrain/gearbox transmission efficiency
    em_efficiency: float = 0.92          # Average electric motor + inverter roundtrip efficiency
    battery_coulombic_efficiency: float = 0.96 # Battery electrochemical internal efficiency

    # Battery State of Charge (SoC) Boundaries
    soc_min: float = 0.20             # Minimum allowable battery SoC (protection threshold)
    soc_target: float = 0.60          # Nominal charge-sustaining target SoC
    soc_max: float = 0.85             # Maximum allowable battery SoC (regen buffer ceiling)
    initial_soc: float = 0.60         # Default starting SoC (60%)

    # ICE Fuel Consumption Modeling (Brake Specific Fuel Consumption - BSFC)
    # Gasoline standard properties:
    fuel_density_g_per_liter: float = 745.0  # g/L (typical Euro 95 RON gasoline density)
    fuel_lhv_mj_per_kg: float = 43.5         # MJ/kg: Lower Heating Value of gasoline
    bsfc_min_g_kwh: float = 225.0            # g/kWh: Minimum BSFC at best efficiency island (sweet spot)
    ice_idle_fuel_rate_g_s: float = 0.22     # g/s: Fuel consumption rate at idle (when idling)
    optimal_ice_power_kw: float = 38.0       # kW: Peak thermal efficiency operating power point

    @property
    def battery_capacity_joules(self) -> float:
        """Battery capacity converted to Joules (1.2 kWh = 4,320,000 J)."""
        return self.battery_capacity_kwh * 3600.0 * 1000.0

    def compute_bsfc(self, p_ice_kw: float) -> float:
        """
        Calculates the Brake Specific Fuel Consumption (BSFC in g/kWh)
        as a smooth convex function of engine load.
        """
        if p_ice_kw <= 0.0:
            return 0.0
        # Load ratio relative to max engine power
        load_ratio = max(0.05, min(1.0, p_ice_kw / self.ice_power_max))
        # Quadratic curve centered around peak efficiency (~35-45% load)
        optimal_ratio = self.optimal_ice_power_kw / self.ice_power_max
        bsfc = self.bsfc_min_g_kwh + 180.0 * (load_ratio - optimal_ratio) ** 2
        if load_ratio < 0.25:
            bsfc += 40.0 * (0.25 - load_ratio) / 0.25
        return float(min(550.0, max(self.bsfc_min_g_kwh, bsfc)))
