"""UI and plotting package for hybrid EMS dashboard."""
from ui.plots import (
    create_fuel_consumption_plot,
    create_soc_trajectory_plot,
    create_ice_pattern_and_power_split_plot,
    create_drive_cycle_speed_plot,
)

__all__ = [
    "create_fuel_consumption_plot",
    "create_soc_trajectory_plot",
    "create_ice_pattern_and_power_split_plot",
    "create_drive_cycle_speed_plot",
]
