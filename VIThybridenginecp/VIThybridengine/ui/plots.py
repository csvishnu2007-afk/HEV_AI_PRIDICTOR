"""
Dynamic interactive Plotly visualizations for the hybrid vehicle EMS simulation.
Generates:
1. Dynamic Fuel Consumption (L/100km) trajectory
2. Battery State-of-Charge (SoC) trajectory with boundary bands
3. ICE On/Off pattern and Powertrain Power Split (P_ICE, P_EM, P_brake, P_demand)
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from models.vehicle import VehicleConfig
from simulation.engine import SimulationMetrics


# Automotive themed color palette
COLOR_PRIMARY = "#00D26A"       # Electric Green (EM / Efficiency)
COLOR_ICE = "#FF5722"           # ICE Thermal Orange/Red
COLOR_DEMAND = "#2196F3"        # Traction Demand Blue
COLOR_BRAKE = "#9C27B0"         # Friction Brake Purple
COLOR_SOC = "#00BCD4"           # Battery Cyan
COLOR_BG = "rgba(0,0,0,0)"
GRID_COLOR = "rgba(128, 128, 128, 0.18)"


def create_fuel_consumption_plot(df: pd.DataFrame, metrics: SimulationMetrics) -> go.Figure:
    """
    Plot 1: Dynamic Fuel Consumption (L/100km) trajectory over the drive cycle,
    with instantaneous fuel rate on secondary y-axis.
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.65, 0.35],
        subplot_titles=(
            "Cumulative Fuel Economy Trajectory (L/100km)",
            "Instantaneous Engine Fuel Rate (g/s)"
        )
    )

    # Subplot 1: Cumulative Fuel Consumption (L/100km)
    fig.add_trace(
        go.Scatter(
            x=df["time_s"],
            y=df["fuel_consumption_l_100km"],
            mode="lines",
            name="Fuel Economy (L/100km)",
            line=dict(color="#FF7043", width=2.8),
            customdata=np.stack((df["distance_km"], df["fuel_cum_liters"], df["speed_kmh"]), axis=-1),
            hovertemplate=(
                "<b>Time</b>: %{x:.0f} s<br>"
                "<b>Distance</b>: %{customdata[0]:.2f} km<br>"
                "<b>Speed</b>: %{customdata[2]:.1f} km/h<br>"
                "<b>Economy</b>: %{y:.2f} L/100km<br>"
                "<b>Total Fuel</b>: %{customdata[1]:.3f} L<extra></extra>"
            ),
        ),
        row=1, col=1
    )

    # Final fuel economy reference horizontal line
    final_l100km = metrics.fuel_consumption_l_per_100km
    fig.add_hline(
        y=final_l100km,
        line_dash="dot",
        line_color="#FFA726",
        line_width=1.5,
        annotation_text=f"Cycle Final: {final_l100km:.2f} L/100km",
        annotation_position="top right",
        row=1, col=1
    )

    # Subplot 2: Instantaneous Fuel Rate (g/s)
    fig.add_trace(
        go.Scatter(
            x=df["time_s"],
            y=df["fuel_rate_g_s"],
            mode="lines",
            name="Fuel Rate (g/s)",
            line=dict(color="#E64A19", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(230, 74, 25, 0.15)",
            hovertemplate="<b>Time</b>: %{x:.0f} s<br><b>Fuel Rate</b>: %{y:.2f} g/s<extra></extra>",
        ),
        row=2, col=1
    )

    fig.update_layout(
        height=480,
        margin=dict(l=40, r=40, t=50, b=30),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        plot_bgcolor=COLOR_BG,
        paper_bgcolor=COLOR_BG,
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False)
    fig.update_yaxes(title_text="L/100km", row=1, col=1)
    fig.update_yaxes(title_text="Fuel Rate (g/s)", row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)

    return fig


def create_soc_trajectory_plot(
    df: pd.DataFrame,
    metrics: SimulationMetrics,
    vehicle: VehicleConfig,
) -> go.Figure:
    """
    Plot 2: Battery State-of-Charge (SoC) trajectory over the drive cycle,
    with safe operating buffer boundaries and target reference.
    """
    fig = go.Figure()

    soc_min_pct = vehicle.soc_min * 100.0
    soc_max_pct = vehicle.soc_max * 100.0
    soc_target_pct = vehicle.soc_target * 100.0

    # 1. Shaded safe operational band [20% - 85%]
    t_arr = df["time_s"].values
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([t_arr, t_arr[::-1]]),
            y=np.concatenate([np.full_like(t_arr, soc_max_pct), np.full_like(t_arr, soc_min_pct)]),
            fill="toself",
            fillcolor="rgba(0, 188, 212, 0.08)",
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            showlegend=True,
            name="Safe SoC Operational Buffer (20% - 85%)",
        )
    )

    # 2. Target SoC horizontal line
    fig.add_hline(
        y=soc_target_pct,
        line_dash="dash",
        line_color="#4DD0E1",
        line_width=1.5,
        annotation_text=f"Target SoC: {soc_target_pct:.0f}%",
        annotation_position="bottom right",
    )

    # 3. Main Battery SoC Trajectory
    fig.add_trace(
        go.Scatter(
            x=df["time_s"],
            y=df["soc_pct"],
            mode="lines",
            name="Battery SoC (%)",
            line=dict(color="#00E5FF", width=2.8),
            customdata=np.stack((df["p_em_kw"], df["p_bat_electrical_kw"]), axis=-1),
            hovertemplate=(
                "<b>Time</b>: %{x:.0f} s<br>"
                "<b>Battery SoC</b>: %{y:.2f}%<br>"
                "<b>Motor Power</b>: %{customdata[0]:.1f} kW<br>"
                "<b>Battery Power</b>: %{customdata[1]:.1f} kW<extra></extra>"
            ),
        )
    )

    # 4. Start and End Markers
    fig.add_trace(
        go.Scatter(
            x=[df["time_s"].iloc[0]],
            y=[df["soc_pct"].iloc[0]],
            mode="markers+text",
            marker=dict(color="#76FF03", size=10, symbol="circle"),
            text=[f"Start: {metrics.initial_soc_pct:.1f}%"],
            textposition="top right",
            name="Initial SoC",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[df["time_s"].iloc[-1]],
            y=[df["soc_pct"].iloc[-1]],
            mode="markers+text",
            marker=dict(color="#FFD600", size=10, symbol="diamond"),
            text=[f"End: {metrics.final_soc_pct:.1f}% (Δ {metrics.delta_soc_pct:+.1f}%)"],
            textposition="top left",
            name="Final SoC",
            showlegend=False,
        )
    )

    # Min/Max buffer boundary reference lines
    fig.add_hline(y=soc_min_pct, line_dash="dot", line_color="#FF5252", line_width=1.0)
    fig.add_hline(y=soc_max_pct, line_dash="dot", line_color="#FF5252", line_width=1.0)

    fig.update_layout(
        title="Battery State-of-Charge (SoC) Trajectory over Drive Cycle",
        xaxis_title="Time (s)",
        yaxis_title="Battery SoC (%)",
        yaxis_range=[10.0, 95.0],
        height=420,
        margin=dict(l=40, r=40, t=50, b=30),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        plot_bgcolor=COLOR_BG,
        paper_bgcolor=COLOR_BG,
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False)

    return fig


def create_ice_pattern_and_power_split_plot(
    df: pd.DataFrame,
    vehicle: VehicleConfig,
) -> go.Figure:
    """
    Plot 3: Dynamic ICE On/Off pattern displayed synchronously with
    the powertrain power split (ICE Power, EM Power, Brake Power, Traction Demand).
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.30, 0.70],
        subplot_titles=(
            "ICE Operational State (On / Off)",
            "Powertrain Power Split & Demand (kW)"
        )
    )

    # -------------------------------------------------------------
    # Subplot 1: ICE Binary On/Off Pattern
    # -------------------------------------------------------------
    ice_state_numeric = df["ice_on"].astype(int).values

    # Step line for ICE state
    fig.add_trace(
        go.Scatter(
            x=df["time_s"],
            y=ice_state_numeric,
            mode="lines",
            name="ICE State (1=ON, 0=OFF)",
            line=dict(color="#FF5722", width=2.2, shape="hv"),
            fill="tozeroy",
            fillcolor="rgba(255, 87, 34, 0.22)",
            customdata=np.stack((df["mode"], df["speed_kmh"]), axis=-1),
            hovertemplate=(
                "<b>Time</b>: %{x:.0f} s<br>"
                "<b>ICE State</b>: %{text}<br>"
                "<b>Mode</b>: %{customdata[0]}<br>"
                "<b>Speed</b>: %{customdata[1]:.1f} km/h<extra></extra>"
            ),
            text=["ON" if v == 1 else "OFF" for v in ice_state_numeric],
        ),
        row=1, col=1
    )

    # -------------------------------------------------------------
    # Subplot 2: Detailed Powertrain Power Split
    # -------------------------------------------------------------
    # Demand Power (Total required traction)
    fig.add_trace(
        go.Scatter(
            x=df["time_s"],
            y=df["p_demand_kw"],
            mode="lines",
            name="Total Traction Demand",
            line=dict(color="rgba(180, 180, 180, 0.6)", width=1.4, dash="dot"),
            hovertemplate="<b>Demand</b>: %{y:.1f} kW<extra></extra>",
        ),
        row=2, col=1
    )

    # ICE Power Output (0 to 100 kW)
    fig.add_trace(
        go.Scatter(
            x=df["time_s"],
            y=df["p_ice_kw"],
            mode="lines",
            name="ICE Power (0-100 kW)",
            line=dict(color=COLOR_ICE, width=2.4),
            fill="tozeroy",
            fillcolor="rgba(255, 87, 34, 0.12)",
            hovertemplate="<b>ICE Power</b>: %{y:.1f} kW<extra></extra>",
        ),
        row=2, col=1
    )

    # Electric Motor Power (-45 kW regen to +45 kW motoring)
    fig.add_trace(
        go.Scatter(
            x=df["time_s"],
            y=df["p_em_kw"],
            mode="lines",
            name="EM Power (-45 to +45 kW)",
            line=dict(color=COLOR_PRIMARY, width=2.4),
            fill="tozeroy",
            fillcolor="rgba(0, 210, 106, 0.12)",
            hovertemplate="<b>EM Power</b>: %{y:.1f} kW (Regen if < 0)<extra></extra>",
        ),
        row=2, col=1
    )

    # Mechanical Friction Brake Power (only when demand < -45 kW or battery full)
    if df["p_brake_mech_kw"].max() > 0.1:
        fig.add_trace(
            go.Scatter(
                x=df["time_s"],
                y=-df["p_brake_mech_kw"],
                mode="lines",
                name="Friction Brake Dissipation",
                line=dict(color=COLOR_BRAKE, width=1.8, dash="dash"),
                hovertemplate="<b>Friction Brake</b>: %{y:.1f} kW<extra></extra>",
            ),
            row=2, col=1
        )

    # Component rating guide lines
    fig.add_hline(y=vehicle.ice_power_max, line_dash="dot", line_color="rgba(255,87,34,0.4)", row=2, col=1)
    fig.add_hline(y=vehicle.em_power_max, line_dash="dot", line_color="rgba(0,210,106,0.4)", row=2, col=1)
    fig.add_hline(y=-vehicle.em_power_max, line_dash="dot", line_color="rgba(0,210,106,0.4)", row=2, col=1)
    fig.add_hline(y=0.0, line_color="rgba(200,200,200,0.3)", line_width=1.0, row=2, col=1)

    fig.update_layout(
        height=540,
        margin=dict(l=40, r=40, t=50, b=30),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        plot_bgcolor=COLOR_BG,
        paper_bgcolor=COLOR_BG,
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False)
    fig.update_yaxes(
        title_text="ICE State",
        tickmode="array",
        tickvals=[0, 1],
        ticktext=["OFF", "ON"],
        range=[-0.15, 1.15],
        row=1, col=1
    )
    fig.update_yaxes(title_text="Power (kW)", row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)

    return fig


def create_drive_cycle_speed_plot(df: pd.DataFrame) -> go.Figure:
    """
    Supporting Plot: Vehicle speed and longitudinal acceleration profile with phase tagging.
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.60, 0.40],
        subplot_titles=("WLTP Vehicle Speed (km/h)", "Longitudinal Acceleration (m/s²)")
    )

    fig.add_trace(
        go.Scatter(
            x=df["time_s"],
            y=df["speed_kmh"],
            mode="lines",
            name="Vehicle Speed (km/h)",
            line=dict(color="#29B6F6", width=2.4),
            fill="tozeroy",
            fillcolor="rgba(41, 182, 246, 0.12)",
            customdata=df["phase"],
            hovertemplate="<b>Time</b>: %{x:.0f} s<br><b>Speed</b>: %{y:.1f} km/h<br><b>Phase</b>: %{customdata}<extra></extra>",
        ),
        row=1, col=1
    )

    # 40 km/h baseline rule threshold indicator
    fig.add_hline(
        y=40.0,
        line_dash="dash",
        line_color="#FFCA28",
        line_width=1.5,
        annotation_text="Rule Threshold: 40 km/h",
        annotation_position="top left",
        row=1, col=1
    )

    # Acceleration trace
    fig.add_trace(
        go.Scatter(
            x=df["time_s"],
            y=df["acceleration_ms2"],
            mode="lines",
            name="Acceleration (m/s²)",
            line=dict(color="#AB47BC", width=1.5),
            hovertemplate="<b>Time</b>: %{x:.0f} s<br><b>Accel</b>: %{y:.2f} m/s²<extra></extra>",
        ),
        row=2, col=1
    )
    fig.add_hline(y=0.0, line_color="rgba(200,200,200,0.3)", row=2, col=1)

    fig.update_layout(
        height=420,
        margin=dict(l=40, r=40, t=50, b=30),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        plot_bgcolor=COLOR_BG,
        paper_bgcolor=COLOR_BG,
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False)
    fig.update_yaxes(title_text="Speed (km/h)", row=1, col=1)
    fig.update_yaxes(title_text="Accel (m/s²)", row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)

    return fig
