"""
Streamlit Dashboard: Parallel Hybrid Energy Management System (EMS) Simulation.
Now featuring side-by-side comparison of Rule-Based Baseline vs. Predictive ML A-ECMS.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from models.vehicle import VehicleConfig
from cycle.wltp import generate_synthetic_wltp
from ems.rule_based import RuleBasedEMS
from ems.predictive_a_ecms import PredictiveAECMSStrategy
from simulation.engine import HybridSimulationEngine, SimulationMetrics
from ui.plots import (
    create_fuel_consumption_plot,
    create_soc_trajectory_plot,
    create_ice_pattern_and_power_split_plot,
    create_drive_cycle_speed_plot,
)

st.set_page_config(
    page_title="Parallel HEV EMS Comparison",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .metric-title {
        font-size: 0.85rem;
        color: #9E9E9E;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #FFFFFF;
        margin: 0;
    }
    .metric-delta {
        font-size: 0.85rem;
        margin-top: 4px;
    }
    .highlight-card {
        background: linear-gradient(135deg, rgba(0,210,106,0.15), rgba(0,210,106,0.05));
        border: 1px solid rgba(0,210,106,0.5);
    }
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    .badge-primary { background: rgba(33, 150, 243, 0.2); color: #64B5F6; border: 1px solid #2196F3; }
    .badge-success { background: rgba(0, 210, 106, 0.2); color: #69F0AE; border: 1px solid #00D26A; }
    .badge-warning { background: rgba(255, 87, 34, 0.2); color: #FF8A65; border: 1px solid #FF5722; }
    .badge-info { background: rgba(0, 188, 212, 0.2); color: #80DEEA; border: 1px solid #00BCD4; }
</style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/hybrid-car.png", width=70)
    st.title("EMS Control Center")
    st.markdown("---")

    st.subheader("1. Drive Cycle (WLTP)")
    cycle_option = st.selectbox(
        "Drive Cycle Profile",
        [
            "Full WLTP (1800s - 23.3 km)",
            "Low Phase - Urban (589s)",
            "Medium Phase - Suburban (433s)",
            "High Phase - Rural (455s)",
            "Extra-High Phase - Highway (323s)",
        ],
        index=0,
    )
    cycle_phase_map = {
        "Full WLTP (1800s - 23.3 km)": "Full WLTP",
        "Low Phase - Urban (589s)": "Low (Urban)",
        "Medium Phase - Suburban (433s)": "Medium (Suburban)",
        "High Phase - Rural (455s)": "High (Rural)",
        "Extra-High Phase - Highway (323s)": "Extra-High (Highway)",
    }
    selected_phase = cycle_phase_map[cycle_option]

    st.markdown("---")
    st.subheader("2. Battery State")
    init_soc_input = st.slider("Initial Battery SoC (%)", 30.0, 80.0, 60.0, 5.0)

    st.markdown("---")
    st.subheader("3. Baseline Parameters")
    speed_thresh = st.slider("EV Speed Threshold (km/h)", 20.0, 65.0, 40.0, 5.0)
    
    st.markdown("---")
    st.subheader("4. A-ECMS Parameters")
    s_init = st.slider("Initial Equivalence Factor s₀", 1.8, 4.0, 2.7, 0.1)


@st.cache_data(show_spinner=False)
def run_comparative_simulations(phase_key: str, initial_soc: float, v_th: float, s0: float):
    vehicle_cfg = VehicleConfig()
    cycle_data = generate_synthetic_wltp(phase_key)

    # 1. Baseline
    strat_base = RuleBasedEMS(v_th, 1.2, True, True)
    engine_base = HybridSimulationEngine(vehicle_cfg)
    df_base, metrics_base = engine_base.run(cycle_data, strat_base, initial_soc=initial_soc / 100.0)

    # 2. Predictive A-ECMS
    strat_ml = PredictiveAECMSStrategy(s_initial=s0)
    engine_ml = HybridSimulationEngine(vehicle_cfg)
    df_ml, metrics_ml = engine_ml.run(cycle_data, strat_ml, initial_soc=initial_soc / 100.0)

    return df_base, metrics_base, df_ml, metrics_ml

with st.spinner("Simulating Baseline vs Predictive A-ECMS side-by-side..."):
    df_base, met_base, df_ml, met_ml = run_comparative_simulations(
        selected_phase, init_soc_input, speed_thresh, s_init
    )

vehicle_instance = VehicleConfig()

st.title("⚡ Parallel Hybrid Energy Management System (EMS)")
st.markdown("Side-by-side comparison of **Rule-Based Baseline** vs **Predictive ML A-ECMS** over synthetic WLTP.")

# Fuel Saved Metric
fuel_saved_pct = 0.0
if met_base.total_fuel_consumed_liters > 0:
    fuel_saved_pct = ((met_base.total_fuel_consumed_liters - met_ml.total_fuel_consumed_liters) / met_base.total_fuel_consumed_liters) * 100.0

st.markdown(f"""
<div class="metric-card highlight-card" style="margin-bottom: 20px;">
    <div class="metric-title" style="color: #69F0AE; font-size: 1.1rem;">Fuel Saved by Predictive A-ECMS vs Baseline</div>
    <div class="metric-value" style="font-size: 2.8rem; color: #69F0AE;">{fuel_saved_pct:.2f}%</div>
    <div class="metric-delta">Baseline: {met_base.fuel_consumption_l_per_100km:.2f} L/100km &nbsp;&nbsp;|&nbsp;&nbsp; A-ECMS: {met_ml.fuel_consumption_l_per_100km:.2f} L/100km</div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# MAIN DASHBOARD TABS
# -----------------------------------------------------------------------------
tab_compare, tab_physics, tab_modes, tab_data = st.tabs([
    "📈 Side-by-Side Comparison",
    "🚗 Kinematics & Physics",
    "⚡ Energy Flow Summaries",
    "📋 Telemetry & Export",
])

with tab_compare:
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("### 🟦 Rule-Based Baseline")
        st.markdown("---")
        st.plotly_chart(create_fuel_consumption_plot(df_base, met_base), use_container_width=True)
        st.plotly_chart(create_soc_trajectory_plot(df_base, met_base, vehicle_instance), use_container_width=True)
        st.plotly_chart(create_ice_pattern_and_power_split_plot(df_base, vehicle_instance), use_container_width=True)
        
    with c2:
        st.markdown("### 🟩 Predictive A-ECMS")
        st.markdown("---")
        st.plotly_chart(create_fuel_consumption_plot(df_ml, met_ml), use_container_width=True)
        st.plotly_chart(create_soc_trajectory_plot(df_ml, met_ml, vehicle_instance), use_container_width=True)
        st.plotly_chart(create_ice_pattern_and_power_split_plot(df_ml, vehicle_instance), use_container_width=True)

with tab_physics:
    fig_cycle = create_drive_cycle_speed_plot(df_base)
    st.plotly_chart(fig_cycle, use_container_width=True)

with tab_modes:
    st.subheader("Energy Balance Comparison")
    mc1, mc2 = st.columns(2)
    
    def plot_energy_bar(met: SimulationMetrics, title: str):
        total_fuel_kwh = (met.total_fuel_energy_mj / 3.6)
        ice_mechanical_kwh = total_fuel_kwh * 0.35 # Approx efficiency for bar chart
        flows_df = pd.DataFrame({
            "Component": ["Traction Work", "Regen Recup.", "Friction Brake"],
            "Energy (kWh)": [
                met.total_traction_energy_kwh,
                met.total_regen_energy_kwh,
                met.friction_brake_energy_kwh,
            ]
        })
        fig_flows = px.bar(
            flows_df, x="Component", y="Energy (kWh)", color="Component", title=title,
            color_discrete_sequence=["#2196F3", "#00D26A", "#9C27B0"],
        )
        fig_flows.update_layout(height=350, showlegend=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        return fig_flows
        
    with mc1:
        st.plotly_chart(plot_energy_bar(met_base, "Baseline Energy Flow"), use_container_width=True)
    with mc2:
        st.plotly_chart(plot_energy_bar(met_ml, "A-ECMS Energy Flow"), use_container_width=True)

with tab_data:
    st.subheader("Telemetry Data (A-ECMS)")
    st.dataframe(df_ml.head(100), use_container_width=True)
