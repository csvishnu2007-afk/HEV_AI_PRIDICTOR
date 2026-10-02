# Parallel Hybrid Vehicle Energy Management System (EMS)

An interactive, physics-based simulation dashboard built in Python and Streamlit modeling a **Rule-Based Energy Management System (EMS)** for a parallel hybrid electric vehicle (P2/parallel HEV architecture) over the **WLTP Class 3** standard drive cycle.

---

## Strict Powertrain Parameters

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Vehicle Mass ($m$)** | **1500 kg** | Total test mass (curb + driver + equipment) |
| **ICE Peak Power ($P_{\text{ICE,max}}$)** | **100 kW** | Internal Combustion Engine maximum mechanical power |
| **Electric Motor Peak Power ($P_{\text{EM,max}}$)** | **45 kW** | Traction Motor / Generator peak power (motoring & regen) |
| **Battery Pack Nominal Capacity ($E_{\text{bat}}$)** | **1.2 kWh** | High-Voltage traction battery (charge-sustaining HEV) |

---

## Longitudinal Kinematic Physics

Required traction power at the wheels is calculated from fundamental road-load and inertial equations:

$$F_{\text{traction}}(t) = F_{\text{acc}}(t) + F_{\text{aero}}(t) + F_{\text{roll}}(t)$$

1. **Aerodynamic Drag Force**:
   $$F_{\text{aero}}(t) = \frac{1}{2} \rho_{\text{air}} C_d A_f v(t)^2$$
   - $\rho_{\text{air}} = 1.225 \text{ kg/m}^3$, $C_d = 0.28$, $A_f = 2.22 \text{ m}^2$

2. **Tire Rolling Resistance**:
   $$F_{\text{roll}}(t) = m g C_{rr} \cos(\theta) \cdot \tanh\left(\frac{v(t)}{v_{\text{threshold}}}\right)$$
   - $C_{rr} = 0.0095$, $g = 9.81 \text{ m/s}^2$, smooth zero-speed cutoff

3. **Inertial Force (including Rotating Inertia)**:
   $$F_{\text{acc}}(t) = m \cdot \delta \cdot a(t) \quad (\delta = 1.05)$$

4. **Power Demand at Powertrain Shaft**:
   $$P_{\text{wheel}}(t) = F_{\text{traction}}(t) \cdot v(t)$$
   $$P_{\text{demand}}(t) = \begin{cases} \frac{P_{\text{wheel}}(t)}{\eta_{\text{trans}}} & \text{if } P_{\text{wheel}} \ge 0 \quad (\text{motoring}) \\ P_{\text{wheel}}(t) \cdot \eta_{\text{trans}} & \text{if } P_{\text{wheel}} < 0 \quad (\text{braking}) \end{cases}$$

---

## Baseline Rule-Based EMS Logic

- **Low Speed ($\mathbf{v < 40\text{ km/h}}$)**:
  - Operate in pure **Electric Motor (EV Mode)** ($P_{\text{ICE}} = 0$, $P_{\text{EM}} = P_{\text{demand}}$).
  - *Override*: If acceleration demand is high ($a \ge 1.2\text{ m/s}^2$), battery is depleted ($\text{SoC} \le 22\%$), or demand exceeds $45\text{ kW}$, engage ICE immediately.
- **Medium & High Speeds ($\mathbf{v \ge 40\text{ km/h}}$) OR High Acceleration**:
  - Engage **Internal Combustion Engine (ICE)**.
  - *Load Point Shifting*: When $\text{SoC} < 60\%$, ICE runs at higher efficiency load and uses excess torque to recharge battery via EM in generator mode.
  - *Torque Boost*: When $P_{\text{demand}} > 100\text{ kW}$, EM provides up to $45\text{ kW}$ boost assist.
- **Deceleration / Braking ($\mathbf{P_{\text{demand}} < 0}$)**:
  - Fuel cutoff on ICE ($P_{\text{ICE}} = 0$).
  - Maximize **Regenerative Braking** via EM up to $-45\text{ kW}$.
  - Any remaining deceleration demand is dissipated by mechanical friction brakes.
- **Standstill ($\mathbf{v = 0}$)**:
  - Automatic Start-Stop system shuts off ICE unless battery needs emergency charge.

---

## Dynamic Streamlit Plots

The dashboard provides the three specific dynamic plots:
1. **Fuel Consumption (L/100km)**: Dynamic cumulative fuel economy trajectory with instantaneous engine fuel rate dual-panel.
2. **Battery State-of-Charge (SoC) Trajectory**: Continuous battery percentage curve with safe $20\%-85\%$ operational band and charge-sustaining target line.
3. **ICE On/Off Pattern & Powertrain Power Split**: Synchronized binary ICE state (On/Off) strip chart and complete multi-component power split ($P_{\text{demand}}$, $P_{\text{ICE}}$, $P_{\text{EM}}$, $P_{\text{brake}}$).

---

## Modular Architecture for Future ML / A-ECMS

The controller follows a **Strategy Pattern** defined in `ems/base.py`:
- `BaseEMS`: Abstract class with standard `decide_power_split(ems_input, vehicle) -> EMSOutput`.
- Both `RuleBasedEMS` and `AECMSStrategy` implement this interface.
- In subsequent phases, predictive machine learning models or Deep RL policies (e.g. PyTorch / RLlib) can be dropped in without altering any simulation or UI code.

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run unit tests
python -m pytest tests -v

# 3. Launch the Streamlit dashboard
streamlit run app.py
```
