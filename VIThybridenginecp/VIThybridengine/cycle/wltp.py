"""
Synthetic WLTP Class 3 Drive Cycle Generator.
Generates realistic time vs. speed profiles adhering to UNECE Worldwide Harmonized Light Vehicles Test Procedure (WLTP):
- Phase 1: Low (Urban, 0-589s, max 56.5 km/h)
- Phase 2: Medium (Suburban, 589-1022s, max 76.6 km/h)
- Phase 3: High (Rural, 1022-1477s, max 97.4 km/h)
- Phase 4: Extra-High (Highway, 1477-1800s, max 131.3 km/h)
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np


@dataclass
class DriveCycle:
    """Drive cycle dataset with time, speed, phase tags, and metadata."""
    name: str
    time_s: np.ndarray
    speed_kmh: np.ndarray
    phases: np.ndarray             # Array of phase names per timestep
    duration_s: float
    total_distance_km: float
    max_speed_kmh: float
    avg_speed_kmh: float
    idle_time_pct: float


def _build_smooth_segment(
    v_start: float,
    v_end: float,
    duration: int
) -> np.ndarray:
    """Creates a smooth S-curve / sinusoidal transition between two velocities."""
    if duration <= 1:
        return np.array([v_end], dtype=np.float64)
    t = np.linspace(0, np.pi, duration)
    # Cosine easing: from v_start to v_end
    curve = v_start + 0.5 * (v_end - v_start) * (1.0 - np.cos(t))
    return curve


def generate_synthetic_wltp(phase_selection: str = "Full WLTP") -> DriveCycle:
    """
    Generates a high-fidelity synthetic WLTP Class 3 speed profile (time vs. speed in km/h).

    Parameters
    ----------
    phase_selection : str
        One of 'Full WLTP', 'Low (Urban)', 'Medium (Suburban)', 'High (Rural)', 'Extra-High (Highway)'.

    Returns
    -------
    DriveCycle
        Data structure containing time array, speed array, phase annotations, and cycle stats.
    """
    # Key nodes: (start_time, target_speed_kmh, hold_seconds)
    # Calibrated to recreate the 4 WLTP Class 3 phases and characteristic microtrips.

    # 1. Low Phase (Urban, 0 - 589 s): Stop and go, low speeds, frequent stops
    low_nodes = [
        # Stop 1: 0-15s idle
        (0.0, 15),
        # Microtrip 1: crawl to 18 km/h, stop
        (18.0, 10), (0.0, 12),
        # Microtrip 2: accel to 32 km/h, dip to 15, cruise at 28, stop
        (32.0, 14), (16.0, 8), (35.0, 16), (0.0, 18),
        # Microtrip 3: stop-start in traffic
        (22.0, 10), (0.0, 10),
        # Microtrip 4: climb to 45 km/h, decelerate to 25, climb to 56.5 km/h peak, slow cruise, stop
        (45.0, 18), (26.0, 12), (56.5, 18), (38.0, 14), (20.0, 10), (0.0, 24),
        # Microtrip 5: short urban pulse
        (25.0, 12), (0.0, 16),
        # Microtrip 6: up to 48 km/h and glide to stop before Medium phase
        (36.0, 15), (48.0, 16), (15.0, 8), (0.0, 20),
    ]

    # 2. Medium Phase (Suburban, 589 - 1022 s, ~433s duration):
    medium_nodes = [
        (25.0, 12), (45.0, 15), (0.0, 14),
        (35.0, 14), (55.0, 18), (28.0, 10), (62.0, 20), (0.0, 16),
        (42.0, 16), (68.0, 22), (50.0, 14), (76.6, 25), (45.0, 16), (20.0, 10), (0.0, 18),
    ]

    # 3. High Phase (Rural, 1022 - 1477 s, ~455s duration):
    high_nodes = [
        (35.0, 12), (60.0, 18), (80.0, 24), (55.0, 14),
        (88.0, 28), (70.0, 16), (97.4, 32), (82.0, 22), (65.0, 16), (30.0, 10), (0.0, 14),
    ]

    # 4. Extra-High Phase (Motorway/Highway, 1477 - 1800 s, ~323s duration):
    extra_high_nodes = [
        (45.0, 12), (80.0, 16), (110.0, 26), (95.0, 18),
        (125.0, 32), (131.3, 38), (115.0, 26), (90.0, 20), (50.0, 14), (0.0, 12),
    ]

    def synthesize_phase(nodes, target_duration: int) -> np.ndarray:
        raw_speeds = []
        current_v = 0.0

        for target_v, duration in nodes:
            # Half transition, half hold
            trans_dur = max(2, int(duration * 0.45))
            hold_dur = max(1, duration - trans_dur)

            seg_trans = _build_smooth_segment(current_v, target_v, trans_dur)
            seg_hold = np.full(hold_dur, target_v)

            raw_speeds.extend(seg_trans)
            raw_speeds.extend(seg_hold)
            current_v = target_v

        raw_array = np.array(raw_speeds, dtype=np.float64)
        # Resample to exact target duration
        if len(raw_array) != target_duration:
            x_old = np.linspace(0, 1, len(raw_array))
            x_new = np.linspace(0, 1, target_duration)
            resampled = np.interp(x_new, x_old, raw_array)
            return np.clip(resampled, 0.0, 140.0)
        return raw_array

    # Generate each phase with exact standard WLTP Class 3 phase durations:
    # Low: 589s, Medium: 433s, High: 455s, Extra-High: 323s -> Total 1800s
    v_low = synthesize_phase(low_nodes, 589)
    v_medium = synthesize_phase(medium_nodes, 433)
    v_high = synthesize_phase(high_nodes, 455)
    v_extra_high = synthesize_phase(extra_high_nodes, 323)

    # Assemble full cycle
    v_full = np.concatenate([v_low, v_medium, v_high, v_extra_high])
    # Append final zero stop if needed to make 1801 points (0 to 1800s inclusive)
    v_full = np.append(v_full, 0.0)
    time_full = np.arange(len(v_full), dtype=np.float64)

    phases_full = np.empty(len(v_full), dtype=object)
    phases_full[:589] = "Low (Urban)"
    phases_full[589:1022] = "Medium (Suburban)"
    phases_full[1022:1477] = "High (Rural)"
    phases_full[1477:] = "Extra-High (Highway)"

    if phase_selection == "Low (Urban)":
        mask = np.arange(590)
        v_sel = np.append(v_low, 0.0)
        t_sel = np.arange(len(v_sel), dtype=np.float64)
        p_sel = np.full(len(v_sel), "Low (Urban)")
        name = "WLTP Phase 1 - Low (Urban)"
    elif phase_selection == "Medium (Suburban)":
        v_sel = np.append(v_medium, 0.0)
        t_sel = np.arange(len(v_sel), dtype=np.float64)
        p_sel = np.full(len(v_sel), "Medium (Suburban)")
        name = "WLTP Phase 2 - Medium (Suburban)"
    elif phase_selection == "High (Rural)":
        v_sel = np.append(v_high, 0.0)
        t_sel = np.arange(len(v_sel), dtype=np.float64)
        p_sel = np.full(len(v_sel), "High (Rural)")
        name = "WLTP Phase 3 - High (Rural)"
    elif phase_selection == "Extra-High (Highway)":
        v_sel = np.append(v_extra_high, 0.0)
        t_sel = np.arange(len(v_sel), dtype=np.float64)
        p_sel = np.full(len(v_sel), "Extra-High (Highway)")
        name = "WLTP Phase 4 - Extra-High (Highway)"
    else:
        v_sel = v_full
        t_sel = time_full
        p_sel = phases_full
        name = "WLTP Class 3 (Full 1800s)"

    # Compute key stats
    speed_ms = v_sel / 3.6
    dt = 1.0
    dist_km = float(np.sum(speed_ms * dt) / 1000.0)
    dur_s = float(t_sel[-1])
    max_spd = float(np.max(v_sel))
    avg_spd = float(np.mean(v_sel))
    idle_pct = float(np.sum(v_sel < 0.5) / len(v_sel) * 100.0)

    return DriveCycle(
        name=name,
        time_s=t_sel,
        speed_kmh=v_sel,
        phases=p_sel,
        duration_s=dur_s,
        total_distance_km=dist_km,
        max_speed_kmh=max_spd,
        avg_speed_kmh=avg_spd,
        idle_time_pct=idle_pct,
    )
