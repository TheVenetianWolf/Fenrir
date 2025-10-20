# ui/streamlit_app.py
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import io
import base64
import matplotlib
import math
from matplotlib.patches import Wedge, Circle
from streamlit.components.v1 import html
matplotlib.use("Agg")   # make sure we're on a non-interactive backend


st.set_page_config(page_title="FENRIR Dashboard", page_icon="🐺", layout="wide")
st.title("🐺 FENRIR — Real-Time Guidance Demo")

# ---- imports from your project ----
from services.scenarios import simple_pn_intercept

# try to import evasive scenario if you've added it; otherwise we'll fallback
def _get_world_builder(name: str):
    if name == "PN Intercept":
        return simple_pn_intercept
    elif name == "Evasive Target":
        try:
            from services.scenarios import pn_vs_evasive
            return pn_vs_evasive
        except Exception:
            # fallback if not implemented yet
            st.warning("Evasive Target not found. Falling back to PN Intercept.")
            return simple_pn_intercept
    else:
        return simple_pn_intercept

# ---------- helpers ----------
def los_metrics(missile, target):
    r = target.state.r - missile.state.r
    v_rel = target.state.v - missile.state.v
    rng = float(np.linalg.norm(r) + 1e-9)
    los = r / rng
    # 2D scalar z-component of cross(los, v_rel)/rng
    lambda_dot = float(
        np.cross(np.append(los, 0.0), np.append(v_rel, 0.0))[2] / rng
    )
    v_closing = float(-np.dot(v_rel, los))
    return rng, lambda_dot, v_closing

def reset_sim(scenario: str, N=3.0, amax=50.0, dt=0.05, T=30.0, hit_radius=10.0):
    world = _get_world_builder(scenario)()
    # expect missile = entities[0], target = entities[1]
    missile = world.entities[0]
    target = world.entities[1]
    # set UI params
    if hasattr(missile, "guidance"):
        missile.guidance.N = N
    if hasattr(missile, "amax"):
        missile.amax = amax

    st.session_state.world = world
    st.session_state.dt = dt
    st.session_state.T = T
    st.session_state.steps = int(T / dt)
    st.session_state.i = 0
    st.session_state.running = False
    st.session_state.hit = None
    st.session_state.hit_radius = hit_radius
    st.session_state.missile_path = []
    st.session_state.target_path = []
    st.session_state.telemetry = []  # dicts per step
    st.session_state.radar_echoes = []  # fading blips for radar

def _angle_wrap(rad: float) -> float:
    return (rad + np.pi) % (2*np.pi) - np.pi

def draw_radar(ax, missile, target, t, echoes,
               range_max=5000.0, sweep_rate_rps=0.35, fov_deg=70.0):
    """Missile-centric radar with rotating sweep and fading echoes."""
    ax.set_facecolor((0.03, 0.07, 0.09))
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-range_max, range_max)
    ax.set_ylim(-range_max, range_max)
    ax.axis('off')

    # Range rings + bearings
    rings = 4
    for k in range(1, rings+1):
        r = range_max * k / rings
        ax.add_patch(Circle((0,0), r, fill=False, lw=1, ec=(1,1,1,0.12)))
    for deg in range(0, 360, 30):
        rad = math.radians(deg)
        ax.plot([0, range_max*math.cos(rad)], [0, range_max*math.sin(rad)],
                lw=0.6, c=(1,1,1,0.08))

    # Sweep wedge
    phi = (t * 2*np.pi * sweep_rate_rps) % (2*np.pi)
    half = math.radians(fov_deg) / 2.0
    wedge = Wedge((0,0), range_max,
                  math.degrees(phi - half), math.degrees(phi + half),
                  facecolor=(0.2, 1.0, 0.6, 0.08),
                  edgecolor=(0.2, 1.0, 0.6, 0.35), lw=1.2)
    ax.add_patch(wedge)

    # Crosshair at center (missile)
    ax.add_patch(Circle((0,0), 45, fill=False, ec=(0.6,1,0.9,0.6), lw=1.2))
    ax.plot([-60, 60], [0, 0], c=(0.6,1,0.9,0.4), lw=1.0)
    ax.plot([0, 0], [-60, 60], c=(0.6,1,0.9,0.4), lw=1.0)

    # Target relative position
    rel = target.state.r - missile.state.r
    r = float(np.linalg.norm(rel))
    theta = math.atan2(rel[1], rel[0])

    # Current bright blip (if in range)
    if r <= range_max:
        ax.scatter([rel[0]], [rel[1]], s=30, c=[(0.4,1.0,0.4,0.95)], zorder=5)
        ax.scatter([rel[0]], [rel[1]], s=120, c=[(0.4,1.0,0.4,0.25)], zorder=4)

    # Drop an echo when sweep passes the target
    inside = (r <= range_max) and (abs(_angle_wrap(theta - phi)) <= half)
    if inside:
        echoes.append({"x": float(rel[0]), "y": float(rel[1]), "ttl": 1.0})

    # Fade echoes
    for e in echoes:
        e["ttl"] *= 0.92
    echoes[:] = [e for e in echoes if e["ttl"] > 0.05]
    for e in echoes:
        ax.scatter([e["x"]], [e["y"]], s=40,
                   c=[(0.3, 1.0, 0.6, 0.35 * e["ttl"])])

    # HUD text
    ax.text(-range_max*0.98, -range_max*0.98, f"Rmax {int(range_max)} m",
            color=(0.8,0.95,0.95,0.5), fontsize=9)

    return echoes


# ---------- sidebar controls ----------
st.sidebar.header("Scenario & Parameters")
scenario = st.sidebar.selectbox("Scenario", ["PN Intercept", "Evasive Target"])

N = st.sidebar.slider("Navigation constant N", 1.0, 6.0, 3.0, 0.5)
amax = st.sidebar.slider("Max lateral accel (m/s²)", 10.0, 100.0, 50.0, 5.0)
dt = st.sidebar.slider("Time step dt (s)", 0.01, 0.2, 0.05, 0.01)
T = st.sidebar.slider("Simulation time T (s)", 5.0, 90.0, 30.0, 5.0)
hit_radius = st.sidebar.slider("Hit radius (m)", 1.0, 50.0, 10.0, 1.0)
fps = st.sidebar.slider("Playback FPS", 5, 60, 20, 1)

colA, colB, colC = st.sidebar.columns(3)
if colA.button("Reset"):
    reset_sim(scenario, N, amax, dt, T, hit_radius)

if "world" not in st.session_state:
    reset_sim(scenario, N, amax, dt, T, hit_radius)

def start_or_resume():
    missile = st.session_state.world.entities[0]
    # apply current knobs to the existing missile
    if hasattr(missile, "guidance"):
        missile.guidance.N = N
    if hasattr(missile, "amax"):
        missile.amax = amax
    st.session_state.dt = dt
    st.session_state.steps = int(T / dt)
    st.session_state.running = True

if colB.button("Start/Resume"):
    start_or_resume()
if colC.button("Pause"):
    st.session_state.running = False

# ---------- main layout ----------
top, bottom = st.columns([2, 1])
plot_area = top.container()
readout   = top.container()
charts    = bottom.container()

# ---------- simulation step(s) ----------
if (
    st.session_state.get("running")
    and st.session_state.i < st.session_state.steps
    and st.session_state.hit is None
):
    # advance one physics frame per render for smooth animation
    frames_per_cycle = 1
    for _ in range(frames_per_cycle):
        world = st.session_state.world
        missile, target = world.entities[0], world.entities[1]

        # optional custom target behaviour hook (if you added it in World.step)
        # handled inside your engine if present

        # step world
        world.step(st.session_state.dt)

        # record paths
        st.session_state.missile_path.append(missile.state.r.copy())
        st.session_state.target_path.append(target.state.r.copy())

        # telemetry (robust to missing fields)
        rng, los_rate, v_closing = los_metrics(missile, target)
        m_speed = float(np.linalg.norm(missile.state.v))
        a_cmd = float(getattr(missile, "last_a_cmd", 0.0))
        a_ach = float(getattr(missile, "last_a_achieved", 0.0))
        a_lat = float(getattr(missile, "last_a_lat", 0.0))

        st.session_state.telemetry.append(
            dict(
                t=float(world.t),
                range=float(rng),
                los_rate=float(los_rate),
                v_closing=float(v_closing),
                m_speed=m_speed,
                a_cmd=a_cmd,
                a_lat=a_lat,
                a_ach=a_ach,
                mx=float(missile.state.r[0]),
                my=float(missile.state.r[1]),
                tx=float(target.state.r[0]),
                ty=float(target.state.r[1]),
            )
        )

        # hit detection
        if rng <= st.session_state.hit_radius:
            st.session_state.hit = dict(t=float(world.t), range=float(rng))
            st.session_state.running = False
            break

        st.session_state.i += 1

# ---------- draw 2D top-down plot ----------
mp = np.array(st.session_state.missile_path) if st.session_state.missile_path else None
tp = np.array(st.session_state.target_path) if st.session_state.target_path else None

with plot_area:

    st.subheader(f"Top-Down Trajectories — {scenario}")

    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    drew_anything = False
    if tp is not None and len(tp) > 1:
        ax.plot(tp[:, 0], tp[:, 1], '--', label='Target path')
        ax.scatter(tp[-1, 0], tp[-1, 1], marker='x', s=80, label='Target now')
        drew_anything = True
    if mp is not None and len(mp) > 1:
        ax.plot(mp[:, 0], mp[:, 1], '-', label='Missile path')
        ax.scatter(mp[-1, 0], mp[-1, 1], s=80, label='Missile now')
        drew_anything = True
    if st.session_state.hit and mp is not None and len(mp) > 0:
        ax.scatter(mp[-1, 0], mp[-1, 1], s=120, facecolors='none',
                   edgecolors='k', linewidths=2, label='Hit')
        ax.add_artist(
            plt.Circle((mp[-1, 0], mp[-1, 1]), st.session_state.hit_radius,
                       fill=False, linestyle=':', linewidth=1)
        )
        drew_anything = True

    # sensible default view so first frames aren’t microscopic
    world = st.session_state.world
    missile, target = world.entities[0], world.entities[1]
    rel = target.state.r - missile.state.r
    dist = float(np.linalg.norm(rel))
    span = max(1000.0, dist * 1.2)
    cx = float((missile.state.r[0] + target.state.r[0]) / 2.0)
    cy = float((missile.state.r[1] + target.state.r[1]) / 2.0)
    ax.set_xlim(cx - span, cx + span)
    ax.set_ylim(cy - span, cy + span)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    if drew_anything:
        ax.legend(loc='best')

    # render to PNG bytes (robust across Streamlit versions)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    png_bytes = buf.getvalue()
    plt.close(fig)

    # Fixed-height HTML container to avoid collapsed render
    b64 = base64.b64encode(png_bytes).decode("ascii")
    html(
        f"""
        <div style="width:100%; height:560px; background:#0b0b0b10; border-radius:8px; overflow:hidden;">
        <img src="data:image/png;base64,{b64}"
            style="width:100%; height:100%; object-fit:contain; display:block;" />
        </div>
        """,
        height=580,   # slightly bigger to ensure full visibility
    )




# ---------- telemetry readouts ----------
with readout:
    world = st.session_state.world
    missile, target = world.entities[0], world.entities[1]
    rng, los_rate, v_closing = los_metrics(missile, target)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("t (s)", f"{world.t:6.2f}")
    col2.metric("Range (m)", f"{rng:8.1f}")
    col3.metric("LOS rate (rad/s)", f"{los_rate: .4f}")
    col4.metric("Closing speed (m/s)", f"{v_closing:7.1f}")
    col5.metric("Missile speed (m/s)", f"{np.linalg.norm(missile.state.v):7.1f}")

    if st.session_state.hit:
        st.success(f"Hit! t={st.session_state.hit['t']:.2f}s, range={st.session_state.hit['range']:.2f} m")

# ---------- radar panel ----------
with top.container():
    world = st.session_state.world
    missile, target = world.entities[0], world.entities[1]
    rng, _, _ = los_metrics(missile, target)
    rmax = max(2000.0, float(rng) * 1.2)

    st.subheader("Radar — Missile-Centric")

    fig_r, ax_r = plt.subplots(figsize=(6.5, 4.5))
    st.session_state.radar_echoes = draw_radar(
        ax_r, missile, target, world.t,
        st.session_state.get("radar_echoes", []),
        range_max=rmax, sweep_rate_rps=0.4, fov_deg=70.0
    )

    buf_r = io.BytesIO()
    fig_r.savefig(buf_r, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig_r)
    b64_r = base64.b64encode(buf_r.getvalue()).decode("ascii")

    html(
        f"""
        <div style="width:100%; height:380px; background:#0b0b0b10; border-radius:8px; overflow:hidden;">
          <img src="data:image/png;base64,{b64_r}"
               style="width:100%; height:100%; object-fit:contain; display:block;" />
        </div>
        """,
        height=400,
    )


# ---------- charts + export ----------
with charts:
    telem = pd.DataFrame(st.session_state.telemetry) if st.session_state.telemetry else pd.DataFrame()
    c1, c2 = st.columns(2)
    if not telem.empty:
        with c1:
            st.subheader("Range vs Time")
            st.line_chart(telem, x="t", y="range", height=220)
        with c2:
            st.subheader("LOS Rate vs Time")
            st.line_chart(telem, x="t", y="los_rate", height=220)

        # optional chart if dynamics expose a_cmd/a_ach
        if {"a_cmd", "a_ach"}.issubset(telem.columns):
            st.subheader("Commanded vs Achieved Lateral Accel")
            accel_df = telem[["t", "a_cmd", "a_ach"]].set_index("t")
            st.line_chart(accel_df, height=220)

        st.subheader("Export Telemetry")
        csv = telem.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv,
                           file_name="fenrir_telemetry.csv", mime="text/csv")
    else:
        st.info("No telemetry yet — press **Start/Resume**.")

# ---------- schedule next frame AFTER rendering ----------
if st.session_state.get("running") and st.session_state.i < st.session_state.steps and not st.session_state.hit:
    time.sleep(max(0.0, 1.0 / fps))  # throttle to target FPS
    st.rerun()
