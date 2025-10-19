# ui/streamlit_app.py
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from services.scenarios import simple_pn_intercept

st.set_page_config(page_title="FENRIR Dashboard", page_icon="🐺", layout="wide")
st.title("🐺 FENRIR — Real-Time PN Intercept Demo")

# ---------- helpers ----------
def los_metrics(missile, target):
    r = target.state.r - missile.state.r
    v_rel = target.state.v - missile.state.v
    rng = float(np.linalg.norm(r) + 1e-9)
    los = r / rng
    # 2D scalar z-component of cross(los, v_rel)/rng
    lambda_dot = float(np.cross(np.append(los, 0.0), np.append(v_rel, 0.0))[2] / rng)
    v_closing = float(-np.dot(v_rel, los))
    return rng, lambda_dot, v_closing

def reset_sim(N=3.0, amax=50.0, dt=0.05, T=30.0, hit_radius=10.0):
    world = simple_pn_intercept()
    missile = world.entities[0]
    target = world.entities[1]
    missile.guidance.N = N
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
    st.session_state.telemetry = []  # dicts: t, range, los_rate, v_closing, mx, my, tx, ty, m_speed

# ---------- sidebar controls ----------
st.sidebar.header("Scenario Parameters")
N = st.sidebar.slider("Navigation constant N", 1.0, 6.0, 3.0, 0.5)
amax = st.sidebar.slider("Max lateral accel (m/s²)", 10.0, 100.0, 50.0, 5.0)
dt = st.sidebar.slider("Time step dt (s)", 0.01, 0.2, 0.05, 0.01)
T = st.sidebar.slider("Sim time T (s)", 5.0, 60.0, 30.0, 5.0)
hit_radius = st.sidebar.slider("Hit radius (m)", 1.0, 50.0, 10.0, 1.0)
fps = st.sidebar.slider("Playback FPS", 5, 60, 20, 1)

colA, colB, colC = st.sidebar.columns(3)
if colA.button("Reset"):
    reset_sim(N, amax, dt, T, hit_radius)
if "world" not in st.session_state:
    reset_sim(N, amax, dt, T, hit_radius)

def start_or_resume():
    # apply current knobs to the existing missile
    missile = st.session_state.world.entities[0]
    missile.guidance.N = N
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

# live plot container
plot_area = top.container()
# telemetry readouts
readout = top.container()
# charts & export
charts = bottom.container()

# ---------- simulation / playback loop ----------
if st.session_state.running and st.session_state.i < st.session_state.steps and st.session_state.hit is None:
    # run a few frames per UI render for smoothness
    frames_per_cycle = max(1, int(1))  # tweak if needed
    for _ in range(frames_per_cycle):
        world = st.session_state.world
        missile, target = world.entities[0], world.entities[1]

        # step world
        world.step(st.session_state.dt)

        # record paths
        st.session_state.missile_path.append(missile.state.r.copy())
        st.session_state.target_path.append(target.state.r.copy())

        # telemetry
        rng, los_rate, v_closing = los_metrics(missile, target)
        m_speed = float(np.linalg.norm(missile.state.v))
        st.session_state.telemetry.append(
            dict(
                t=world.t,
                range=rng,
                los_rate=los_rate,
                v_closing=v_closing,
                m_speed=m_speed,
                mx=float(missile.state.r[0]),
                my=float(missile.state.r[1]),
                tx=float(target.state.r[0]),
                ty=float(target.state.r[1]),
            )
        )

        # hit detection
        if rng <= st.session_state.hit_radius:
            st.session_state.hit = dict(t=world.t, range=rng)
            st.session_state.running = False
            break

        st.session_state.i += 1




# ---------- draw 2D top-down plot ----------
mp = np.array(st.session_state.missile_path) if st.session_state.missile_path else None
tp = np.array(st.session_state.target_path) if st.session_state.target_path else None

with plot_area:
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    if tp is not None and len(tp) > 1:
        ax.plot(tp[:, 0], tp[:, 1], '--', label='Target path')
        ax.scatter(tp[-1, 0], tp[-1, 1], marker='x', s=80, label='Target now')
    if mp is not None and len(mp) > 1:
        ax.plot(mp[:, 0], mp[:, 1], '-', label='Missile path')
        ax.scatter(mp[-1, 0], mp[-1, 1], s=80, label='Missile now')
    if st.session_state.hit:
        ax.scatter(mp[-1, 0], mp[-1, 1], s=120, facecolors='none', edgecolors='k', linewidths=2, label='Hit')
        ax.add_artist(plt.Circle((mp[-1, 0], mp[-1, 1]), st.session_state.hit_radius, fill=False, linestyle=':', linewidth=1))

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Top-Down Trajectories")
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    st.pyplot(fig)

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

        st.subheader("Export Telemetry")
        csv = telem.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv, file_name="fenrir_telemetry.csv", mime="text/csv")
    else:
        st.info("No telemetry yet — press **Start/Resume**.")

# --- schedule next frame AFTER rendering ---
if st.session_state.get("running") and st.session_state.i < st.session_state.steps and not st.session_state.hit:
    time.sleep(max(0.0, 1.0 / fps))  # throttle
    st.rerun()
