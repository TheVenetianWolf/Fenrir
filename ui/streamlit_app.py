# ui/streamlit_app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from services.scenarios import simple_pn_intercept

st.set_page_config(page_title="FENRIR Dashboard", page_icon="🐺", layout="centered")

st.title("🛰️ FENRIR — Flight & Engagement Kinematics Demo")
st.markdown("### Proportional Navigation Intercept")

# Sidebar controls
st.sidebar.header("Scenario Parameters")
N = st.sidebar.slider("Navigation constant (N)", 1.0, 6.0, 3.0, 0.5)
amax = st.sidebar.slider("Max lateral acceleration (m/s²)", 10.0, 100.0, 50.0, 5.0)
dt = st.sidebar.slider("Time step (s)", 0.01, 0.2, 0.05, 0.01)
T = st.sidebar.slider("Simulation time (s)", 5.0, 60.0, 30.0, 5.0)

# Button to start
if st.button("Run Simulation"):
    world = simple_pn_intercept()
    missile = world.entities[0]
    target = world.entities[1]
    missile.guidance.N = N
    missile.amax = amax

    missile_path = []
    target_path = []

    steps = int(T / dt)
    for _ in range(steps):
        world.step(dt)
        missile_path.append(missile.state.r.copy())
        target_path.append(target.state.r.copy())

    missile_path = np.array(missile_path)
    target_path = np.array(target_path)

    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(target_path[:, 0], target_path[:, 1], 'r--', label='Target path')
    ax.plot(missile_path[:, 0], missile_path[:, 1], 'b-', label='Missile path')
    ax.scatter(target_path[-1, 0], target_path[-1, 1], c='r', marker='x', s=80, label='Target final')
    ax.scatter(missile_path[-1, 0], missile_path[-1, 1], c='b', marker='o', s=80, label='Missile final')
    ax.set_xlabel('X position (m)')
    ax.set_ylabel('Y position (m)')
    ax.legend()
    ax.set_title("Missile Intercept Trajectories")

    st.pyplot(fig)
else:
    st.info("Adjust parameters in the sidebar and click **Run Simulation**.")
