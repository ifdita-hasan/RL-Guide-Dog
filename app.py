
import streamlit as st
import numpy as np
from envs.guide_dog_env import GuideDogEnv
from stable_baselines3 import PPO
from visualize import render_storyboard_v2

st.set_page_config(page_title="RL Guide Dog Simulator", layout="wide")
st.title("🦮 RL Guide Dog Simulator")
st.markdown(
    "Simulates a trained RL agent (guide dog) navigating a grid with a visually impaired user. "
    "Built with Gymnasium + Stable Baselines3 + Streamlit."
)

# Sidebar controls
st.sidebar.header("Environment Settings")
grid_size = 6  
num_obstacles = st.sidebar.slider("Number of obstacles", min_value=3, max_value=10, value=5)
max_steps = st.sidebar.slider("Max steps per rollout", min_value=5, max_value=30, value=12)

# Load model
@st.cache_resource
def load_model():
    return PPO.load("ppo_guide_dog")

model = load_model()

# Initialize environment
env = GuideDogEnv(grid_size=grid_size, num_obstacles=num_obstacles)
obs, _ = env.reset()
frames = []
done = False
step = 0

# Run rollout
if st.button("Run Simulation"):
    with st.spinner("Simulating..."):
        while not done and step < max_steps:
            grid_copy = env.grid.copy()
            # frames.append((grid_copy, env.agent_pos, env.user_pos, step))
            
            action, _ = model.predict(obs)
            action = int(action)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            frames.append((env.grid.copy(), env.agent_pos, env.user_pos, step, reward, done))
            step += 1

        st.success("Simulation complete!")
        # st.pyplot(render_storyboard(frames, columns=4, figsize=(16, 4 * (len(frames) // 4 + 1))))
        fig = ren