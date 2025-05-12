import asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

import streamlit as st
import numpy as np
from envs.guide_dog_env import GuideDogEnv
from visualize import render_storyboard_v2
import torch


# Must be first
st.set_page_config(page_title="🦮 RL Guide Dog Simulator", layout="wide")

st.title("🦮 RL Guide Dog Simulator")

st.markdown("""
Welcome to the **RL Guide Dog Simulator**!  
This simulation shows a trained RL agent acting as a guide dog 🐕‍🦺 helping a visually impaired user avoid obstacles and reach the goal 🎯  
Built with **Gymnasium + PPO + Streamlit** 💡
""")

# Sidebar controls
st.sidebar.header("🔧 Environment Settings")
grid_size = 6
num_obstacles = st.sidebar.slider("🧱 Number of obstacles", 3, 10, 5)
max_steps = st.sidebar.slider("🕒 Max steps per rollout", 5, 30, 12)
use_seed = st.sidebar.checkbox("🎲 Use fixed seed", value=False)

model_choice = st.sidebar.selectbox(
    "📦 Select model source",
    options=["PPO From SB3", "PPO From Scratch"])

st.sidebar.markdown("🚀 **Powered by PPO**")
st.sidebar.markdown("🐾 Agent learns to lead the user to safety!")

# Load model
@st.cache_resource
def load_model(choice, obs_dim, n_actions):
    if choice.startswith("From Scratch"):
        from ppo import ActorCritic  # ensure it's importable
        model = ActorCritic(obs_dim, n_actions)
        model.load_state_dict(torch.load("ppo_from_scratch.pth", map_location="cpu"))
        model.eval()
        return model
    else:
        from stable_baselines3 import PPO
        return PPO.load("ppo_guide_dog")

obs_dim = grid_size * grid_size
n_actions = 5
model = load_model(model_choice, obs_dim, n_actions)
is_custom = model_choice.startswith("From Scratch")

# Run simulation
if st.button("▶️ Run Simulation"):
    st.toast("Running PPO agent 🦮...", icon="🌀")

    env = GuideDogEnv(grid_size=grid_size, num_obstacles=num_obstacles)
    obs, _ = env.reset(seed=42 if use_seed else None)

    frames, done, step = [], False, 0
    total_reward = 0

    with st.spinner("Simulating agent behavior..."):
        while not done and step < max_steps:
            if is_custom:
                state_tensor = torch.tensor(obs, dtype=torch.float32)
                with torch.no_grad():
                    logits, _ = model(state_tensor)
                    action = torch.distributions.Categorical(logits=logits).sample().item()
            else:
                action, _ = model.predict(obs)

            obs, reward, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated
            total_reward += reward
            frames.append((env.grid.copy(), env.agent_pos, env.user_pos, step, reward, done))
            step += 1

        st.success("✅ Simulation complete!")
        if reward > 0:
            st.balloons()
            st.info("🎯 The user reached the goal safely!")
        elif reward < 0:
            st.warning("💥 The user hit an obstacle!")

        col1, col2, col3 = st.columns(3)
        col1.metric("Steps Taken", step)
        col2.metric("Final Reward", round(total_reward, 2))
        col3.metric("Done", "✅" if done else "🚧")

        fig = render_storyboard_v2(frames, columns=4, figsize=(16, 4 * ((len(frames)+3)//4)))
        st.pyplot(fig)

        st.subheader("🧾 Final Grid State")
        grid_str = "\n".join(
            [" ".join(str(cell) for cell in row) for row in env.grid]
        )
        st.code(grid_str, language="text")
else:
    st.info("⬅️ Adjust the settings on the left and click **Run Simulation** to watch the agent guide the user!")
