# test.py
from stable_baselines3 import PPO
from envs.guide_dog_env import GuideDogEnv
from visualize import render_storyboard_v2
import time

env = GuideDogEnv()
model = PPO.load("ppo_from_scratch")

obs, _ = env.reset()
done = False
step = 0
frames = []

while not done and step < 12:
    grid_copy = env.grid.copy()
    frames.append((env.grid.copy(), env.agent_pos, env.user_pos, step, reward, terminated or truncated))
    action, _ = model.predict(obs)
    action = int(action)
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    step += 1

render_storyboard_v2(frames)
