# test.py
import torch
from envs.guide_dog_env import GuideDogEnv
from visualize import render_storyboard_v2
from ppo import ActorCritic  # make sure class is importable

# Environment
env = GuideDogEnv()
obs_dim = env.observation_space.shape[0]
n_actions = env.action_space.n
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your custom PPO model
model = ActorCritic(obs_dim, n_actions).to(device)
model.load_state_dict(torch.load("ppo_from_scratch.pth", map_location=device))
model.eval()

obs, _ = env.reset()
done = False
step = 0
frames = []

while not done and step < 12:
    state_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits, _ = model(state_tensor)
        action = torch.distributions.Categorical(logits=logits).sample().item()

    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

    frames.append((env.grid.copy(), env.agent_pos, env.user_pos, step, reward, done))
    step += 1

render_storyboard_v2(frames)
