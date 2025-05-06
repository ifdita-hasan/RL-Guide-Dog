from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from envs.guide_dog_env import GuideDogEnv

env = GuideDogEnv()
check_env(env)  # Check if the environment follows the Gym API

# Train the model
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

# Save the model
model.save("ppo_guide_dog")