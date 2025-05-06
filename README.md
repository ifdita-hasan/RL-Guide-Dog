# RL-Guide-Dog

This is a reinforcement learning-based navigation system inspired by guide dogs, where a guide dog agent helps a simulated user to reach their goal while avoiding obstacles. The grid world envrionment was made using gymnasium.

![alt text](image.png)

# File Structure
```bash
rl-guide-dog/
├── envs/
│   └── guide_dog_env.py   # Gymnasium environment definition
├── train.py               # Script to train the PPO agent
├── test.py                # Script to run and collect episode frames
├── visualize.py           # Storyboard rendering utilities
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```
# How to run

Train a PPO agent on the GuideDog environment using:
```bash
python train.py
```
A trained model will be saved as ppo_guide_dog.zip.

Then run an episode and display a storyboard using:
```bash
python test.py
```
This generates a visual grid-by-grid plot showing agent, user, goal, rewards, and terminal status.

To visualize the rollouts on the streamlit app, run:
```bash
streamlit run app.py
```

# Proximal Policy Optimization
The algorithm used to train the agent is called Proximal Policy Optimization, an on-policy actor‑critic method which learns a policy (actor) and a value function (critic) simultaneously. It optimizes a Clipped Surrogate Objective to prevent excessive divergence.
# License
MIT License
