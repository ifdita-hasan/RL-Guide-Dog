# RL-Guide-Dog

This is a grid world envrionment made using gymnasium, where a guide dog agent helps a simulated user to reach their goal while avoiding obstacles. 

![alt text](image.png)

# File Structure
rl-guide-dog/
├── envs/
│   └── guide_dog_env.py   # Gymnasium environment definition
├── train.py               # Script to train the PPO agent
├── test.py                # Script to run and collect episode frames
├── visualize.py           # Storyboard rendering utilities
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation

# How to run

Train a PPO agent on the GuideDog environment:
"""python train.py"""
A trained model will be saved as ppo_guide_dog.zip.

Run an episode and display a storyboard using:
"""python test.py"""
This generates a visual grid-by-grid plot showing agent, user, goal, rewards, and terminal status.

# License
MIT License