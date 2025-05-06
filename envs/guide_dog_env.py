# envs/guide_dog_env.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import gymnasium as gym
from gymnasium import spaces

class GuideDogEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, grid_size=6, num_obstacles=5, render_mode=None):
        self.grid_size = grid_size
        self.num_obstacles = num_obstacles
        self.render_mode = render_mode
        self.EMPTY, self.OBSTACLE, self.AGENT, self.USER, self.GOAL = 0, 1, 2, 3, 4

        self.action_space = spaces.Discrete(5)  # up, down, left, right, stay
        self.observation_space = spaces.Box(
            low=0,
            high=4,
            shape=(self.grid_size * self.grid_size,),
            dtype=np.int32
        )

        self.steps = 0
        self.agent_pos = None
        self.user_pos = None
        self.goal = None
        self.grid = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.goal = self._random_empty()
        self.grid[self.goal] = self.GOAL

        for _ in range(self.num_obstacles):
            y, x = self._random_empty()
            self.grid[y, x] = self.OBSTACLE

        self.agent_pos = self._random_empty()
        self.grid[self.agent_pos] = self.AGENT

        self.user_pos = self._random_empty()
        self.grid[self.user_pos] = self.USER

        self.steps = 0
        obs = self._get_obs()
        return obs, {}

    def _random_empty(self):
        while True:
            y, x = np.random.randint(0, self.grid_size, size=2)
            if self.grid[y, x] == self.EMPTY:
                return (y, x)

    def _get_obs(self):
        return self.grid.flatten().astype(np.int32)

    def step(self, action):
        reward = -0.01
        terminated = False
        truncated = False

        move = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1), 4: (0, 0)}
        dy, dx = move[action]
        new_y = self.agent_pos[0] + dy
        new_x = self.agent_pos[1] + dx

        if self._valid(new_y, new_x):
            self.grid[self.agent_pos] = self.EMPTY
            self.agent_pos = (new_y, new_x)
            self.grid[self.agent_pos] = self.AGENT

        self._move_user()

        if self.user_pos == self.goal:
            reward = 5.0
            terminated = True
        elif self.grid[self.user_pos] == self.OBSTACLE:
            reward = -2.0
            terminated = True

        self.steps += 1
        if self.steps >= 100:
            truncated = True

        obs = self._get_obs().astype(np.int32)
        return obs, reward, terminated, truncated, {}

    def _valid(self, y, x):
        return 0 <= y < self.grid_size and 0 <= x < self.grid_size and self.grid[y, x] != self.OBSTACLE

    def _move_user(self):
        uy, ux = self.user_pos
        ay, ax = self.agent_pos
        dy = np.sign(ay - uy)
        dx = np.sign(ax - ux)
        new_y = uy + dy if 0 <= uy + dy < self.grid_size else uy
        new_x = ux + dx if 0 <= ux + dx < self.grid_size else ux

        if self.grid[new_y, new_x] != self.OBSTACLE:
            self.grid[self.user_pos] = self.EMPTY
            self.user_pos = (new_y, new_x)
            self.grid[self.user_pos] = self.USER

    def render(self):
        label_map = {
            self.EMPTY: " ", self.OBSTACLE: "X",
            self.AGENT: "A", self.USER: "U", self.GOAL: "G"
        }

        fig, ax = plt.subplots(figsize=(6, 6))
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                val = self.grid[y, x]
                ax.add_patch(patches.Rectangle((x, y), 1, 1, edgecolor='black', facecolor='white'))
                ax.text(x+0.5, y+0.5, label_map[val], ha='center', va='center', fontsize=16)
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.set_xticks([]), ax.set_yticks([])
        plt.gca().invert_yaxis()
        plt.title(f"Step {self.steps} | Agent at {self.agent_pos} | User at {self.user_pos}")
        plt.show()
