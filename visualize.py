import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle\


def render_storyboard_v2(frames, columns=4, figsize=(16, 8)):
    """
    Enhanced storyboard rendering with:
    - agent-user distance
    - per-step reward
    - terminal status
    - final frame border highlight
    Each frame: (grid, agent_pos, user_pos, step, reward, done)
    """
    rows = int(np.ceil(len(frames) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=figsize)
    axes = axes.flatten()

    emoji_map = {0: " ", 1: "X", 2: "A", 3: "U", 4: "G"}
    color_map = {0: "white", 1: "#666666", 2: "#FFD700", 3: "#87CEEB", 4: "#90EE90"}

    for idx, frame in enumerate(frames):
        grid, agent_pos, user_pos, step, reward, done = frame
        ax = axes[idx]
        n_rows, n_cols = grid.shape
        ax.set_xlim(0, n_cols)
        ax.set_ylim(0, n_rows)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Step {step}", fontsize=10)

        # Draw grid
        for y in range(n_rows):
            for x in range(n_cols):
                val = grid[y, x]
                color = color_map.get(val, "white")
                symbol = emoji_map.get(val, " ")
                ax.add_patch(Rectangle((x, y), 1, 1, color=color, ec='black'))
                ax.text(x + 0.5, y + 0.5, symbol, ha='center', va='center', fontsize=12, fontweight='bold')

        # Annotate with stats
        dist = abs(agent_pos[0] - user_pos[0]) + abs(agent_pos[1] - user_pos[1])
        ax.text(0, -0.3, f"Dist: {dist} | R: {round(reward,2)} | Done: {done}", fontsize=7)

        # Highlight last frame
        if idx == len(frames) - 1:
            color = "green" if done else "red"
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)

        ax.invert_yaxis()

    for ax in axes[len(frames):]:
        ax.axis('off')

    plt.tight_layout()
    return fig
