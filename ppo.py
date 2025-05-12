# ppo_from_scratch

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from envs.guide_dog_env import GuideDogEnv

HORIZON        = 2048
EPOCHS         = 10
MINI_BATCH    = 64
GAMMA         = 0.99
GAE_LAMBDA    = 0.95
CLIP_EPS      = 0.2
LR            = 3e-4
ENT_COEF      = 0.01
VF_COEF       = 0.5
MAX_TIMESTEPS = 200_000
SEED          = 42
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define the Actor-Critic network
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_size=64):
        """
        Initialize the Actor-Critic network.
        Args:
            obs_dim (int): Dimension of the observation space.
            n_actions (int): Number of actions.
            hidden_size (int): Size of the hidden layers.
        """
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(obs_dim, hidden_size), nn.Tanh(), nn.Linear(hidden_size, hidden_size), nn.Tanh(),)
        self.policy_head = nn.Linear(hidden_size, n_actions)
        self.value_head  = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        Forward pass through the network.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, obs_dim).
        Returns:
            logits (torch.Tensor): Output tensor of shape (batch_size, n_actions).
            value (torch.Tensor): Output tensor of shape (batch_size, 1).
        """ 
        shared = self.shared(x)
        logits = self.policy_head(shared)
        value  = self.value_head(shared).squeeze(-1)
        return logits, value
    
    def get_action(self, state):
        """
        Samples actions using a Categorical distribution and provides log probabilities and entropy.
        Args:
            state (torch.Tensor): Input tensor of shape (obs_dim,).
        Returns:
            action (int): Sampled action.
            log_prob (torch.Tensor): Log probability of the sampled action.
            entropy (torch.Tensor): Entropy of the distribution.
            value (torch.Tensor): Value estimate for the state.
        """ 
        logits, value = self(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy(), value
    
    def evaluate(self, states, actions):
        """
        Correctly recomputes the log probabilities and entropy for the given actions.
        Args:
            states (torch.Tensor): Input tensor of shape (batch_size, obs_dim).
            actions (torch.Tensor): Actions taken.
        Returns:
            logprobs (torch.Tensor): Log probabilities of the actions.
            entropies (torch.Tensor): Entropy of the distribution.
            values (torch.Tensor): Value estimates for the states.
        """ 
        logits, values = self(states)
        dist = Categorical(logits=logits)
        logprobs = dist.log_prob(actions)
        entropies = dist.entropy()
        return logprobs, entropies, values
    

def compute_gae(rewards, values, dones):
    """
    Computes advantages and returns
    Args:
        rewards (torch.Tensor): Rewards received.
        values (torch.Tensor): Value estimates for the states.
        dones (torch.Tensor): Done flags for the states.
    Returns:
        returns (torch.Tensor): Returns for the states.
        advantages (torch.Tensor): Advantages for the states.
    """
    advantages = torch.zeros_like(rewards, device=DEVICE)
    lastgaelam = 0
    for t in reversed(range(len(rewards))):
        mask = 1.0 - dones[t]
        delta = rewards[t] + GAMMA * values[t+1] * mask - values[t]
        advantages[t] = lastgaelam = delta + GAMMA * GAE_LAMBDA * mask * lastgaelam
    returns = advantages + values[:-1]
    return advantages, returns


def train():
    # Set seeds for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    # For Gym environments, you might also want to seed the environment's action space for full reproducibility
    # env.action_space.seed(SEED) 

    # Initialize environment
    env = GuideDogEnv() 
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # Initialize model and optimizer
    model = ActorCritic(obs_dim, n_actions).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Initial observation from the environment
    # Modern Gym returns obs, info. Older Gym might just return obs.
    obs, info = env.reset(seed=SEED) 
    total_steps = 0

    print("Starting PPO training...")
    while total_steps < MAX_TIMESTEPS:
        # Rollout storage lists
        states_list   = []
        actions_list  = []
        logprobs_list = [] # Stores log_prob(a_t | s_t) from the policy used for rollout
        rewards_list  = []
        dones_list    = []
        values_list   = [] # Stores V(s_t) from the critic during rollout

        # 3.1) Collect HORIZON steps (rollout phase)
        model.eval() # Set model to evaluation mode for rollout (disables dropout, batchnorm updates etc.)
        for step_num_in_horizon in range(HORIZON):
            state_tensor = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
            
            # Get action, log_prob, and value from the current policy (actor and critic)
            with torch.no_grad(): # Disable gradient calculations for this block
                action, logp, _, value_from_model = model.get_action(state_tensor)

            # Take a step in the environment
            next_obs, reward, done, truncated, info = env.step(action)

            # Store transition data
            states_list.append(state_tensor) 
            actions_list.append(torch.tensor(action, device=DEVICE, dtype=torch.long)) # Actions are discrete
            logprobs_list.append(logp) # logp is a scalar tensor
            rewards_list.append(torch.tensor(reward, dtype=torch.float32, device=DEVICE))
            dones_list.append(torch.tensor(done or truncated, dtype=torch.float32, device=DEVICE)) # Combine done and truncated
            values_list.append(value_from_model) # value_from_model is a scalar tensor

            obs = next_obs
            total_steps += 1
            
            # If episode ends, reset environment
            if done or truncated:
                obs, info = env.reset(seed=SEED if total_steps < 2 * HORIZON else None) # Re-seed for first few resets for consistency

        # After collecting HORIZON steps, calculate the bootstrap value V(s_T) for the last state
        with torch.no_grad():
            obs_tensor_for_last_val = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
            # Ensure it has a batch dimension
            if obs_tensor_for_last_val.ndim == 1:
                 obs_tensor_for_last_val = obs_tensor_for_last_val.unsqueeze(0)
            # Get value from critic. model() calls forward(), which returns (logits, value)
            _, last_val = model(obs_tensor_for_last_val) 
        values_list.append(last_val.squeeze()) # Squeeze to make it a scalar tensor, append to V(s_0)...V(s_{T-1})

        # Convert lists of tensors/data to single tensors
        # These tensors represent the collected experience batch of size HORIZON
        states_tensor = torch.stack(states_list)              # Shape: [HORIZON, obs_dim]
        actions_tensor = torch.stack(actions_list)            # Shape: [HORIZON]
        old_logprobs_tensor = torch.stack(logprobs_list).detach() # Shape: [HORIZON], detach as these are targets from old policy
        
        rewards_tensor = torch.stack(rewards_list)            # Shape: [HORIZON]
        dones_tensor = torch.stack(dones_list)                # Shape: [HORIZON]
        values_tensor = torch.stack(values_list)              # Shape: [HORIZON+1] (includes bootstrap V(s_T))

        # 3.2) Compute GAE and returns
        # These are the targets for the actor and critic updates
        advantages_tensor, returns_tensor = compute_gae(rewards_tensor, values_tensor, dones_tensor)
        
        # Normalize advantages (often crucial for PPO stability)
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        # Detach advantages and returns as they are used as targets in the loss functions
        # and we don't want to backpropagate through their computation graph from the GAE step.
        advantages_detached = advantages_tensor.detach()
        returns_detached = returns_tensor.detach()
        
        # 3.3) PPO updates (learning phase)
        model.train() # Set model to training mode
        
        # Create a dataset and dataloader for iterating over mini-batches
        dataset = torch.utils.data.TensorDataset(
            states_tensor, 
            actions_tensor, 
            old_logprobs_tensor, 
            returns_detached,  # Target for value function
            advantages_detached # Target for policy gradient
        )
        # Shuffle ensures that mini-batches are random
        loader  = torch.utils.data.DataLoader(dataset, batch_size=MINI_BATCH, shuffle=True)

        # Perform multiple epochs of updates on the collected data
        for epoch_num in range(EPOCHS):
            for batch_idx, batch in enumerate(loader):
                b_states, b_actions, b_old_logp, b_returns, b_adv = batch
                
                # Get new logprobs, entropy, and values from the current policy
                # This involves a forward pass with the current model parameters
                new_logprobs, entropies, current_values = model.evaluate(b_states, b_actions)

                # Calculate PPO ratio: r_t(theta) = pi_theta(a_t|s_t) / pi_theta_old(a_t|s_t)
                # In log space: exp(log(pi_theta) - log(pi_theta_old))
                ratio = (new_logprobs - b_old_logp).exp()

                # Clipped surrogate objective (Actor/Policy loss)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * b_adv
                actor_loss = -torch.min(surr1, surr2).mean() # Negative because we want to maximize

                # Value function loss (Critic loss) - Mean Squared Error
                critic_loss = (b_returns - current_values).pow(2).mean()
                
                # Entropy bonus (to encourage exploration)
                entropy_loss = -entropies.mean() # Negative because we want to maximize entropy

                # Total loss: combination of actor, critic, and entropy losses
                loss = actor_loss + VF_COEF * critic_loss + ENT_COEF * entropy_loss

                # Optimization step
                optimizer.zero_grad() # Clear old gradients
                loss.backward()       # Compute gradients of the loss w.r.t. model parameters
                # Optional: Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5) 
                optimizer.step()      # Update model parameters
        
        # Logging after each rollout and update phase
        avg_reward_this_rollout = rewards_tensor.mean().item() # Average actual reward from the rollout
        print(f"[Step {total_steps}/{MAX_TIMESTEPS}] Loss: {loss.item():.4f}, Avg Rollout Reward: {avg_reward_this_rollout:.4f}, Entropy: {entropies.mean().item():.4f}")

    # Save final policy weights
    torch.save(model.state_dict(), "ppo_from_scratch.pth")
    print(f"Training finished. Model saved to ppo_from_scratch.pth after {total_steps} timesteps.")
    env.close() # Close the environment

if __name__ == "__main__":
    train()