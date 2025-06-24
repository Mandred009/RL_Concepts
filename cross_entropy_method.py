# Cross Entropy Method for RL on Cartpole. It takes around 50 epochs and 180 min to train on GTX 1650

# Core libraries
import numpy as np
import gymnasium as gym
from dataclasses import dataclass
from torch.utils.tensorboard.writer import SummaryWriter

# PyTorch components
import torch
import torch.nn as nn
import torch.optim as optim

import time


# Define a simple structure to store an episode's data
@dataclass
class Episode:
    states: list
    actions: list
    reward_sum: float


# Policy Network (MLP) that outputs action probabilities
class NN(nn.Module):
    def __init__(self, inp_size, hidden_size, out_size):
        super().__init__()

        self.nn = nn.Sequential(
            nn.Linear(inp_size, hidden_size),
            nn.LeakyReLU(),  # LeakyReLU generally converges faster than ReLU in practice
            nn.Linear(hidden_size, out_size),
            nn.Softmax(dim=1)  # Output probabilities over actions
        )

    def forward(self, x):
        return self.nn(x)
    

# Cross Entropy Method agent
class cross_entropy_rl:
    def __init__(self, env: gym.Env, nn, num_of_episodes, reward_percentile):
        self.env = env
        self.nn_net = nn
        self.num_of_episodes = num_of_episodes
        self.reward_percentile = reward_percentile
        self.episodes = []  # stores all episodes played in current epoch

    def play_n_episodes(self):
        # Plays N episodes and stores them in self.episodes
        for i in range(self.num_of_episodes):
            states = []
            rewards = []
            actions = []
            obs, _ = self.env.reset()
            is_done = False
            is_trunc = False
            
            while not is_done and not is_trunc:
                # Convert observation to tensor and predict action probabilities
                obs_v = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                action_prob = self.nn_net(obs_v).data.cpu().numpy()[0]  # extract probabilities

                # Choose action based on predicted distribution
                action = np.random.choice(len(action_prob), p=action_prob)

                # Apply action
                next_obs, reward, is_done, is_trunc, _ = env.step(action)

                # Log data
                states.append(obs)
                rewards.append(reward)
                actions.append(action)

                obs = next_obs

            reward_sum = sum(rewards)

            # Save the episode
            self.episodes.append(Episode(states, actions, reward_sum))

    def filter_episodes(self):
        # Filter top episodes based on reward percentile
        epi_rewards = [x.reward_sum for x in self.episodes]
        reward_band = np.percentile(epi_rewards, self.reward_percentile)

        train_obs = []
        train_action = []

        good_episodes = []

        for i in self.episodes:
            if i.reward_sum >= reward_band:
                good_episodes.append(i)
                for j in range(len(i.states)):
                    train_obs.append(i.states[j])
                    train_action.append(i.actions[j])
        
        mean_reward = sum(epi_rewards) / len(epi_rewards)

        # Keep only high-reward episodes for next epoch
        self.episodes = good_episodes

        # Convert to PyTorch tensors
        train_obs_tensor = torch.tensor(np.stack(train_obs), dtype=torch.float32).to(device)
        train_action_tensor = torch.tensor(np.stack(train_action), dtype=torch.long).to(device)
    
        return train_obs_tensor, train_action_tensor, mean_reward, reward_band


# ===================== Main Training Loop ===================== #
if __name__ == "__main__":
    EPOCHS = 50                  # Max training epochs
    EPISODE_PLAY = 50           # Number of episodes per epoch
    REWARD_PERCENTILE = 92      # Top percentile to retain

    # Initialize CartPole environment with rendering
    env = gym.make("CartPole-v1", render_mode="human")
    env.reset()
    env.render()
    
    obs_size = env.observation_space.shape[0]  # e.g., 4
    n_actions = 2                              # left or right

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    # Initialize policy network
    nn_net = NN(obs_size, 128, n_actions).to(device)
    print(f"The neural net architecture: {nn_net}")

    # Define loss and optimizer
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=nn_net.parameters(), lr=0.1)

    # TensorBoard logger
    writer = SummaryWriter(comment="-cartpole")

    # Cross entropy method agent
    cross_en_m = cross_entropy_rl(env, nn_net, EPISODE_PLAY, REWARD_PERCENTILE)

    start_time = time.time()

    # Training loop
    for i in range(EPOCHS):
        cross_en_m.play_n_episodes()
        train_obs, train_action, mean_reward, band_reward = cross_en_m.filter_episodes()

        optimizer.zero_grad()
        action_scores_v = nn_net(train_obs)
        loss_v = objective(action_scores_v, train_action)
        loss_v.backward()
        optimizer.step()

        print(f"Epoch: {i} || Loss: {loss_v.item()} || Reward Mean: {mean_reward} || Reward Band: {band_reward}")

        # Log to TensorBoard
        writer.add_scalar("loss", loss_v.item(), i)
        writer.add_scalar("reward_mean", mean_reward, i)

        # If environment is considered solved
        if mean_reward >= 490:
            break

    end_time = time.time()

    # Elapsed time in minutes
    elapsed_time = (end_time - start_time)/60
    print(f"Time taken for {EPOCHS} epochs: {elapsed_time:.2f} min")

    writer.close()
    print("Done")
