# PPO for Bipedal Walker Environment. Solved under 2 hours on GTX 1650. Better results as compared to other algo.

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard.writer import SummaryWriter
import collections
import time
import os

# Actor network – outputs mean (mu) and learnable standard deviation (std) for Gaussian policy
class PPO_ACTOR(nn.Module):
    def __init__(self, input_size, n_actions):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU()
        )
        self.mu_head = nn.Linear(128, n_actions)  # Mean of the action distribution
        self.log_std = nn.Parameter(torch.zeros(n_actions))  # Learnable log standard deviation

    def forward(self, x):
        x = self.base(x)
        mu = self.mu_head(x)
        std = torch.exp(self.log_std)  # Convert log_std to std
        return mu, std

# Critic network – outputs state-value estimate V(s)
class PPO_CRITIC(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),  # Single output for state-value
        )

    def forward(self, x):
        return self.model(x)

# PPO Agent for collecting experience and computing GAE (Generalized Advantage Estimation)
class PPO:
    def __init__(self, env: gym.Env, act_net:PPO_ACTOR, crt_net:PPO_CRITIC):
        self.state = []
        self.action = []
        self.reward = []
        self.log_prob = []
        self.env = env
        self.act_net = act_net
        self.crt_net = crt_net
        self.episode_no = 0
        self.trajectory_len = 0

    def reset_buffer(self):
        # Clears the buffer for next trajectory
        self.state = []
        self.action = []
        self.reward = []
        self.log_prob = []
        self.trajectory_len = 0

    # Computes advantage and reference values using GAE
    def calculate_advantage_reference_vals(self):
        states_v = torch.FloatTensor(self.state).to(DEVICE)
        action_v = torch.FloatTensor(self.action).to(DEVICE)
        log_probs_v = torch.FloatTensor(self.log_prob).to(DEVICE)

        value_v = self.crt_net(states_v)
        values = value_v.squeeze().data.cpu().numpy()

        last_gae = 0.0 # gae is generalized advantage estimation
        adv_list = []
        ref_list = []

        # Loop backward through trajectory to compute advantages
        for i in range(len(self.state) - 1, -1, -1):
            if i == len(self.state) - 1:
                delta = self.reward[i] + (GAMMA * 0) - values[i]  # no next value at terminal
                last_gae = delta
            else:
                delta = self.reward[i] + (GAMMA * values[i + 1]) - values[i]
                last_gae = delta + (GAMMA * GAE_LAMBDA * last_gae)

            adv_list.append(last_gae)
            ref_list.append(last_gae + values[i])  # Q = A + V

        adv_v = torch.FloatTensor(list(reversed(adv_list))).to(DEVICE)
        ref_v = torch.FloatTensor(list(reversed(ref_list))).to(DEVICE)

        return states_v, action_v, adv_v, ref_v, log_probs_v

    @torch.no_grad
    def play_episode(self):
        state, _ = self.env.reset()
        total_r = 0
        while True:
            s_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            mu, std = self.act_net(s_tensor)
            dist = torch.distributions.Normal(mu, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=1)

            # Apply tanh to squash actions to [-1, 1] range
            action_env = torch.tanh(action).squeeze(0).cpu().numpy()
            self.state.append(state)
            self.action.append(action.squeeze(0).cpu().numpy())  # raw action
            self.log_prob.append(log_prob.item())

            next_state, reward, terminated, truncated, _ = self.env.step(action_env)
            self.reward.append(reward)
            total_r += reward

            self.trajectory_len += 1

            if terminated or truncated:
                break
            state = next_state

        self.episode_no += 1
        return self.episode_no, self.trajectory_len, total_r

# Evaluation logic without gradient updates
@torch.no_grad()
def test_agent(env: gym.Env, actor_net: PPO_ACTOR):
    total_r = 0
    actor_net.eval()

    for t in range(NO_OF_TEST_EPI):
        print(f"test:{t}")
        state, _ = env.reset()
        while True:
            s_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            mu, std = actor_net(s_tensor)
            action = torch.tanh(mu).squeeze(0).cpu().numpy()

            next_state, reward, is_done, is_trunc, _ = env.step(action)
            if is_done or is_trunc:
                break
            state = next_state
            total_r += reward

    return total_r / NO_OF_TEST_EPI


# === MAIN TRAINING LOGIC ===
if __name__ == "__main__":
    # Hyperparameters
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    LEARNING_RATE_ACTOR = 0.0001
    LEARNING_RATE_CRITIC = 0.001
    TRAJECTORY_SIZE = 2049
    PPO_EPSILON = 0.2
    PPO_EPOCHS = 10
    PPO_BATCH_SIZE = 64
    TEST_EPS = 100
    NO_OF_TEST_EPI = 3
    REWARD_LIMIT = 290
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {DEVICE}")

    # Directory to save models
    save_path = os.path.join("saves")
    os.makedirs(save_path, exist_ok=True)

    # Environment setup
    env = gym.make("BipedalWalker-v3")
    test_env = gym.make("BipedalWalker-v3", render_mode="human")
    N_ACTIONS = env.action_space.shape[0]

    # Initialize networks and optimizers
    actor_net = PPO_ACTOR(env.observation_space.shape[0], N_ACTIONS).to(DEVICE)
    critic_net = PPO_CRITIC(env.observation_space.shape[0]).to(DEVICE)
    print(f"The actor net architecture: {actor_net}")
    print(f"The critic net architecture: {critic_net}")

    actor_optimizer = optim.Adam(actor_net.parameters(), lr=LEARNING_RATE_ACTOR)
    critic_optimizer = optim.Adam(critic_net.parameters(), lr=LEARNING_RATE_CRITIC)

    ppo_agent = PPO(env, actor_net, critic_net)
    writer = SummaryWriter(comment="-PPO_ALGO")

    best_reward = 0
    start_time = time.time()

    # Main training loop
    while True:
        eps, traj_len, eps_reward = ppo_agent.play_episode()
        writer.add_scalar("train reward", eps_reward, eps)
        print(f"EPISODE NO: {eps} || REWARD: {eps_reward} || TRAJ: {traj_len}")

        # Periodic testing
        if eps % TEST_EPS == 0:
            test_reward = test_agent(test_env, actor_net)
            print(f"TESTING REWARD {test_reward}")
            writer.add_scalar("test reward", test_reward, eps)

            if test_reward > best_reward:
                best_reward = test_reward
                torch.save(actor_net.state_dict(), os.path.join(save_path, f"best_reward_{best_reward}"))

            if best_reward > REWARD_LIMIT:
                print(f"SOLVED AT Total Episodes: {eps}")
                break

        # Wait until full trajectory collected
        if traj_len < TRAJECTORY_SIZE:
            continue

        # Compute GAE and reference values for policy update
        states_v, actions_v, adv_v, ref_v, old_log_probs_v = ppo_agent.calculate_advantage_reference_vals()
        adv_v = (adv_v - adv_v.mean()) / (adv_v.std() + 1e-8)  # Normalize advantage

        sum_loss_policy = 0.0
        sum_loss_value = 0.0
        steps = 0

        print("Training")
        for epoch in range(PPO_EPOCHS):
            for batch in range(0, len(states_v), PPO_BATCH_SIZE):
                b_states = states_v[batch:batch+PPO_BATCH_SIZE]
                b_actions = actions_v[batch:batch+PPO_BATCH_SIZE]
                b_adv = adv_v[batch:batch+PPO_BATCH_SIZE]
                b_ref = ref_v[batch:batch+PPO_BATCH_SIZE]
                b_old_logp = old_log_probs_v[batch:batch+PPO_BATCH_SIZE]

                # === Critic Update ===
                critic_optimizer.zero_grad()
                values = critic_net(b_states).squeeze(-1)
                loss_value = F.mse_loss(values, b_ref)
                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(critic_net.parameters(), 1.0)
                critic_optimizer.step()

                # === Actor Update (PPO Clipped Objective) ===
                actor_optimizer.zero_grad()
                mu, std = actor_net(b_states)
                dist = torch.distributions.Normal(mu, std)
                logp = dist.log_prob(b_actions).sum(dim=1)
                ratio = torch.exp(logp - b_old_logp)

                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - PPO_EPSILON, 1.0 + PPO_EPSILON) * b_adv
                loss_policy = -torch.min(surr1, surr2).mean()
                loss_policy.backward()
                torch.nn.utils.clip_grad_norm_(actor_net.parameters(), 1.0)
                actor_optimizer.step()

                sum_loss_policy += loss_policy.item()
                sum_loss_value += loss_value.item()
                steps += 1

        # Logging and buffer reset
        ppo_agent.reset_buffer()
        writer.add_scalar("advantage", adv_v.mean().item(), eps)
        writer.add_scalar("values", ref_v.mean().item(), eps)
        writer.add_scalar("loss_policy", sum_loss_policy / steps, eps)
        writer.add_scalar("loss_value", sum_loss_value / steps, eps)

    # Final training summary
    end_time = time.time()
    elapsed_time = (end_time - start_time)/60
    print(f"Time taken for complete training: {elapsed_time:.2f} min")
    writer.close()
