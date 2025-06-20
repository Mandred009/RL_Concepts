# Soft Actor Critic (SAC) on Half-Cheetah.
# Achieved a score of 1187 in 500 minutes on GTX 1650. Train on better device for more time for better results.
# The learned gait is functional but quite funny.

# === Imports ===
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard.writer import SummaryWriter
from dataclasses import dataclass
import collections
import time
import os
import random

# === Data structure to store a single transition in the replay buffer ===
@dataclass
class Experience:
    state: np.array
    action: np.array
    reward: float
    next_state: np.array
    done: bool

# === Experience Replay Buffer ===
# Stores past experiences to decorrelate training data and stabilize learning
class ExperienceReplay:
    def __init__(self, max_buffer_size):
        self.buffer = collections.deque(maxlen=max_buffer_size)
    
    def __len__(self):
        return len(self.buffer)

    def add_experience(self, experience: Experience):
        self.buffer.append(experience)

    def sample_experiences(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[idx] for idx in indices]

# === SAC Actor Network ===
# Outputs mean and std for the Gaussian policy
class SAC_ACTOR(nn.Module):
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
        std = torch.exp(self.log_std)  # Convert log_std to std deviation
        return mu, std

# === SAC Critic Network ===
# Q-value estimator: takes (state, action) as input
class SAC_CRITIC(nn.Module):
    def __init__(self, input_size, n_actions):
        super().__init__()
        self.obs_net = nn.Sequential(  # Observation processing
            nn.Linear(input_size, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(),
        )
        self.out_net = nn.Sequential(  # Combined (obs + action) processing
            nn.Linear(256 + n_actions, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Linear(128, 1)  # Outputs scalar Q-value
        )

    def forward(self, x, a):
        obs = self.obs_net(x)
        return self.out_net(torch.cat([obs, a], dim=1))

# === SAC Agent (Environment interaction wrapper) ===
class SAC:
    def __init__(self, env: gym.Env, net: SAC_ACTOR, buffer: ExperienceReplay):
        self.env = env
        self.net = net
        self.buffer = buffer
        self.total_r = 0
        self.steps = 0

    @torch.no_grad()
    def play_episode(self):
        self.net.eval()
        self.total_r = 0
        self.steps = 0

        state, _ = self.env.reset()
        while True:
            s_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            mu, std = self.net(s_tensor)
            dist = torch.distributions.Normal(mu, std)
            action = dist.rsample()
            action = torch.tanh(action).squeeze(0).cpu().numpy()

            next_state, reward, is_done, is_trunc, _ = self.env.step(action)

            self.total_r += reward
            exp = Experience(state, action, reward, next_state, (is_done or is_trunc))
            self.buffer.add_experience(exp)

            if is_done or is_trunc:
                break

            state = next_state
            self.steps += 1

        return self.total_r, self.steps

# === Converts a batch of experiences into PyTorch tensors ===
def calculate_batch(batch):
    states, actions, rewards, next_states, dones = zip(*[(e.state, e.action, e.reward, e.next_state, e.done) for e in batch])
    
    states_t = torch.from_numpy(np.array(states, dtype=np.float32)).to(DEVICE)
    actions_t = torch.from_numpy(np.array(actions, dtype=np.float32)).to(DEVICE)
    rewards_t = torch.from_numpy(np.array(rewards, dtype=np.float32)).unsqueeze(dim=-1).to(DEVICE)
    next_states_t = torch.from_numpy(np.array(next_states, dtype=np.float32)).to(DEVICE)
    dones_t = torch.from_numpy(np.array(dones, dtype=np.uint8)).unsqueeze(dim=-1).to(DEVICE).bool()

    return states_t, actions_t, rewards_t, next_states_t, dones_t

# === Soft update of target network parameters ===
def soft_update(target_net: nn.Module, source_net: nn.Module):
    for target_param, param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(TAU_SOFT_UP * param.data + (1.0 - TAU_SOFT_UP) * target_param.data)

# === Evaluate current policy performance in test environment ===
@torch.no_grad()
def test_agent(env: gym.Env, actor_net: SAC_ACTOR):
    total_r = 0
    actor_net.eval()

    for t in range(NOT_OF_TEST_EPI):
        print(f"test:{t}")
        state, _ = env.reset()
        while True:
            s_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            mu, _ = actor_net(s_tensor)
            action = torch.tanh(mu).squeeze(0).cpu().numpy()

            next_state, reward, is_done, is_trunc, _ = env.step(action)
            if is_done or is_trunc:
                break
            state = next_state

            total_r += reward

    return total_r / NOT_OF_TEST_EPI

# === MAIN TRAINING LOOP ===
if __name__ == "__main__":
    # Hyperparameters
    GAMMA = 0.99  # Discount factor for future rewards
    LEARNING_RATE_ACTOR = 3e-4
    LEARNING_RATE_CRITIC = 3e-4
    ENTROPY_ALPHA = 0.3  # Entropy regularization coefficient
    ACTION_MIN = -1
    ACTION_MAX = 1
    MAX_BUFFER = 100000
    MIN_BUFFER_TRAIN = 10000  # Minimum buffer size before training starts
    BATCH_SIZE = 64
    TAU_SOFT_UP = 0.005  # Soft update coefficient
    TEST_ITER = 100  # Test policy every N episodes
    NOT_OF_TEST_EPI = 3  # Number of test episodes for evaluation
    REWARD_LIMIT = 1000  # Stop training if test reward exceeds this value

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {DEVICE}")

    save_path = os.path.join("saves")
    os.makedirs(save_path, exist_ok=True)

    # === Environment setup ===
    env = gym.make("HalfCheetah-v5")
    test_env = gym.make("HalfCheetah-v5", render_mode="human")
    N_ACTIONS = env.action_space.shape[0]

    # === Network initialization ===
    actor_net = SAC_ACTOR(env.observation_space.shape[0], N_ACTIONS).to(DEVICE)
    critic_net_q1 = SAC_CRITIC(env.observation_space.shape[0], N_ACTIONS).to(DEVICE)
    critic_net_q2 = SAC_CRITIC(env.observation_space.shape[0], N_ACTIONS).to(DEVICE)
    print(f"The actor net architecture: {actor_net}")
    print(f"The critic net architecture: {critic_net_q2}")

    target_critic_q1 = SAC_CRITIC(env.observation_space.shape[0], N_ACTIONS).to(DEVICE)
    target_critic_q2 = SAC_CRITIC(env.observation_space.shape[0], N_ACTIONS).to(DEVICE)

    # === Synchronize target networks with initial critic weights ===
    target_critic_q1.load_state_dict(critic_net_q1.state_dict())
    target_critic_q2.load_state_dict(critic_net_q2.state_dict())

    # === Optimizers ===
    actor_optimizer = optim.Adam(actor_net.parameters(), lr=LEARNING_RATE_ACTOR)
    critic_optimizer_q1 = optim.Adam(critic_net_q1.parameters(), lr=LEARNING_RATE_CRITIC)
    critic_optimizer_q2 = optim.Adam(critic_net_q2.parameters(), lr=LEARNING_RATE_CRITIC)

    # === Replay buffer and agent setup ===
    exp_buffer = ExperienceReplay(MAX_BUFFER)
    sac_agent = SAC(env, actor_net, exp_buffer)

    # === TensorBoard logging ===
    writer = SummaryWriter(comment="-SAC")
    steps, episodes = 0, 0
    best_reward = 0
    start_time = time.time()

    # === Main training loop ===
    while True:
        reward, stp = sac_agent.play_episode()
        steps += stp
        episodes += 1

        print(f"Training EP: {episodes} || Steps: {steps} || Reward: {reward}")
        writer.add_scalar("episodes", episodes, steps)
        writer.add_scalar("reward", reward, steps)

        # Wait until buffer has enough samples to start training
        if len(exp_buffer) < MIN_BUFFER_TRAIN:
            continue

        batch = exp_buffer.sample_experiences(BATCH_SIZE)
        states_t, actions_t, rewards_t, next_states_t, dones_t = calculate_batch(batch)

        # === Critic training ===
        critic_optimizer_q1.zero_grad()
        critic_optimizer_q2.zero_grad()
        q1_v = critic_net_q1(states_t, actions_t)
        q2_v = critic_net_q2(states_t, actions_t)

        with torch.no_grad():
            mu_next, std_next = actor_net(next_states_t)
            dist = torch.distributions.Normal(mu_next, std_next)
            next_action = dist.rsample()
            log_prob_next = dist.log_prob(next_action).sum(dim=1)
            next_q1_v = target_critic_q1(next_states_t, next_action)
            next_q2_v = target_critic_q2(next_states_t, next_action)
            not_done_mask = (~dones_t).float()
            next_q_v = torch.min(next_q1_v, next_q2_v) * not_done_mask

        q_ref_v = rewards_t + GAMMA * (next_q_v - ENTROPY_ALPHA * log_prob_next.unsqueeze(-1))
        critic_loss_q1 = F.mse_loss(q1_v, q_ref_v.detach())
        critic_loss_q2 = F.mse_loss(q2_v, q_ref_v.detach())
        critic_loss_q1.backward()
        critic_loss_q2.backward()
        torch.nn.utils.clip_grad_norm_(critic_net_q1.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(critic_net_q2.parameters(), 1.0)
        critic_optimizer_q1.step()
        critic_optimizer_q2.step()
        writer.add_scalar("critic loss q1", critic_loss_q1.item(), steps)
        writer.add_scalar("critic loss q2", critic_loss_q2.item(), steps)

        # === Actor training ===
        actor_optimizer.zero_grad()
        curr_mu, curr_std = actor_net(states_t)
        dist = torch.distributions.Normal(curr_mu, curr_std)
        curr_actions_v = dist.rsample()
        log_prob_curr = dist.log_prob(curr_actions_v).sum(dim=1)

        q1 = critic_net_q1(states_t, curr_actions_v)
        q2 = critic_net_q2(states_t, curr_actions_v)
        actor_loss_v = -(torch.min(q1, q2) - (ENTROPY_ALPHA * log_prob_curr.unsqueeze(-1)))
        actor_loss_f = actor_loss_v.mean()
        actor_loss_f.backward()
        torch.nn.utils.clip_grad_norm_(actor_net.parameters(), 1.0)
        actor_optimizer.step()
        writer.add_scalar("actor loss", actor_loss_f.item(), steps)

        # === Soft update of target networks ===
        soft_update(target_critic_q1, critic_net_q1)
        soft_update(target_critic_q2, critic_net_q2)

        # === Periodic policy testing and model saving ===
        if episodes % TEST_ITER == 0:
            print("TESTING....")
            test_reward = test_agent(test_env, actor_net)
            print(f"TEST REWARD: {test_reward}")
            writer.add_scalar("test reward", test_reward, steps)

            if test_reward > best_reward:
                best_reward = test_reward
                torch.save(actor_net.state_dict(), os.path.join(save_path, f"sac_best_{best_reward}"))

            if best_reward >= REWARD_LIMIT:
                print(f"SOLVED AT STEP: {steps} || Total Episodes: {episodes}")
                break

    # === Final training summary ===
    elapsed_time = (time.time() - start_time) / 60
    print(f"Time taken for complete training: {elapsed_time:.2f} min")
    writer.close()
