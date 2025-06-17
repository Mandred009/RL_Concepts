# Deep Deterministic Policy Gradient (DDPG) implementation for MountainCarContinuous-v0
# Solves the environment in ~80 minutes on a GTX 1650

# Imports
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

# Ornstein-Uhlenbeck Noise Process: Adds temporally correlated exploration noise for continuous actions
class OUNoise:
    def __init__(self, size, mu=0, theta=0.15, sigma=0.2):
        self.epsilon = 0.9
        self.epsion_decaying = 0.99995
        self.epsilon_min = 0.01
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.size = size
        self.reset()

    def reset(self):
        """Reset internal noise state."""
        self.state = np.ones(self.size) * self.mu
    
    def decay_eps(self):
        """Decay exploration noise over time."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsion_decaying)

    def sample(self):
        """Generate a new noise sample using the OU process."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.epsilon


# Data structure for storing a single transition in replay buffer
@dataclass
class Experience:
    state: np.array
    action: np.array
    reward: float
    next_state: np.array
    done: bool


# Experience Replay Buffer: Stores past experiences for training stability
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


# Actor Network: Maps states to actions
class DDPG_ACTOR(nn.Module):
    def __init__(self, input_size, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 400),
            nn.BatchNorm1d(400),
            nn.LeakyReLU(),
            nn.Linear(400, 300),
            nn.BatchNorm1d(300),
            nn.LeakyReLU(),
            nn.Linear(300, n_actions),
            nn.Tanh()  # Output bounded in [-1, 1]
        )

    def forward(self, x):
        return self.net(x)


# Critic Network: Estimates Q-value for (state, action) pair
class DDPG_CRITIC(nn.Module):
    def __init__(self, input_size, n_actions):
        super().__init__()
        self.obs_net = nn.Sequential( # Observation path
            nn.Linear(input_size, 400),
            nn.BatchNorm1d(400),
            nn.LeakyReLU(),
        )
        self.out_net = nn.Sequential( # Action path
            nn.Linear(400 + n_actions, 300),
            nn.BatchNorm1d(300),
            nn.LeakyReLU(),
            nn.Linear(300, 1)  # Q-value output
        )

    def forward(self, x, a):
        obs = self.obs_net(x)
        return self.out_net(torch.cat([obs, a], dim=1))


# DDPG agent class: Interacts with the environment and fills the replay buffer
class DDPGM:
    def __init__(self, env: gym.Env, net: DDPG_ACTOR, buffer: ExperienceReplay):
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
        ou_noise.reset()

        state, _ = self.env.reset()
        while True:
            s_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            action = self.net(s_tensor).cpu().numpy()[0]
            action += ou_noise.sample()
            action = np.clip(action, ACTION_MIN, ACTION_MAX) # Clip actions

            next_state, reward, is_done, is_trunc, _ = self.env.step(action)

            # Bonus shaping reward for positive progress
            if next_state[0] > 0.1: # If dist is more than 0.1 add distance and abs velocity
                reward += next_state[0] + abs(next_state[1])

            self.total_r += reward
            exp = Experience(state, action, reward, next_state, (is_done or is_trunc))
            self.buffer.add_experience(exp)

            if is_done or is_trunc:
                break

            state = next_state
            self.steps += 1

        ou_noise.decay_eps() # Decay the epsilon every episode
        return self.total_r, self.steps


# Converts a batch of experiences into torch tensors for training
def calculate_batch(batch):
    states, actions, rewards, next_states, dones = zip(*[(e.state, e.action, e.reward, e.next_state, e.done) for e in batch])
    
    states_t = torch.from_numpy(np.array(states, dtype=np.float32)).to(DEVICE)
    actions_t = torch.from_numpy(np.array(actions, dtype=np.float32)).to(DEVICE)
    rewards_t = torch.from_numpy(np.array(rewards, dtype=np.float32)).unsqueeze(dim=-1).to(DEVICE)
    next_states_t = torch.from_numpy(np.array(next_states, dtype=np.float32)).to(DEVICE)
    dones_t = torch.from_numpy(np.array(dones, dtype=np.uint8)).unsqueeze(dim=-1).to(DEVICE).bool()

    return states_t, actions_t, rewards_t, next_states_t, dones_t


# Soft update of target networks
def soft_update(target_net: nn.Module, source_net: nn.Module):
    for target_param, param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(TAU_SOFT_UP * param.data + (1.0 - TAU_SOFT_UP) * target_param.data)


# Evaluate current policy on test environment
def test_agent(env: gym.Env, actor_net: DDPG_ACTOR):
    total_r = 0
    actor_net.eval()

    for _ in range(NOT_OF_TEST_EPI):
        state, _ = env.reset()
        while True:
            s_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            action = actor_net(s_tensor).cpu().numpy()[0]
            action = np.clip(action, ACTION_MIN, ACTION_MAX)

            next_state, reward, is_done, is_trunc, _ = env.step(action)
            if is_done or is_trunc:
                break
            state = next_state

        total_r += reward

    return total_r / NOT_OF_TEST_EPI


# === MAIN TRAINING LOOP ===
if __name__ == "__main__":
    # Hyperparameters
    GAMMA = 0.99 # Discount value for q-val 
    LEARNING_RATE_ACTOR = 1e-4
    LEARNING_RATE_CRITIC = 1e-4
    ACTION_MIN = -1 # Min action val for clipping
    ACTION_MAX = 1 # Max action val for clipping
    MAX_BUFFER = 1000000
    MIN_BUFFER_TRAIN = 50000 # The min size of buffer required to start training
    BATCH_SIZE = 64
    TAU_SOFT_UP = 0.005 # tau val for soft update
    TEST_ITER = 100 # No of episodes after which the testing should be done.
    NOT_OF_TEST_EPI = 3 # Num of test episodes to evaluate on 
    REWARD_LIMIT = 90 # The test reward exceeding this val will end training

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {DEVICE}")

    save_path = os.path.join("saves", "ddpg")
    os.makedirs(save_path, exist_ok=True)

    # Environment setup
    env = gym.make("MountainCarContinuous")
    test_env = gym.make("MountainCarContinuous", render_mode="human")
    N_ACTIONS = env.action_space.shape[0]

    # OU noise for exploration
    ou_noise = OUNoise(size=N_ACTIONS)

    # Networks
    actor_net = DDPG_ACTOR(env.observation_space.shape[0], N_ACTIONS).to(DEVICE)
    critic_net = DDPG_CRITIC(env.observation_space.shape[0], N_ACTIONS).to(DEVICE)
    target_actor_net = DDPG_ACTOR(env.observation_space.shape[0], N_ACTIONS).to(DEVICE)
    target_critic_net = DDPG_CRITIC(env.observation_space.shape[0], N_ACTIONS).to(DEVICE)

    # Sync target networks
    target_actor_net.load_state_dict(actor_net.state_dict())
    target_critic_net.load_state_dict(critic_net.state_dict())

    # Optimizers
    actor_optimizer = optim.Adam(actor_net.parameters(), lr=LEARNING_RATE_ACTOR)
    critic_optimizer = optim.Adam(critic_net.parameters(), lr=LEARNING_RATE_CRITIC)

    # Replay buffer and agent
    exp_buffer = ExperienceReplay(MAX_BUFFER)
    ddpg_agent = DDPGM(env, actor_net, exp_buffer)

    # Logging
    writer = SummaryWriter(comment="-DDPGM")
    steps, episodes = 0, 0
    best_reward = 0
    start_time = time.time()

    # Main loop
    while True:
        reward, stp = ddpg_agent.play_episode()
        steps += stp
        episodes += 1

        print(f"Training EP: {episodes} || Steps: {steps} || Reward: {reward}")

        writer.add_scalar("OU Noise Epsilon", ou_noise.epsilon, steps)
        writer.add_scalar("episodes", episodes, steps)
        writer.add_scalar("reward", reward, steps)

        # Start training only after the replay buffer has enough data
        if len(exp_buffer) < MIN_BUFFER_TRAIN:
            continue

        actor_net.train() # Training mode
        batch = exp_buffer.sample_experiences(BATCH_SIZE)
        states_t, actions_t, rewards_t, next_states_t, dones_t = calculate_batch(batch)

        # === Critic training ===
        critic_optimizer.zero_grad()
        q_v = critic_net(states_t, actions_t)

        with torch.no_grad():
            next_act_v = target_actor_net(next_states_t)
            next_q_v = target_critic_net(next_states_t, next_act_v)
            not_done_mask = (~dones_t).float()
            next_q_v *= not_done_mask

        q_ref_v = rewards_t + GAMMA * next_q_v
        critic_loss_v = F.mse_loss(q_v, q_ref_v.detach())
        critic_loss_v.backward()
        torch.nn.utils.clip_grad_norm_(critic_net.parameters(), 1.0) # Clipping gradients
        critic_optimizer.step()
        writer.add_scalar("critic loss", critic_loss_v.item(), steps)

        # === Actor training ===
        actor_optimizer.zero_grad()
        curr_actions_v = actor_net(states_t)
        actor_loss_v = -critic_net(states_t, curr_actions_v)
        actor_loss_f = actor_loss_v.mean()
        actor_loss_f.backward()
        torch.nn.utils.clip_grad_norm_(actor_net.parameters(), 1.0) # Clipping gradients
        actor_optimizer.step()
        writer.add_scalar("actor loss", actor_loss_f.item(), steps)

        # === Soft update target networks ===
        soft_update(target_actor_net, actor_net)
        soft_update(target_critic_net, critic_net)

        # === Test and save model ===
        if episodes % TEST_ITER == 0:
            print("TESTING....")
            test_reward = test_agent(test_env, actor_net)
            print(f"TEST REWARD: {test_reward}")
            writer.add_scalar("test reward", test_reward, steps)

            if test_reward > best_reward:
                best_reward = test_reward
                torch.save(actor_net.state_dict(), os.path.join(save_path, f"best_reward_{best_reward}"))

            if best_reward >= REWARD_LIMIT:
                print(f"SOLVED AT STEP: {steps} || Total Episodes: {episodes}")
                break

    # Training complete
    elapsed_time = (time.time() - start_time) / 60
    print(f"Time taken for complete training: {elapsed_time:.2f} min")
    writer.close()
