# Twin Delayed Deep Deterministic Policy Gradient (TD3) Implementation for Inverted Double Pendulum
# Achieves ~9350 reward in under 60 minutes on GTX 1650 GPU

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

# === Ornstein-Uhlenbeck Noise ===
# Used to add temporally correlated noise to continuous actions for better exploration
class OUNoise:
    def __init__(self, size, mu=0, theta=0.15, sigma=0.2):
        self.epsilon = 0.9
        self.epsilon_decaying = 0.99995
        self.epsilon_min = 0.01
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.size = size
        self.reset()

    def reset(self):
        """Resets the internal state of the noise process."""
        self.state = np.ones(self.size) * self.mu
    
    def decay_eps(self):
        """Decays the exploration factor epsilon over time."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decaying)

    def sample(self):
        """Generates a new noise sample using the OU process."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.epsilon


# === Experience Tuple ===
# Used to store individual transitions in the replay buffer
@dataclass
class Experience:
    state: np.array
    action: np.array
    reward: float
    next_state: np.array
    done: bool


# === Experience Replay Buffer ===
# Stores and samples experiences to break correlations and stabilize training
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


# === Actor Network ===
# Maps observations to continuous actions
class TD3_ACTOR(nn.Module):
    def __init__(self, input_size, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Linear(128, n_actions),
            nn.Tanh()  # Ensures actions are in the range [-1, 1]
        )

    def forward(self, x):
        return self.net(x)


# === Critic Network ===
# Estimates Q-values for (state, action) pairs
class TD3_CRITIC(nn.Module):
    def __init__(self, input_size, n_actions):
        super().__init__()
        self.obs_net = nn.Sequential(  # Processes state input
            nn.Linear(input_size, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(),
        )
        self.out_net = nn.Sequential(  # Combines state and action
            nn.Linear(256 + n_actions, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Linear(128, 1)  # Outputs scalar Q-value
        )

    def forward(self, x, a):
        obs = self.obs_net(x)
        return self.out_net(torch.cat([obs, a], dim=1))


# === TD3 Agent ===
# Plays episodes, adds experiences to buffer
class TD3:
    def __init__(self, env: gym.Env, net: TD3_ACTOR, buffer: ExperienceReplay):
        self.env = env
        self.net = net
        self.buffer = buffer
        self.total_r = 0
        self.steps = 0

    @torch.no_grad()
    def play_episode(self):
        self.total_r = 0
        self.steps = 0
        ou_noise.reset()

        state, _ = self.env.reset()
        while True:
            s_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            # Select action with exploration noise
            action = self.net(s_tensor).cpu().numpy()[0]
            action += ou_noise.sample()
            action = np.clip(action, ACTION_MIN, ACTION_MAX)

            # Interact with the environment
            next_state, reward, is_done, is_trunc, _ = self.env.step(action)

            self.total_r += reward
            exp = Experience(state, action, reward, next_state, (is_done or is_trunc))
            self.buffer.add_experience(exp)

            if is_done or is_trunc:
                break

            state = next_state
            self.steps += 1

        ou_noise.decay_eps()
        return self.total_r, self.steps


# === Utility Function ===
# Converts a batch of Experience into tensors for training
def calculate_batch(batch):
    states, actions, rewards, next_states, dones = zip(*[(e.state, e.action, e.reward, e.next_state, e.done) for e in batch])
    
    states_t = torch.from_numpy(np.array(states, dtype=np.float32)).to(DEVICE)
    actions_t = torch.from_numpy(np.array(actions, dtype=np.float32)).to(DEVICE)
    rewards_t = torch.from_numpy(np.array(rewards, dtype=np.float32)).unsqueeze(dim=-1).to(DEVICE)
    next_states_t = torch.from_numpy(np.array(next_states, dtype=np.float32)).to(DEVICE)
    dones_t = torch.from_numpy(np.array(dones, dtype=np.uint8)).unsqueeze(dim=-1).to(DEVICE).bool()

    return states_t, actions_t, rewards_t, next_states_t, dones_t


# === Target Network Soft Update ===
def soft_update(target_net: nn.Module, source_net: nn.Module):
    for target_param, param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(TAU_SOFT_UP * param.data + (1.0 - TAU_SOFT_UP) * target_param.data)


# === Evaluate Policy ===
@torch.no_grad()
def test_agent(env: gym.Env, actor_net: TD3_ACTOR):
    total_r = 0

    for t in range(NUM_OF_TEST_EPI):
        print(f"t:{t}")
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

    return total_r / NUM_OF_TEST_EPI


# === MAIN TRAINING LOOP ===
if __name__ == "__main__":
    # === Hyperparameters ===
    GAMMA = 0.99
    LEARNING_RATE_ACTOR = 1e-4
    LEARNING_RATE_CRITIC = 1e-3
    POLICY_DELAY = 2  # Delays actor updates to stabilize training
    TARGET_NOISE_STD = 0.2 # Noise to be added to target actor output
    TARGET_NOISE_CLIP = 0.5
    ACTION_MIN = -1
    ACTION_MAX = 1
    MAX_BUFFER = 1000000
    MIN_BUFFER_TRAIN = 10000
    BATCH_SIZE = 64
    TAU_SOFT_UP = 0.005
    TEST_ITER = 100
    NUM_OF_TEST_EPI = 3
    REWARD_LIMIT = 10000

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {DEVICE}")

    save_path = os.path.join("saves")
    os.makedirs(save_path, exist_ok=True)

    # === Environment Setup ===
    env = gym.make("InvertedDoublePendulum-v5")
    test_env = gym.make("InvertedDoublePendulum-v5", render_mode="human")
    N_ACTIONS = env.action_space.shape[0]

    # === Initialize Components ===
    ou_noise = OUNoise(size=N_ACTIONS)

    actor_net = TD3_ACTOR(env.observation_space.shape[0], N_ACTIONS).to(DEVICE)
    critic_net_q1 = TD3_CRITIC(env.observation_space.shape[0], N_ACTIONS).to(DEVICE)
    critic_net_q2 = TD3_CRITIC(env.observation_space.shape[0], N_ACTIONS).to(DEVICE)

    target_actor_net = TD3_ACTOR(env.observation_space.shape[0], N_ACTIONS).to(DEVICE)
    target_critic_net_q1 = TD3_CRITIC(env.observation_space.shape[0], N_ACTIONS).to(DEVICE)
    target_critic_net_q2 = TD3_CRITIC(env.observation_space.shape[0], N_ACTIONS).to(DEVICE)

    # === Sync Target Networks ===
    target_actor_net.load_state_dict(actor_net.state_dict())
    target_critic_net_q1.load_state_dict(critic_net_q1.state_dict())
    target_critic_net_q2.load_state_dict(critic_net_q2.state_dict())

    # === Optimizers ===
    actor_optimizer = optim.Adam(actor_net.parameters(), lr=LEARNING_RATE_ACTOR)
    critic_optimizer_q1 = optim.Adam(critic_net_q1.parameters(), lr=LEARNING_RATE_CRITIC)
    critic_optimizer_q2 = optim.Adam(critic_net_q2.parameters(), lr=LEARNING_RATE_CRITIC)

    # === Agent and Replay Buffer ===
    exp_buffer = ExperienceReplay(MAX_BUFFER)
    td3_agent = TD3(env, actor_net, exp_buffer)

    # === Logger ===
    writer = SummaryWriter(comment="-TD3")
    steps, episodes = 0, 0
    best_reward = 0
    start_time = time.time()

    # === Training Loop ===
    while True:
        reward, stp = td3_agent.play_episode()
        steps += stp
        episodes += 1

        print(f"Training EP: {episodes} || Steps: {steps} || Reward: {reward}")

        writer.add_scalar("OU Noise Epsilon", ou_noise.epsilon, steps)
        writer.add_scalar("episodes", episodes, steps)
        writer.add_scalar("reward", reward, steps)

        # Skip training until enough experiences are gathered
        if len(exp_buffer) < MIN_BUFFER_TRAIN:
            continue

        batch = exp_buffer.sample_experiences(BATCH_SIZE)
        states_t, actions_t, rewards_t, next_states_t, dones_t = calculate_batch(batch)

        # === Critic Update ===
        critic_optimizer_q1.zero_grad()
        critic_optimizer_q2.zero_grad()
        q1_v = critic_net_q1(states_t, actions_t)
        q2_v = critic_net_q2(states_t, actions_t)

        with torch.no_grad():
            next_act_v = target_actor_net(next_states_t)
            noise_target = torch.normal(0, TARGET_NOISE_STD, size=next_act_v.shape, device=DEVICE)
            noise_target = noise_target.clamp(-TARGET_NOISE_CLIP, TARGET_NOISE_CLIP)
            next_act_v = (next_act_v + noise_target).clamp(ACTION_MIN, ACTION_MAX)
            next_q1_v = target_critic_net_q1(next_states_t, next_act_v)
            next_q2_v = target_critic_net_q2(next_states_t, next_act_v)
            not_done_mask = (~dones_t).float()
            next_q_v = torch.min(next_q1_v, next_q2_v) * not_done_mask

        q_ref_v = rewards_t + GAMMA * next_q_v
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

        # === Delayed Actor Update ===
        if episodes % POLICY_DELAY == 0:
            actor_optimizer.zero_grad()
            curr_actions_v = actor_net(states_t)
            q1 = critic_net_q1(states_t, curr_actions_v)
            q2 = critic_net_q2(states_t, curr_actions_v)
            actor_loss_v = -torch.min(q1, q2).mean()
            actor_loss_v.backward()
            torch.nn.utils.clip_grad_norm_(actor_net.parameters(), 1.0)
            actor_optimizer.step()
            writer.add_scalar("actor loss", actor_loss_v.item(), steps)

            # Soft update of target networks
            soft_update(target_actor_net, actor_net)
            soft_update(target_critic_net_q1, critic_net_q1)
            soft_update(target_critic_net_q2, critic_net_q2)

        # === Evaluation and Model Saving ===
        if episodes % TEST_ITER == 0:
            print("TESTING....")
            test_reward = test_agent(test_env, actor_net)
            print(f"TEST REWARD: {test_reward}")
            writer.add_scalar("test reward", test_reward, steps)

            if test_reward > best_reward:
                best_reward = test_reward
                torch.save(actor_net.state_dict(), os.path.join(save_path, f"td3_best_{best_reward}"))

            if best_reward >= REWARD_LIMIT:
                print(f"SOLVED AT STEP: {steps} || Total Episodes: {episodes}")
                break

    # === Training Complete ===
    elapsed_time = (time.time() - start_time) / 60
    print(f"Time taken for complete training: {elapsed_time:.2f} min")
    writer.close()
