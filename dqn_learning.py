# DQN for Atari Breakout. I was not able to make this converge training on GTX 1650. Max reward was 5.

# Core libraries
import numpy as np
import gymnasium as gym
from dataclasses import dataclass
from torch.utils.tensorboard.writer import SummaryWriter

# PyTorch for NN and optimization
import torch
import torch.nn as nn
import torch.optim as optim

# Utilities
import time
import ale_py
import collections
import cv2
import os

# Define a dataclass to store each experience tuple
@dataclass
class Experience:
    state: np.array
    action: float
    reward: float
    next_state: np.array
    done: bool

# Neural Network used to approximate Q-values
class NN(nn.Module):
    def __init__(self, inp_size, no_actions):
        super().__init__()
        # Convolutional layers to process image inputs
        self.conv = nn.Sequential(
            nn.Conv2d(inp_size[0], 32, kernel_size=8, stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.Flatten(),
        )
        # Compute output size after conv layers
        size = self.conv(torch.zeros(1, *inp_size)).size()[-1]
        
        # Fully connected layers for Q-value prediction
        self.fc = nn.Sequential(
            nn.Linear(size, 512),
            nn.LeakyReLU(),
            nn.Linear(512, no_actions)
        )

    def forward(self, x):
        # Normalize image pixels from [0,255] to [0,1]
        xx = x / 255.0
        return self.fc(self.conv(xx))

# Experience Replay Buffer to store and sample experiences
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

# DQN Agent
class DQN:
    def __init__(self, env: gym.Env, buffer:ExperienceReplay, net:NN):
        self.env = env
        self.buffer = buffer
        self.net = net
        self.total_reward = 0.0
        self.state = np.zeros(INPUT_SHAPE)
        self.state_buffer = collections.deque(maxlen=4)  # stores last 4 frames
        self.current_life = 5
        self.reset_dqn()

    def reset_dqn(self):
        # Reset the environment and start with action 1 to fire the ball
        self.state, _ = self.env.reset(seed=42)
        self.state, _, _, _, _ = self.env.step(1)
        self.state = turn_to_grayscale(self.state)
        self.total_reward = 0.0
        self.state_buffer.clear()
        self.state_buffer.append(self.state)

    @torch.no_grad()
    def play_episode(self):
        # Runs one full episode and collects experiences
        is_done = False
        is_trunc = False
        final_reward = 0.0
        frame_cnt = 0

        while True:
            frame_cnt += 1

            # If we have 4 frames, use the DQN to predict action
            if len(self.state_buffer) == 4 and np.random.random() > EPSILON:
                state_stack = np.array(self.state_buffer)
                state_val = torch.as_tensor(state_stack).to(DEVICE).unsqueeze(0)
                q_val = self.net(state_val)
                _, act_v = torch.max(q_val, dim=1)
                action = int(act_v.item())
            else:
                # Otherwise take a random action (exploration)
                action = self.env.action_space.sample()

            # Execute action in the environment
            next_state, reward, is_done, is_trunc, info = self.env.step(action)

            # End episode early if life is lost
            if info['lives'] < 5:
                is_done = True

            next_state = turn_to_grayscale(next_state)
            self.total_reward += reward

            # Store experience only if we have full 4-frame stack
            if len(self.state_buffer) == 4:
                state_stack = np.array(self.state_buffer)
                next_state_stack = np.append(state_stack[1:], [next_state], axis=0)
                exp = Experience(state_stack, action, reward, next_state_stack, (is_done or is_trunc))
                self.buffer.add_experience(exp)

            self.state_buffer.append(next_state)
            self.state = next_state

            if is_done or is_trunc:
                final_reward = self.total_reward
                self.reset_dqn()
                break

        return final_reward, frame_cnt

# Convert RGB image to 84x84 grayscale
def turn_to_grayscale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (INPUT_SHAPE[1], INPUT_SHAPE[2]), interpolation=cv2.INTER_AREA)
    return gray

# Compute DQN loss using Bellman equation
def calculate_loss(batch, net: DQN, target_net: DQN):
    states, actions, rewards, next_states, dones = [], [], [], [], []

    for e in batch:
        states.append(e.state)
        actions.append(e.action)
        rewards.append(e.reward)
        next_states.append(e.next_state)
        dones.append(e.done)
    
    # Convert to tensors
    states_t = torch.as_tensor(np.asarray(states)).to(DEVICE)
    actions_t = torch.LongTensor(actions).to(DEVICE)
    rewards_t = torch.FloatTensor(rewards).to(DEVICE)
    new_states_t = torch.as_tensor(np.asarray(next_states)).to(DEVICE)
    dones_t = torch.BoolTensor(dones).to(DEVICE)

    # Q(s,a)
    state_action_values = net(states_t).gather(1, actions_t.unsqueeze(-1)).squeeze(-1)

    # Q(s',a') from target network
    with torch.no_grad():
        next_state_values = target_net(new_states_t).max(1)[0]
        next_state_values[dones_t] = 0.0  # zero for terminal states

    expected_state_action_values = next_state_values * GAMMA + rewards_t
    return nn.MSELoss()(state_action_values, expected_state_action_values)

# ========== Main Training Loop ==========
if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    INPUT_SHAPE = (4, 84, 84)  # 4 grayscale frames stacked
    MAX_EXP_BUFFER_SIZE = 1_000_000
    MIN_BUFFER_TRAIN_SIZE = 50_000
    SYNC_FRAME = 1000  # target_net sync interval
    SYNC_CNT = 0

    GAMMA = 0.99
    LEARNING_RATE = 0.0001
    BATCH_SIZE = 64

    EPSILON_START = 1.0
    EPSILON_END = 0.1
    EPSILON_DECAY_RATE = 1e-8
    EPSILON = 1.0

    FRAME_CNT = 0
    EPISODE_CNT = 0
    best_reward = 0

    # Create directory for saving models
    save_path = os.path.join("saves")
    os.makedirs(save_path, exist_ok=True)

    # Initialize Atari Breakout environment
    env = gym.make("ALE/Breakout-v5", render_mode="rgb_array", repeat_action_probability=0.0)

    writer = SummaryWriter(comment="-DQN_AGENT")  # TensorBoard logger
 
    # Initialize policy and target networks
    net = NN(INPUT_SHAPE, env.action_space.n).to(DEVICE)
    target_net = NN(INPUT_SHAPE, env.action_space.n).to(DEVICE)
    print(f"The neural net architecture: {net}")

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    exp_buffer = ExperienceReplay(MAX_EXP_BUFFER_SIZE)
    dqn_agent = DQN(env, exp_buffer, net)

    # Training loop
    while True:
        reward, frame_cnt = dqn_agent.play_episode()
        FRAME_CNT += frame_cnt
        SYNC_CNT += frame_cnt 
        EPISODE_CNT += 1

        print(f"EPISODE NO: {EPISODE_CNT} || Current Frames: {FRAME_CNT} || Reward: {reward}")
        EPSILON = max(EPSILON_END, EPSILON - (EPSILON_DECAY_RATE * FRAME_CNT))

        # Save model if new best reward achieved
        if reward > best_reward:
            best_reward = reward
            print(f"BEST REWARD UPDATE: {best_reward} at FRAME CNT: {FRAME_CNT}")
            torch.save(net.state_dict(), os.path.join(save_path, f"dqn_best_{best_reward}"))

        # Break if task solved
        if best_reward > 800:
            print(f"Solved! Final Frame Count: {FRAME_CNT}")
            break

        # Log to TensorBoard
        writer.add_scalar("epsilon", EPSILON, FRAME_CNT)
        writer.add_scalar("reward", best_reward, FRAME_CNT)

        # Update target network
        if SYNC_CNT >= SYNC_FRAME:
            target_net.load_state_dict(net.state_dict())
            print('Copied')
            SYNC_CNT = 0

        # Skip training if not enough experiences
        if len(exp_buffer) < MIN_BUFFER_TRAIN_SIZE:
            continue

        # Sample batch and update Q-network
        optimizer.zero_grad()
        batch = exp_buffer.sample_experiences(BATCH_SIZE)
        loss_t = calculate_loss(batch, net, target_net)
        loss_t.backward()
        optimizer.step()

        writer.add_scalar("loss", loss_t.item(), FRAME_CNT)

    # Save final model and close
    torch.save(net.state_dict())
    writer.close()
    env.close()
