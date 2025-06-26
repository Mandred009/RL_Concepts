# A2C method for Cartpole. Converged under 30 min on GTX 1650.

# Imports
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard.writer import SummaryWriter
import collections
import time


# Neural network with shared layers and separate heads for policy and value
class NN(nn.Module):
    def __init__(self, input_size, n_actions):
        super().__init__()
        self.shared = nn.Sequential(  # Shared body
            nn.Linear(input_size, 128),
            nn.ReLU()
        )
        self.policy_head = nn.Sequential(  # Actor head
            nn.Linear(128, n_actions),
            nn.Softmax(dim=1)
        )
        self.value_head = nn.Linear(128, 1)  # Critic head

    def forward(self, x):
        base = self.shared(x)
        policy = self.policy_head(base)
        value = self.value_head(base)
        return policy, value


# A2C agent to play episodes and collect experience
class A2C:
    def __init__(self, env: gym.Env, net: NN):
        self.state = []
        self.action = []
        self.reward = []
        self.env = env
        self.net = net
        self.episode_no = 0

    def reset_buffer(self):
        self.state = []
        self.action = []
        self.reward = []

    def calc_qvals(self):
        # Computes discounted return (Q values)
        res = []
        sum_r = 0.0
        for r in reversed(self.reward):
            sum_r = r + GAMMA * sum_r
            res.append(sum_r)
        return list(reversed(res))

    @torch.no_grad
    def play_episode(self):
        self.reset_buffer()
        state, _ = self.env.reset()
        while True:
            s_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            policy, _ = self.net(s_tensor)
            probs = policy.squeeze(0).cpu().detach().numpy()
            action = np.random.choice(N_ACTIONS, p=probs)

            self.state.append(state)
            self.action.append(action)

            next_state, reward, terminated, truncated, _ = self.env.step(action)
            self.reward.append(reward)

            if terminated or truncated:
                break
            state = next_state

        self.episode_no += 1
        q_vals = self.calc_qvals()
        total_r = sum(self.reward)
        print(f"Episode No: {self.episode_no} Reward: {total_r}")
        return q_vals, self.state, self.action, total_r


# ===================== Main Training Script ===================== #
if __name__ == "__main__":
    GAMMA = 0.99                         # Discount factor
    LEARNING_RATE = 0.01                # Learning rate for optimizer
    EPISODES_BEFORE_TRAINING = 4        # Batch size for training
    ENTROPY_BETA = 0.01                 # Coefficient for entropy regularization
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {DEVICE}")

    # Environment setup
    env = gym.make("CartPole-v1", render_mode="human")
    N_ACTIONS = env.action_space.n

    # Initialize actor-critic network
    net = NN(env.observation_space.shape[0], N_ACTIONS).to(DEVICE)
    print(f"The neural net architecture: {net}")

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    a2c_agent = A2C(env, net)

    # Buffers for experience
    batch_states, batch_qvals, batch_actions = [], [], []
    writer = SummaryWriter(comment="-cartpole")

    steps = 0
    reward_deque = collections.deque(maxlen=4)
    mean_reward = 0

    start_time = time.time()

    while True:
        q_v, s, a, t_r = a2c_agent.play_episode()

        reward_deque.append(t_r)
        mean_reward = sum(reward_deque) / len(reward_deque)
        print(f"Mean Reward: {mean_reward}")

        if mean_reward > 490:
            print(f"Solved in {a2c_agent.episode_no} Episodes")
            break

        batch_states.append(s)
        batch_qvals.append(q_v)
        batch_actions.append(a)

        if a2c_agent.episode_no % EPISODES_BEFORE_TRAINING == 0:
            optimizer.zero_grad()

            # Flatten batches for training
            flat_states = [item for episode in batch_states for item in episode]
            flat_qvals = [item for episode in batch_qvals for item in episode]
            flat_actions = [item for episode in batch_actions for item in episode]

            states_v = torch.FloatTensor(flat_states).to(DEVICE)
            actions_v = torch.LongTensor(flat_actions).to(DEVICE)
            qvals_v = torch.FloatTensor(flat_qvals).to(DEVICE)

            # Normalize Q-values to stabilize learning
            qvals_v = (qvals_v - qvals_v.mean()) / (qvals_v.std() + 1e-8)

            # Forward pass
            logits_v, value_v = net(states_v)

            # Critic loss (value regression)
            loss_value_v = F.mse_loss(value_v.squeeze(-1), qvals_v)

            # Advantage estimation: A(s,a) = Q(s,a) - V(s)
            adv_v = qvals_v - value_v.squeeze(-1).detach()

            # Actor loss (policy gradient)
            log_probs = torch.log(logits_v + 1e-8)
            log_prob_actions_v = adv_v * log_probs[range(len(actions_v)), actions_v]
            loss_policy_v = -log_prob_actions_v.mean()

            # Entropy loss to encourage exploration
            entropy_loss_v = -ENTROPY_BETA * (logits_v * log_probs).sum(dim=1).mean()

            # Total loss = actor + critic + entropy
            loss_total = loss_value_v + loss_policy_v + entropy_loss_v

            # Backpropagation
            loss_total.backward()
            optimizer.step()

            # TensorBoard logging
            writer.add_scalar("reward_mean", mean_reward, steps)
            writer.add_scalar("episode_no", a2c_agent.episode_no, steps)
            writer.add_scalar("loss_value(critic)", loss_value_v, steps)
            writer.add_scalar("loss_policy(actor)", loss_policy_v, steps)
            writer.add_scalar("loss_entropy", entropy_loss_v, steps)
            writer.add_scalar("loss", loss_total, steps)

            # Clear batch buffers
            batch_states, batch_qvals, batch_actions = [], [], []
        
        steps += len(q_v)  # Step count for logging

    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print(f"Time taken for complete training: {elapsed_time:.2f} min")
    
    writer.close()
