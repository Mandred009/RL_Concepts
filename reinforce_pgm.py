# REINFORCE method for Cartpole. Converged.

# Libraries
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter
import collections
import time

# Policy Network – outputs action probabilities
class NN(nn.Module):
    def __init__(self, input_size, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
            nn.Softmax(dim=1)  # Converts output to action probability distribution
        )

    def forward(self, x):
        return self.net(x)

# REINFORCE Agent class
class REINFORCE:
    def __init__(self, env: gym.Env, net: NN):
        self.state = []     # states visited in an episode
        self.action = []    # actions taken
        self.reward = []    # rewards received
        self.env = env
        self.net = net
        self.episode_no = 0

    def reset_buffer(self):
        # Clear episode data
        self.state = []
        self.action = []
        self.reward = []

    def calc_qvals(self):
        # Calculate cumulative discounted rewards (return) for each timestep
        res = []
        sum_r = 0.0
        for r in reversed(self.reward):
            sum_r = r + GAMMA * sum_r
            res.append(sum_r)
        return list(reversed(res))

    @torch.no_grad
    def play_episode(self):
        # Play one episode using current policy
        self.reset_buffer()
        state, _ = self.env.reset()

        while True:
            s_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            probs = self.net(s_tensor).squeeze(0).cpu().detach().numpy()
            action = np.random.choice(N_ACTIONS, p=probs)  # Sample action based on policy

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


# ===================== Main Training Loop ===================== #
if __name__ == "__main__":
    GAMMA = 0.99                        # Discount factor
    LEARNING_RATE = 0.01               # Optimizer learning rate
    EPISODES_BEFORE_TRAINING = 4       # Batch size for policy updates
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {DEVICE}")

    env = gym.make("CartPole-v1", render_mode="human")
    N_ACTIONS = env.action_space.n     # Number of discrete actions (2 for CartPole)

    # Initialize policy network
    net = NN(env.observation_space.shape[0], N_ACTIONS).to(DEVICE)
    print(f"The neural net architecture: {net}")

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    reinforce = REINFORCE(env, net)

    # Buffers to accumulate training data over multiple episodes
    batch_states, batch_qvals, batch_actions = [], [], []
    writer = SummaryWriter(comment="-cartpole")

    steps = 0
    reward_deque = collections.deque(maxlen=4)  # Used to compute moving average reward
    mean_reward = 0

    start_time = time.time()

    while True:
        # Play one episode and collect data
        q_v, s, a, t_r = reinforce.play_episode()

        reward_deque.append(t_r)
        mean_reward = sum(reward_deque) / len(reward_deque)
        print(f"Mean Reward: {mean_reward}")

        if mean_reward > 490:
            print(f"Solved in {reinforce.episode_no} Episodes")
            break

        batch_states.append(s)
        batch_qvals.append(q_v)
        batch_actions.append(a)

        # Train policy every few episodes
        if reinforce.episode_no % EPISODES_BEFORE_TRAINING == 0:
            optimizer.zero_grad()

            # Flatten all episodes into a single batch
            flat_states = [item for episode in batch_states for item in episode]
            flat_qvals = [item for episode in batch_qvals for item in episode]
            flat_actions = [item for episode in batch_actions for item in episode]

            states_v = torch.FloatTensor(flat_states).to(DEVICE)
            actions_v = torch.LongTensor(flat_actions).to(DEVICE)
            qvals_v = torch.FloatTensor(flat_qvals).to(DEVICE)

            # Normalize Q-values to reduce variance
            qvals_v = (qvals_v - qvals_v.mean()) / (qvals_v.std() + 1e-8)

            # Compute log probabilities of taken actions
            logits_v = net(states_v)
            log_probs = torch.log(logits_v + 1e-8)  # Add epsilon to avoid log(0)
            selected_log_probs = log_probs[range(len(actions_v)), actions_v]

            # Policy Gradient Loss = -logπ(a|s) * G_t
            loss = -(selected_log_probs * qvals_v).mean() # Negative sign means we need to do gradient ascent.

            loss.backward()
            optimizer.step()

            # Log metrics to TensorBoard
            writer.add_scalar("reward_mean", mean_reward, steps)
            writer.add_scalar("episode_no", reinforce.episode_no, steps)
            writer.add_scalar("loss", loss, steps)

            # Reset batch buffers
            batch_states, batch_qvals, batch_actions = [], [], []
        
        steps += len(q_v)  # Approximate number of steps taken in that episode

    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print(f"Time taken for complete training: {elapsed_time:.2f} min")
    
    writer.close()
