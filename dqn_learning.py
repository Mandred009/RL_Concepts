import numpy as np
import gymnasium as gym
from dataclasses import dataclass
from torch.utils.tensorboard.writer import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim

import time

import ale_py

import collections

import cv2


@dataclass
class Experience:
    state: np.array
    action: float
    reward: float
    next_state: np.array
    done: bool


class NN(nn.Module):
    def __init__(self, inp_size, no_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inp_size[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        size = self.conv(torch.zeros(1, *inp_size)).size()[-1]
        self.fc = nn.Sequential(
            nn.Linear(size, 512),
            nn.ReLU(),
            nn.Linear(512, no_actions)
        )

    def forward(self, x):
        # image scaling
        xx = x / 255.0
        return self.fc(self.conv(xx))


class ExperienceReplay:
    def __init__(self, max_buffer_size):
        self.buffer=collections.deque(maxlen=max_buffer_size)
    
    def __len__(self):
        return len(self.buffer)

    def add_experience(self, experience: Experience):
        self.buffer.append(experience)

    def sample_experiences(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[idx] for idx in indices]


class DQN:
    def __init__(self, env: gym.Env, buffer:ExperienceReplay, net:NN):
        self.env=env
        self.buffer=buffer
        self.net=net
        self.total_reward=0.0
        self.state=np.zeros(INPUT_SHAPE)
        self.reset_dqn()

    def reset_dqn(self):
        self.state,_=self.env.reset(seed=42)
        self.state=turn_to_grayscale(self.state)
        self.total_reward=0.0

    def play_episode(self):
        is_done=False
        is_trunc=False
        final_reward=0.0
        frame_cnt=0

        while True:
            frame_cnt+=1
            if np.random.random() < EPSILON:
                action=self.env.action_space.sample()
            else:
                state_val=torch.as_tensor(self.state).to(DEVICE)
                state_val=state_val.unsqueeze(0)
                q_val=self.net(state_val)
                _, act_v = torch.max(q_val, dim=1)
                action = int(act_v.item())
            
            next_state,reward,is_done,is_trunc,_=self.env.step(action)
            next_state=turn_to_grayscale(next_state)
            self.total_reward+=reward

            exp=Experience(self.state,action,reward,next_state,is_done)
            self.buffer.add_experience(exp)

            self.state=next_state

            if is_done or is_trunc:
                final_reward=self.total_reward
                self.reset_dqn()
                break
        
        return final_reward,frame_cnt

def turn_to_grayscale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (INPUT_SHAPE[1], INPUT_SHAPE[2]), interpolation=cv2.INTER_AREA)
    return np.expand_dims(gray, axis=0)  # Add channel dimension


def calculate_loss(batch,net:DQN,target_net:DQN):
    states, actions, rewards, next_states, dones=[], [], [], [], []

    for e in batch:
        states.append(e.state)
        actions.append(e.action)
        rewards.append(e.reward)
        next_states.append(e.next_state)
        dones.append(e.done)
    
    states_t = torch.as_tensor(np.asarray(states)).to(DEVICE)
    actions_t = torch.LongTensor(actions).to(DEVICE)
    rewards_t = torch.FloatTensor(rewards).to(DEVICE)
    new_states_t = torch.as_tensor(np.asarray(next_states)).to(DEVICE)
    dones_t = torch.BoolTensor(dones).to(DEVICE)

    state_action_values = net(states_t).gather(
        1, actions_t.unsqueeze(-1)
    ).squeeze(-1)

    with torch.no_grad():
        next_state_values = target_net(new_states_t).max(1)[0]
        next_state_values[dones_t] = 0.0
        next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * GAMMA + rewards_t
    return nn.MSELoss()(state_action_values, expected_state_action_values)


if __name__=="__main__":
    DEVICE="cuda" if torch.cuda.is_available() else "cpu"
    INPUT_SHAPE=(1,84,84) # Channel Height Width is the format taken by pytorch
    MAX_EXP_BUFFER_SIZE=10000

    SYNC_FRAME=1000# After how many frames target net should gain main net weights
    SYNC_CNT=0

    GAMMA = 0.99
    LEARNING_RATE=0.0001
    BATCH_SIZE=32

    EPSILON_START=1.0
    EPSILON_END=0.01
    EPSILON_DECAY_RATE=1e-6
    EPSILON=1.0

    FRAME_CNT=0
    EPISODE_CNT=0
    best_reward=0

    gym.register_envs(ale_py)

    # Initialise the environment
    env = gym.make("ALE/Breakout-v5", render_mode="human")

    writer = SummaryWriter(comment="-DQN_AGENT")
 
    net=NN(INPUT_SHAPE,env.action_space.n).to(DEVICE)
    target_net=NN(INPUT_SHAPE,env.action_space.n).to(DEVICE)
    print(f"The neural net architecture: {net}")

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    exp_buffer=ExperienceReplay(MAX_EXP_BUFFER_SIZE)


    dqn_agent=DQN(env,exp_buffer,net)

    while True:
        reward,frame_cnt = dqn_agent.play_episode()
        FRAME_CNT += frame_cnt
        SYNC_CNT += frame_cnt 
        EPISODE_CNT+=1
        print(f"EPISODE NO: {EPISODE_CNT} DONE. Current Frames: {FRAME_CNT}")
        EPSILON = max(EPSILON_END, EPSILON - (EPSILON_DECAY_RATE*FRAME_CNT))

        if reward>best_reward:
            best_reward=reward
            print(f"BEST REWARD UPDATE: {best_reward} at FRAME CNT: {FRAME_CNT}")
        
        if best_reward>800:
            print(f"Solved! Final Frame Count: {FRAME_CNT}")
            break

        writer.add_scalar("epsilon", EPSILON, FRAME_CNT)
        writer.add_scalar("reward", best_reward, FRAME_CNT)

        if SYNC_CNT>=SYNC_FRAME:
            target_net.load_state_dict(net.state_dict())
            print('Copied')
            SYNC_CNT=0

        if len(exp_buffer)<MAX_EXP_BUFFER_SIZE:
            continue
        

        optimizer.zero_grad()
        batch=exp_buffer.sample_experiences(BATCH_SIZE)
        loss_t=calculate_loss(batch,net,target_net)
        loss_t.backward()
        optimizer.step()
        writer.add_scalar("loss", loss_t.item(), FRAME_CNT)

    torch.save(net.state_dict())
    writer.close()
    env.close() 