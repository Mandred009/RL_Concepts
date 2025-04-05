import numpy as np
import gymnasium as gym
from dataclasses import dataclass
from torch.utils.tensorboard.writer import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim


class NN(nn.Module):
    def __init__(self, inp_size, hidden_size, out_size):
        super().__init__()

        self.nn=nn.Sequential(
            nn.Linear(inp_size,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,out_size)
        )

    def forward(self,x):
        return self.nn(x)
    

class cross_entropy_rl:
    def __init__(self, env, nn, num_of_episodes, reward_percentile):
        self.env=env
        self.nn=nn
        self.num_of_episodes=num_of_episodes
        self.reward_percentile=reward_percentile

    
    def play_n_episodes():
        pass

    def filter_episodes():
        pass

    



if __name__=="__main__":
    env=gym.make("CartPole-v1")

    print(env.action_space.n)
