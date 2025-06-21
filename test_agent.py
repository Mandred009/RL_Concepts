## CODE TO TEST THE DIFFERENT AGENTS THAT HAVE BEEN TRAINED

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

from ppo import PPO_ACTOR
from sac import SAC_ACTOR
from t3d_pgm import T3D_ACTOR


@torch.no_grad()
def test_agent(env: gym.Env, actor_net: nn.Module):
    total_r = 0
    actor_net.eval()

    for t in range(NO_OF_TESTS):
        print(f"test:{t}")
        state, _ = env.reset()
        while True:
            s_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            mu,_= actor_net(s_tensor) # Change this depending on testing net
            action = torch.tanh(mu).squeeze(0).cpu().numpy()

            next_state, reward, is_done, is_trunc, _ = env.step(action)
            if is_done or is_trunc:
                break
            state = next_state

            total_r += reward

    return total_r / NO_OF_TESTS

if __name__=="__main__":
    NO_OF_TESTS=1
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(DEVICE)

    env = gym.make("HalfCheetah-v5",render_mode='human') # Change envrionment depending on network
    N_ACTIONS = env.action_space.shape[0]

    # ppo_agent=PPO_ACTOR(env.observation_space.shape[0], N_ACTIONS).to(DEVICE)

    # ppo_agent.load_state_dict(torch.load("saves/ppo_best",weights_only=True))

    sac_agent=SAC_ACTOR(env.observation_space.shape[0], N_ACTIONS).to(DEVICE)

    sac_agent.load_state_dict(torch.load("saves/sac_best",weights_only=True))

    # t3d_agent=T3D_ACTOR(env.observation_space.shape[0], N_ACTIONS).to(DEVICE)

    # t3d_agent.load_state_dict(torch.load("saves/t3d_best",weights_only=True))

    print(test_agent(env,sac_agent))
