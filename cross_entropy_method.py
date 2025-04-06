import numpy as np
import gymnasium as gym
from dataclasses import dataclass
from torch.utils.tensorboard.writer import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim

import time


@dataclass
class Episode:
    states: list
    actions: list
    reward_sum: float


class NN(nn.Module):
    def __init__(self, inp_size, hidden_size, out_size):
        super().__init__()

        self.nn=nn.Sequential(
            nn.Linear(inp_size,hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size,out_size),
            nn.Softmax(dim=1)
        )

    def forward(self,x):
        return self.nn(x)
    

class cross_entropy_rl:
    def __init__(self, env:gym.Env, nn, num_of_episodes, reward_percentile):
        self.env=env
        self.nn_net=nn
        self.num_of_episodes=num_of_episodes
        self.reward_percentile=reward_percentile

        self.episodes=[]

    
    def play_n_episodes(self):
        
        for i in range(0,self.num_of_episodes):
            states=[]
            rewards=[]
            actions=[]
            obs,_=self.env.reset()
            is_done=False
            is_trunc=False
            
            while not is_done and not is_trunc:
                obs_v=torch.tensor(obs,dtype=torch.float32).unsqueeze(0).to(device) # Unsqueeze add extra dim ie (4) to (1,4)
                action_prob=self.nn_net(obs_v).data.cpu().numpy()[0] #.data extracts the tensor from the output not the gradients
                action = np.random.choice(len(action_prob), p=action_prob) # select based on action probabilities
                next_obs, reward, is_done, is_trunc, _ = env.step(action)
                
                states.append(obs)
                rewards.append(reward)
                actions.append(action)

                obs=next_obs
            
            reward_sum=sum(rewards)
            # print(f"Steps: {len(states)}")
            self.episodes.append(Episode(states,actions,reward_sum))


    def filter_episodes(self):
        epi_rewards=[x.reward_sum for x in self.episodes]
        reward_band=np.percentile(epi_rewards,self.reward_percentile)
        
        train_obs=[]
        train_action=[]

        good_episodes=[]

        for i in self.episodes:
            reward=i.reward_sum

            if reward>=reward_band:
                good_episodes.append(i)
                for j in range(0,len(i.states)):
                    train_obs.append(i.states[j])
                    train_action.append(i.actions[j])
        
        mean_reward=sum(epi_rewards)/len(epi_rewards)

        self.episodes=good_episodes

        train_obs_tensor=torch.tensor(np.stack(train_obs),dtype=torch.float32).to(device)
        train_action_tensor=torch.tensor(np.stack(train_action),dtype=torch.long).to(device)
        # self.episodes=[]
    
        return train_obs_tensor,train_action_tensor,mean_reward,reward_band





if __name__=="__main__":
    EPOCHS=100
    EPISODE_PLAY=50
    REWARD_PERCENTILE=90

    env=gym.make("CartPole-v1",render_mode="human")
    env.reset()
    env.render()
    
    obs_size = env.observation_space.shape[0]
    n_actions = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    nn_net=NN(obs_size,128,n_actions).to(device)
    print(f"The neural net archi: {nn_net}")

    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=nn_net.parameters(), lr=0.1)
    writer = SummaryWriter(comment="-cartpole")

    cross_en_m=cross_entropy_rl(env,nn_net,EPISODE_PLAY,REWARD_PERCENTILE)

    start_time = time.time()
    for i in range(0,EPOCHS):
        cross_en_m.play_n_episodes()
        train_obs,train_action,mean_reward,band_reward=cross_en_m.filter_episodes()

        optimizer.zero_grad()
        action_scores_v = nn_net(train_obs)
        loss_v = objective(action_scores_v, train_action)
        loss_v.backward()
        optimizer.step()

        print(f"Epoch: {i} || Loss: {loss_v.item()} || Reward Mean: {mean_reward} || Reward Band: {band_reward}")

        writer.add_scalar("loss", loss_v.item(), i)
        writer.add_scalar("reward_mean", mean_reward, i)

        if mean_reward>=490:
            break

    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = (end_time - start_time)/60

    # Print the elapsed time
    print(f"Time taken for {EPOCHS} epochs: {elapsed_time:.2f} min")
    writer.close()
    print("Done")


