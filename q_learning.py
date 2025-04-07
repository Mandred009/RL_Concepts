# Q-Table Learning for FrozenLake Environment. Its a highly stochastic method and sometimes takes 10 and sometimes 200 iterations to converge

import numpy as np
import gymnasium as gym
from dataclasses import dataclass
from torch.utils.tensorboard.writer import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim

import time

class Q_Agent:
    def __init__(self,no_of_states,no_of_actions,env:gym.Env):
        self.env=env
        self.q_table=np.zeros((no_of_states,no_of_actions)) # Q values array
        self.transition_table=np.zeros((no_of_states,no_of_states)) # Transition count table
        self.reward_dict={} # Dict to store rewards and states action pair keys

    def play_n_random_episodes(self,n): # Random exploration
        for i in range(0,n):
            state,_=self.env.reset()

            is_done=False
            is_trunc=False

            while True:
                action = self.env.action_space.sample()
                new_state,reward,is_done,is_trunc,_=self.env.step(action)

                self.reward_dict[(state,action,new_state)]=reward
                self.transition_table[state][new_state]+=1
                state=new_state

                if is_done or is_trunc:
                    break 
        


    def play_episode(self): # Takes steps based on q-table
        total_reward=0
        state,_=self.env.reset()

        is_done=False
        is_trunc=False

        while True:
            action = self.select_action(state)
            new_state,reward,is_done,is_trunc,_=self.env.step(action)

            self.reward_dict[(state,action,new_state)]=reward
            self.transition_table[state][new_state]+=1
            state=new_state
            total_reward+=reward

            if is_done or is_trunc:
                break 

        return total_reward



    def value_iteration(self): # Iterate and update the q-table
        for s in range(0,NO_OF_STATES):
            for a in range(0,NO_OF_ACTIONS):
                q_val=0.0
                total_transitions=self.transition_table.sum()
                for s_ in range(0,NO_OF_STATES):
                    if self.transition_table[s][s_]>0:
                        try:
                            reward=self.reward_dict[(s,a,s_)]
                        except:
                            continue
                        next_action=self.select_action(s_)
                        q_v=reward+(GAMMA*self.q_table[s_][next_action])
                        q_val+=(self.transition_table[s][s_]/total_transitions)*q_v
                
                self.q_table[s][a]=q_val



    def select_action(self,state): # pick the action with best q-val
        best_action=None
        best_value=None

        for a in range(0,NO_OF_ACTIONS):
            val=self.q_table[state][a]
            if best_value==None or best_value<val:
                best_action=a
                best_value=val
        return best_action

if __name__=="__main__":

    GAMMA=0.92 # Discount 
    N_EPISODES=10 # Random exploration episodes
    TEST_EPISODES=20 # Testing episodes
    BREAK_REWARD=0.8 # That is average reward of all test episodes if greater than this then solved

    env=gym.make("FrozenLake-v1",render_mode="human")
    env.reset()
    env.render()

    NO_OF_STATES=env.observation_space.n
    NO_OF_ACTIONS=env.action_space.n

    q_agent=Q_Agent(NO_OF_STATES,NO_OF_ACTIONS,env)

    iter_no=0
    best_avg_reward=0.0

    while True:
        iter_no+=1
        print(f"ITER NO: {iter_no}")

        q_agent.play_n_random_episodes(N_EPISODES)
        q_agent.value_iteration()

        reward_sum=0

        
        for i in range(0,TEST_EPISODES):
            reward_sum+=q_agent.play_episode()
            

        avg_reward=reward_sum/TEST_EPISODES

        if avg_reward>best_avg_reward:
            print(f"Best Reward Updated Iter No {iter_no} || {best_avg_reward} >>> {avg_reward}")
            best_avg_reward=avg_reward

        if avg_reward>=BREAK_REWARD:
            print(f"Solved at Iteration: {iter_no}")
            break



    