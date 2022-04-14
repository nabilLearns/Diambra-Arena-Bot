from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
import cv2 
import os

import diambraArena
import argparse
from diambraArena.gymUtils import showGymObs

'''
Q = Our net's output
x = image
a = action
r = reward
s = sequence
phi = processed s
gamma = discount factor
D = replay buffer
'''

# preprocessing
# need to rescale image, generate

# Image - Network
        
class metaData(nn.Module):
    """Network produces part of the input vector for finalNetwork from numerical metadata (i.e. opponent health, player health)"""
    
    def __init__(self, hidden_sz, n_outputs):        
        super(metaData, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(20, hidden_sz),
            nn.ReLU(),
            nn.Linear(hidden_sz, hidden_sz) ,
            nn.ReLU(),
            nn.Linear(hidden_sz, n_outputs)
        )

    def forward(self, inp):
        return self.linear_relu_stack(inp)
        
class finalNetwork(nn.Module):
    def __init__(self, c_in, hidden_sz, n_actions=72):
        super(finalNetwork, self).__init__()
        self.dense_network = nn.Sequential(
            nn.Linear(c_in, hidden_sz),
            nn.ReLU(),
            nn.Linear(hidden_sz, n_actions)
        )
            
    def forward(self, inp):
        return self.dense_network(inp)
        
    def predict(self, inp):  
        out = self.forward(inp)
        top_k, top_k_idxs = torch.topk(out, k=10)
        top_k, tok_k_idxs = top_k.detach().numpy(), top_k_idxs.detach().numpy()
        action = np.random.choice(top_k_idxs, p= top_k/ top_k.sum())
        return action#self.forward(inp).argmax() 
    
# DQN Algorithm

def preprocess(observation):
    """given an obser	vation and past 5 frames give us the preprocessed input"""
    metaData = observation['P1']
    actions = metaData['actions']
    meta = torch.zeros(20)
    meta[:3] = torch.tensor([metaData['ownSide'], metaData['ownHealth'], metaData['oppHealth']])
    meta[2 + actions['move']] = 1
    meta[11 + actions['attack']] = 1
    output = {'meta': meta}
    return output

def SELECT_ACTION(epsilon, inp):
    """Selects random action w/ prob epsilon, otherwise picks the action of highest value from the Q-Network"""
    
    action_type = np.random.choice([0, 1], p=[epsilon, 1-epsilon])
    if action_type == 0:
        #print("DOING RANDOM")
        action = env.action_space.sample()#["P1"]
    else:
        metaout = metanet(inp['meta'])
        action = finalnet.predict(metaout)   #f (observation)   # select best action from output of Q net-work
        action = np.array([action.item()//8, action.item()%8])
    return action

def LOSS(y, Q):
    return nn.functional.mse_loss(y, Q)

def TRAIN_STEP(REPLAY_MEMORY):
    inds = np.random.choice(len(REPLAY_MEMORY), num_sample, replace=False)
    replay_vals = [REPLAY_MEMORY[ind] for ind in inds]
    y_js = torch.zeros(num_sample)
    Q_js = torch.zeros(num_sample)
    optimizer.zero_grad()
    for i in range(len(replay_vals)):
        if not (replay_vals[i][4]):
            metaout = metanet(replay_vals[i][3]['meta'])
            y_js[i] = replay_vals[i][2] + discount_rate*(torch.max(finalnet(metaout)))
        else:
            y_js[i] = replay_vals[i][2]
        metaout = metanet(replay_vals[i][0]['meta'])
        ind = replay_vals[i][1][0]*8 + replay_vals[i][1][1]
        Q_js[i] = finalnet(metaout)[ind]
    myloss = LOSS(y_js, Q_js)
    myloss.backward()
    optimizer.step()
    
learning_rate = 0.01
metanet = metaData(32, 16)
finalnet = finalNetwork(16, 64, 72)
optimizer = optim.Adam(list(metanet.parameters()) + list(finalnet.parameters()), learning_rate)
num_sample = 32
discount_rate = 0.6
rolling_frames = []

def RUN_DQN_ALGORITHM(num_episodes, num_time_steps, eps=1, min_eps=0.05):
    """
    Runs through DQN algo. as described in https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
    
    Returns
    -------
    cumulative_reward (list): list of cumulative rewards earned in every round of training 
    
    Parameters
    ----------
    eps (float): probability of selecting a random action. this can decay as the learning process progresses (i.e. do more exploration initially, then less later on) 
    
    Variables
    ---------
    phi_t (dict): Stores processed observations in the form {'frames': framedata, 'meta': metadata}
    """
    REPLAY_MEMORY = []
    all_cumulative_rewards = []
    for episode in range(num_episodes):
        #print("REPLAY MEM", len(REPLAY_MEMORY))

        if episode != 0:
            observation = env.reset()
        else:
            observation = init_observation
        cumulative_reward = [0]
        currRound = 0
        phi_t = None
        steps = 0
        
        while True:
            steps += 1
            #print(f"step: {steps}, eps: {eps}")
            
            #if steps % 50 == 0:
            #    print(f"OBSERVATION: {observation}")
            phi_t = preprocess(observation)
                
            eps = max(eps*0.995, min_eps)
            action = SELECT_ACTION(eps, phi_t)
            #print("ACTION: ", action, type(action))
            #actions = np.append(action, env.action_space.sample()["P2"])

            observation, reward, done, info = env.step(action)
            cumulative_reward[currRound] += reward
            
            if (info['roundDone']):
                print(f"Cumulative Round Reward {cumulative_reward[currRound]}, Average Reward per Step: {cumulative_reward[currRound]/steps}")
                #cumulative_reward[currRound] = np.array(cumulative_reward[currRound])
                currRound += 1
                cumulative_reward.append(0)
                phi_t = None
            if (phi_t != None):
                phi_tplus1 = preprocess(observation)
                newtuple = (phi_t, action, reward, phi_tplus1, info['roundDone'])
                REPLAY_MEMORY.append(newtuple)
                if (len(REPLAY_MEMORY) == 501):
                    REPLAY_MEMORY = REPLAY_MEMORY[1:]
            if (info['gameDone']):
                if cumulative_reward[-1] == 0:
                    cumulative_reward.pop()
                all_cumulative_rewards.append(cumulative_reward)
                print(f"EPISODE {episode} DONE")
                break
            if (len(REPLAY_MEMORY)>50):
                TRAIN_STEP(REPLAY_MEMORY)
    
    
    saved_models_fldr = os.path.join(os.getcwd(), 'saved_models')
    models_path = os.path.join(saved_models_fldr, f'{steps}')
    if not os.path.isdir(saved_models_fldr):
        os.mkdir(saved_models_fldr)
    if not os.path.isdir(models_path):
        os.mkdir(models_path)
    torch.save(metanet, os.path.join(models_path, f"ablation_metanet_{steps}"))            
    torch.save(finalnet, os.path.join(models_path, f"ablation_finalnet_{steps}"))   
            
    return sum(all_cumulative_rewards, [])
    
if __name__ == '__main__':
    #global REPLAY_MEMORY
    #REPLAY_MEMORY = []
    
    settings = {}
    settings["romsPath"] = "/home/nabil/Downloads/"
    settings["gameId"] = "doapp"
    settings["characters"] = [["Hayabusa"], ["Gen-Fu"]]
    settings["player"] = "P1"#P2"
    
    #settings["actionSpace"] = "discrete"
    #settings["attackButCombination"] = False # reduce action space size

    envId = "TestEnv"
    env = diambraArena.make(envId, settings)
    #print("initial obs")
    init_observation = env.reset()
    #showGymObs(init_observation, env.charNames) this brings up annoying black screen
    print(env.action_space)

    cumulative_reward = RUN_DQN_ALGORITHM(num_episodes=10, num_time_steps=10, eps=0.8, min_eps=0.05)
    print(f"CUMULATIVE REWARD {cumulative_reward}")
    results_file = open("dqn_results.txt", "w")
    results_file.write(f"CUMULATIVE REWARD LIST {cumulative_reward} \n AVG CUMULATIVE REWARD PER ROUND {sum(cumulative_reward)/len(cumulative_reward)} \n # Rounds: {len(cumulative_reward)}")
    results_file.close()
