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
class ImageNetwork(nn.Module):
    def __init__(self, hidden_size):
        super(ImageNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(4, 8, kernel_size=5, stride=3),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(start_dim=1)
        )

        self.fc = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, hidden_size)
        )

    def forward(self, in_vals):
        #print(in_vals.shape)
        out = self.conv(in_vals)
        out = self.fc(out)
        return out 
        
        
class TimeLSTM(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_size, num_layers):
        super(TimeLSTM, self).__init__()
        self.size_in = input_shape
        self.size_out = output_shape
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.LSTM = nn.LSTM(input_size=self.size_in, hidden_size=self.hidden_size, num_layers=self.num_layers)
        self.ac1 = nn.LeakyReLU()
        self.FC = nn.Linear(in_features = self.hidden_size, out_features = self.size_out)
        self.ac2 = nn.ReLU()

    def forward(self, in_vals):
        in_vals = in_vals.reshape(in_vals.shape[0], 1, in_vals.shape[1])
        out, (h_n, c_n) = self.LSTM(in_vals)
        #print(h_n.shape)
        output = h_n[-1,:,:]
        output = output.reshape(output.shape[1])
        output = self.ac1(output)
        output = self.FC(output)
        output = self.ac2(output)
        return output

        
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
    def __init__(self, c_in, hidden_sz, n_actions, using_discrete=False):
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
        top_k, top_k_idxs = torch.topk(out, k=10 if not using_discrete else 2)
        top_k, tok_k_idxs = top_k.detach().numpy(), top_k_idxs.detach().numpy()
        action = np.random.choice(top_k_idxs, p= top_k/ top_k.sum())
        return action#self.forward(inp).argmax() 

def preprocess(observation, past_5_frames):
    """given an obser	vation and past 5 frames give us the preprocessed input"""
    past_5_fr = []
    for frame in past_5_frames:
        current_frame = observation['frame']
        small = cv2.resize(current_frame, (128, 128)) 
        img_gray = cv2.cvtColor(np.float32(small), cv2.COLOR_RGB2GRAY)[None, :][None, :] # shape 1 by 1 by 256 by 256
        past_5_fr.append(torch.tensor(img_gray))
    past_5_frames = torch.cat(past_5_fr)
    current_frame = observation['frame']
    metaData = observation['P1']
    actions = metaData['actions']
    small = cv2.resize(current_frame, (128, 128)) 
    img_gray = cv2.cvtColor(np.float32(small), cv2.COLOR_RGB2GRAY)[None, :][None, :] # shape 1 by 1 by 256 by 256
    img_gray = torch.tensor(img_gray)
    frames = torch.row_stack((past_5_frames, img_gray))

    meta = torch.zeros(20)
    meta[:3] = torch.tensor([metaData['ownSide'], metaData['ownHealth'], metaData['oppHealth']])
    meta[2 + actions['move']] = 1
    meta[11 + actions['attack']] = 1
    output = {'frames': frames, 'meta': meta}

    return output
    
# DQN Algorithm

def SELECT_ACTION(epsilon, inp, using_discrete = False):
    """Selects random action w/ prob epsilon, otherwise picks the action of highest value from the Q-Network"""
    
    action_type = np.random.choice([0, 1], p=[epsilon, 1-epsilon])
    if action_type == 0 or not(inp):
        #print("DOING RANDOM")
        action = env.action_space.sample()
    else:
        frameout1 = imagenet(inp['frames'])
        frameout = timenet(frameout1)
        metaout = metanet(inp['meta'])
        concat_data = torch.cat((frameout, metaout))
        action = finalnet.predict(concat_data)   # select one of top-k actions from output of DQN
        action = action.item() if using_discrete else np.array([action.item()//8, action.item()%8])
        #action = {'move': action//8, 'attack': action%8}
    return action

def LOSS(y, Q):
    return nn.functional.mse_loss(y, Q)

def TRAIN_STEP(REPLAY_MEMORY, discount_rate, using_discrete = False):
    inds = np.random.choice(len(REPLAY_MEMORY), num_sample, replace=False)
    replay_vals = [REPLAY_MEMORY[ind] for ind in inds]
    y_js = torch.zeros(num_sample)
    Q_js = torch.zeros(num_sample)
    optimizer.zero_grad()
    for i in range(len(replay_vals)):
        if not (replay_vals[i][4]):
            frameout1 = imagenet(replay_vals[i][3]['frames'])
            frameout = timenet(frameout1)
            metaout = metanet(replay_vals[i][3]['meta'])
            concat_data = torch.cat((frameout, metaout))
            y_js[i] = replay_vals[i][2] + discount_rate*(torch.max(finalnet(concat_data)))
        else:
            y_js[i] = replay_vals[i][2]
        frameout1 = imagenet(replay_vals[i][0]['frames'])
        frameout = timenet(frameout1)
        metaout = metanet(replay_vals[i][0]['meta'])
        concat_data = torch.cat((frameout, metaout))
        ind = replay_vals[i][1][0]*8 + replay_vals[i][1][1] if not using_discrete else replay_vals[i][1]
        Q_js[i] = finalnet(concat_data)[ind]
    myloss = LOSS(y_js, Q_js)
    myloss.backward()
    optimizer.step()
    #print(finalnet.parameters())
    
using_discrete = True
learning_rate = 0.01
imagenet = ImageNetwork(64)
timenet = TimeLSTM(64, 16, 32, 1)
metanet = metaData(32, 16)
finalnet = finalNetwork(32, 64, n_actions=16 if using_discrete else 72, using_discrete)
optimizer = optim.Adam(list(imagenet.parameters()) + list(timenet.parameters()) + list(metanet.parameters()) + list(finalnet.parameters()), learning_rate)
num_sample = 32
#discount_rate = 0.5
rolling_frames = []

def RUN_DQN_ALGORITHM(results_file, num_episodes, num_time_steps, eps=1, min_eps=0.05, discount_rate=0.6, using_discrete=False):
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
        rolling_frames = []
        cumulative_reward = [0]
        currRound = 0
        phi_t = None
        steps = 0
        
        while True:
            steps += 1
            #print(f"step: {steps}, eps: {eps}")
            
            #if steps % 50 == 0:
            #    print(f"OBSERVATION: {observation}")
            if (len(rolling_frames) == 5):
                phi_t = preprocess(observation, rolling_frames)
                            
            eps = max(eps*0.995, min_eps)
            action = SELECT_ACTION(eps, phi_t, using_discrete)
            #print("ACTION: ", action, type(action))
            #actions = np.append(action, env.action_space.sample()["P2"])
            
            rolling_frames.append(observation['frame'])
            if (len(rolling_frames) == 6):
                rolling_frames = rolling_frames[1:]

            observation, reward, done, info = env.step(action)
            cumulative_reward[currRound] += reward
            
            if (info['roundDone']):
                rolling_frames = []
                #print(f"Cumulative Round Reward {cumulative_reward[currRound]}, Average Reward per Step: {cumulative_reward[currRound]/steps}")
                #cumulative_reward[currRound] = np.array(cumulative_reward[currRound])
                currRound += 1
                cumulative_reward.append(0)
                phi_t = None
            if (phi_t != None):
                phi_tplus1 = preprocess(observation, rolling_frames)
                newtuple = (phi_t, action, reward, phi_tplus1, info['roundDone'])
                REPLAY_MEMORY.append(newtuple)
                if (len(REPLAY_MEMORY) == 501):
                    REPLAY_MEMORY = REPLAY_MEMORY[1:]
            if (info['gameDone']):
                if cumulative_reward[-1] == 0:
                    cumulative_reward.pop()
                all_cumulative_rewards.append(cumulative_reward)
                print(f"EPISODE {episode} DONE")
                print(cumulative_reward)
                break
            if (len(REPLAY_MEMORY)>50):
                TRAIN_STEP(REPLAY_MEMORY, discount_rate, using_discrete)
            #if steps == 100:
            #    break
    
    
    saved_models_fldr = os.path.join(os.getcwd(), 'saved_models')
    models_path = os.path.join(saved_models_fldr, f'{steps}')
    if not os.path.isdir(saved_models_fldr):
        os.mkdir(saved_models_fldr)
    if not os.path.isdir(models_path):
        os.mkdir(models_path)
    torch.save(imagenet, os.path.join(models_path, f"imagenet_{steps}_g={discount_rate}"))            
    torch.save(timenet, os.path.join(models_path, f"timenet_{steps}_g={discount_rate}"))            
    torch.save(metanet, os.path.join(models_path, f"metanet_{steps}_g={discount_rate}"))            
    torch.save(finalnet, os.path.join(models_path, f"finalnet_{steps}_g={discount_rate}"))
    
    cumulative_reward = sum(all_cumulative_rewards, [])
    print(f"CUMULATIVE REWARD:\n{cumulative_reward}")
    results_file.write(f"Experiment -- gamma={discount_rate}:\n")
    results_file.write("----------------------\n")
    results_file.write(f"num_rounds = {len(cumulative_reward)}\n")
    results_file.write(f"avg_cumulative_rew_per_round = {sum(cumulative_reward)/len(cumulative_reward)}\n")
    results_file.write(f"{cumulative_reward} \n\n")
    
    #results_file = open(f"dqn_results_g={gamma}.txt", "w")
    #results_file.write(f"Discount Exp -- gamma={discount_rate}: CUMULATIVE REWARD LIST {cumulative_reward} \n AVG CUMULATIVE REWARD PER ROUND {sum(cumulative_reward)/len(cumulative_reward)} \n # Rounds: {len(cumulative_reward)}")
    #results_file.close()
        
    #return sum(all_cumulative_rewards, [])
    
if __name__ == '__main__':
    settings = {}
    settings["romsPath"] = "/home/nabil/Downloads/"
    settings["gameId"] = "doapp"
    settings["characters"] = [["Hayabusa"], ["Gen-Fu"]]
    settings["player"] = "P1"
    settings["actionSpace"] = "discrete" # reduce action space size to 16 from 72 (multidiscrete w/ comb.)
    settings["attackButCombination"] = True

    envId = "TestEnv"
    env = diambraArena.make(envId, settings)
    #print("initial obs")
    init_observation = env.reset()
    #showGymObs(init_observation, env.charNames) this brings up annoying black screen
    print(env.action_space)
    
    results_file = open(f"dqn_results.txt", "w")
    RUN_DQN_ALGORITHM(results_file, num_episodes=10, num_time_steps=10, eps=0.8, min_eps=0.05, discount_rate = 0.3, using_discrete = True)
    RUN_DQN_ALGORITHM(results_file, num_episodes=10, num_time_steps=10, eps=0.8, min_eps=0.05, discount_rate = 0.6, using_discrete = True)
    #RUN_DQN_ALGORITHM(results_file, num_episodes=10, num_time_steps=10, eps=0.8, min_eps=0.05, discount_rate = 0.7, using_discrete = True)
    RUN_DQN_ALGORITHM(results_file, num_episodes=10, num_time_steps=10, eps=0.8, min_eps=0.05, discount_rate = 0.9, using_discrete = True)
    results_file.close()
    
    #cumulative_reward = RUN_DQN_ALGORITHM(num_episodes=10, num_time_steps=10, eps=0.8, min_eps=0.05)
    #print(f"CUMULATIVE REWARD {cumulative_reward}")
    #results_file = open("dqn_results.txt", "w")
    #results_file.write(f"CUMULATIVE REWARD LIST {cumulative_reward} \n AVG CUMULATIVE REWARD PER ROUND {sum(cumulative_reward)/len(cumulative_reward)} \n # Rounds: {len(cumulative_reward)}")
    #results_file.close()
