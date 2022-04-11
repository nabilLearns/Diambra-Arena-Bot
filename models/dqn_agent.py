from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import math
import cv2 

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

'''


Moves (0-8): (No-Move, Left, Left+Up, Up, Up+Right, Right, Right+Down, Down, Down+Left) 
Attacks (0-3): (No-Attack, Hold, Punch, Kick)


EXAMPLE INPUT:
actions = env.action_space
observation, reward, done, info = env.step(actions)
observation['frame'] = frame array shape = (480, 512, 3)
observation['P1'] = {'oppChar1': 4, 'ownChar1': 8, 'oppChar': 4, 'ownChar': 8, 'oppHealth': 143, 'ownHealth': 187, 'oppSide': 0, 'ownSide': 1, 'oppWins': 1, 'ownWins': 0, 'actions': {'move': 5, 'attack': 7}}
observation['stage'] = 1


info = {'roundDone': False, 'stageDone': False, 'gameDone': False, 'epDone': False}
Info: {'roundDone': False, 'stageDone': False, 'gameDone': False, 'epDone': False}
observation["frame"].shape: (480, 512, 3)
observation["stage"]: 1
observation["P1"]["ownChar1"]: Kasumi
observation["P1"]["oppChar1"]: Bass
observation["P1"]["ownChar"]: Kasumi
observation["P1"]["oppChar"]: Bass
observation["P1"]["ownHealth"]: 84
observation["P1"]["oppHealth"]: 23
observation["P1"]["ownSide"]: 1
observation["P1"]["oppSide"]: 1
observation["P1"]["ownWins"]: 0
observation["P1"]["oppWins"]: 0
observation["P1"]["actions"]: {'move': 6, 'attack': 0}
Reward: 0.9423076923076923

Done: False
'''


#In: 6 of (480, 512, 3)
#To net: (6, 1, 256, 256)
#Out of imagenet: (6, hidden_size)
#Out of timenet: (hidden_size2)
#Out of metanet: (hidden_size3)
#Out of final net: one-val
#One-val//8 = move, One-vack

"""
# GREYSCALE
img = cv2.imread('sample_out_2.png')
small = cv2.resize(img, (256,256)) 
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
print(img_gray.shape)
plt.imshow(img_gray)
plt.show()
"""

    
# Image - Network
class ImageNetwork(nn.Module):
    def init(self, hidden_size):
        super(ImageNetwork, self).init()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(4, 8, kernel_size=5, stride=3),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(start_dim=1)
        )

        self.fc = nn.Sequential(
            nn.Linear(2592, 512),
            nn.ReLU(),
            nn.Linear(512, hidden_size)
        )

    def forward(self, in_vals):
        print(in_vals.shape)
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
        out, (h_n, c_n) = self.LSTM(in_vals)
        output = h_n[-1,:]
        output = self.ac1(output)
        output = self.FC(output)
        output = self.ac2(output)
        return output
      
      
class metaData(nn.Module):
    def __init__(self, input_shape, hidden_sz, n_outputs):
        """
        input_shape = 12
        """
        super(metaData, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_shape[0], hidden_sz),
            nn.ReLU(),
            nn.Linear(hidden_sz, hidden_sz) ,
            nn.ReLU(),
            nn.Linear(hidden_sz, n_outputs)
        )
     
    def forward(self, input):
        return self.linear_relu_stack(input)  
        
#observation['P1'] = {'oppChar1': 4, 'ownChar1': 8, 'oppChar': 4, 'ownChar': 8, 'oppHealth': 143, 'ownHealth': 187, 'oppSide': 0, 'ownSide': 1, 'oppWins': 1, 'ownWins': 0, 'actions': {'move': 5, 'attack': 7}}
#observation['stage'] = 1
        
class finalNetwork(nn.Module):
    def __init__(self, input_shape, hidden_sz, n_actions=72):
        super(finalNetwork, self).__init__()
        self.dense_network = nn.Sequential(
            nn.Linear(input_shape[0], hidden_sz),
            nn.ReLU(),
            nn.Linear(hidden_sz, n_actions)
        )
            
    def forward(self, input):
        return self.dense_network(input)
        
    def predict(self, input):  
        return self.forward(input).argmax() 
      
def preprocess(observation, past_5_frames):
    """given an observation and past 5 frames give us the preprocessed input"""

    current_frame = observation['frame']
    metaData = observation['P1']
    actions = metaData['actions']
    small = cv2.resize(current_frame, (128, 128)) 
    img_gray = cv2.cvtColor(np.float32(small), cv2.COLOR_RGB2GRAY)[None, :][None, :] # shape 1 by 1 by 256 by 256
    img_gray = torch.tensor(img_gray)
    frames = torch.row_stack((img_gray, past_5_frames))

    meta = np.zeros(20)
    meta[:2] = metaData['ownSide'], metaData['ownHealth'], metaData['oppHealth']
    meta[2 + actions['move']] = 1
    meta[11 + actions['attack']] = 1
    output = {'frames': frames, 'meta': meta}

    return output
  
# DQN Algorithm

REPLAY_MEMORY = []
learning_rate = 0.0001
imagenet = ImageNetwork(64)
timenet = TimeLSTM(64, 16, 32, 1)
metanet = metaData(20, 32, 16)
finalnet = finalNetwork(32, 64, 72)
optimizer = optim.Adam(list(imagenet.parameters()) + list(timenet.parameters()) + list(metanet.parameters()) + list(finalnet.parameters()), learning_rate)
num_sample = 16
discount_rate = 0.9
rolling_frames = []

def SELECT_ACTION(epsilon, input):
    """Selects random action w/ prob epsilon, otherwise picks the action of highest value from the Q-Network"""
    
    action_type = np.random.choice([0, 1], p=[epsilon, 1-epsilon])
    if action_type == 0 or not(input):
        action = env.action_space.sample()
    else:
        frameout1 = imagenet(input['frames'])
        frameout = timenet(frameout1)
        metaout = metanet(input['meta'])
        concat_data = torch.cat((frameout, metaout))
        action = finalnet.predict(concat_data)   #f (observation)   # select best action from output of Q net-work
        action = {'move': action//8, 'attack': action%8}
    return action

def LOSS(y, Q):
    return nn.functional.mse_loss(y, Q)

def TRAIN_STEP():
    inds = np.random.choice(len(REPLAY_MEMORY), num_sample, replace=False)
    replay_vals = [REPLAY_MEMORY[ind] for ind in inds]
    y_js = torch.zeros(num_sample, requires_grad=True)
    Q_js = torch.zeros(num_sample, requires_grad=True)
    optimizer.zero_grad()
    for i in range(len(replay_vals)):
        if (replay_vals[i][4]):
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
        ind = replay_vals[i][1]['move']*8 + replay_vals[i][1]['attack']
        Q_js[i] = finalnet(concat_data)[ind]
    myloss = LOSS(y_js, Q_js)
    myloss.backward()
    optimizer.step()

def RUN_DQN_ALGORITHM(num_episodes, num_time_steps, eps=1, min_eps=0.05):
    """
    Runs through DQN algo. as described in https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

    Parameters
    ----------
    eps (float): probability of selecting a random action. this can decay as the learning process progresses (i.e. do more exploration initially, then less later on) 
    """

    for episode in range(num_episodes):
        observation = env.reset()
        rolling_frames = []
        phi_t = None
        while True:
            if (len(rolling_frames) == 5):
                phi_t = preprocess(observation, rolling_frames)
            eps = max(eps*0.99, min_eps)
            action = SELECT_ACTION(eps, phi_t)
            rolling_frames.append(observation['frame'])
            if (len(rolling_frames) == 6):
                rolling_frames = rolling_frames[1:]
            observation, reward, done, info = env.step(action)
            if (info['roundDone']):
                rolling_frames = []
                phi_t = None
            if (phi_t != None):
                phi_tplus1 = preprocess(observation, rolling_frames)
                newtuple = (phi_t, action, reward, phi_tplus1, info['roundDone'])
                REPLAY_MEMORY.append(newtuple)
                if (len(REPLAY_MEMORY) == 101):
                    REPLAY_MEMORY = REPLAY_MEMORY[1:]
            if (info['gameDone']):
                break
            if (len(REPLAY_MEMORY)>20):
                TRAIN_STEP()
            # store transition in REPLAY_MEMORY
            # sample random minibatch of transitions from REPLAY_MEMORY
            # set target value y_i according to eqn in paper
            # perform grad. desc. on LOSS(y_i, Q_value)
