# RL Agent Plays Dead or Alive ++

https://user-images.githubusercontent.com/43799733/163399157-c6364d4d-f1b7-4dd3-b248-7752a08c8969.mp4

ECE324 Course Project <br>
Code for the Deep Q Learning Model can be found in dqn_agent.py. 

## Setup
```
git clone https://github.com/diambra/diambraArena.git
cd diambraArena
./setupOS.sh
pip3 install .
```
After cloning the diambraArena repository, you'll need to download the ROM for DEAD OR ALIVE ++ [JAPAN].
Afterwards, you should be able to run the files in /models.

## Explanation of file structure
### .circleci
Contains yaml file needed for integration with CircleCI. We haven't done much work with this as of yet, so this folder isn't really important at the moment.

### models
Contains files for agents that we run in the diambra environment. The two most important files in this directory are dqn_agent.py and meta_ablation.py. <br><br>
**models/dqn_agent.py**: Runs the Deep Q Learning (DQL) algorithm with our original model (metanet + CNN + LSTM) for the Deep Q Network (DQN), and writes experiment results (i.e. list of cumulative reward earned every round) to a file in the current working directory. Running this requires that you have modules installed for libraries we use (i.e. torch, matplotlib, etc.). Also, you need to modify the line of code shown below with your devices' path to the downloaded ROM file for DOA.

```
    settings["romsPath"] = "/home/nabil/Downloads/"
```

**models/meta_ablation.py**: Runs DQL with out metanet model (a subset of the original model) and writes experimental results (same as for dqn_agent.py) to a file in the current working directory. The requirements for running this are the same as for dqn_agent.py

**models/test_agent.py**: Runs an agent that randomly samples actions from the diambra environment action space at every time step. This file also prints out information about environment state at every timestep. To run this, you need to make the same change to a line of code as shown above for the dqn_agent.py file.
