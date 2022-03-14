import diambraArena
import argparse
from diambraArena.gymUtils import showGymObs
from stable_baselines3 import DQN

settings = {}

settings["romsPath"] = "/home/nabil/Downloads/"
settings["gameId"] = "doapp"
settings["actionSpace"] = "discrete"
settings["attackButCombination"] = False # reduce action space size

envId = "TestEnv"
env = diambraArena.make(envId, settings)
observation = env.reset()
showGymObs(observation, env.charNames)

'''
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000, log_interval=4)
model.save("deepq_doapp")
del model
model = DQN.load("deepq_doapp")
'''

while True:

    actions = env.action_space.sample()
    #action, _states = model.predict(observation["frame"], deterministic=True)

    observation, reward, done, info = env.step(actions)
    showGymObs(observation, env.charNames)

    print("Reward: {}".format(reward))
    print("Done: {}".format(done))
    print("Info: {}".format(info))

    if done:
        observation = env.reset()
        showGymObs(observation, env.charNames)
        break

env.close()
