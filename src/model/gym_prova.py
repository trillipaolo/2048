import gym
from src.model.env2048 import Env2048

env = Env2048()

for i_episode in range(3):
    observation = env.reset()
    for t in range(3):
        env.render()

        action = 3
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
