from mc import FiniteMCModel as MC
import gym
import numpy as np


# use mountaincar env
env = gym.make('MountainCar-v0')

episodes = int(10e4)

# 2d space
# velocity within a state
min_position = -1.2
max_position = 0.6
max_speed = 0.07

# states
# spatial position and possilbe speed
S = [(x, y, z) for x in np.linspace(min_position, max_position, 10) 
                for y in np.linspace(min_position, max_position, 10) 
                for z in np.linspace(0., 0.07, 10)]

# three actions
# left, nothing, right
A = 2

# define MC class
m = MC(S, A, epsilon=1)

# loop through episodes
for i in range(1, episodes+1):
    ep = []
    # reset position of cart for episode
    # https://github.com/openai/gym/blob/master/gym/envs/classic_control/mountain_car.py
    # shows random uniform reset
    observation = env.reset()

    


    



