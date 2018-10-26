from mc import FiniteMCModel as MC
import gym
import numpy as np
import random

# print(np.array([np.random.uniform(-0.6, -0.4), 0]))

# use mountaincar env
env = gym.make('Taxi-v2')

episodes = int(10e6)

# states
# S = [(row, col, passidx, destidx) for row in range(5) 
#                                     for col in range(5)
#                                     for passidx in range(5) 
#                                     for destidx in range(4)]
S = 500

# three actions
# left, nothing, right
A = 5

# define MC class
m = MC(S, A, epsilon=1)

# loop through episodes
for i in range(1, episodes+1):
    ep = []
    # reset state
    observation = env.reset()

    while True:
        # choose policy
        # based on random action selected by b methid 
        action = m.choose_action(m.b, observation)

        # simulate
        next_observation, reward, done, _ = env.step(action)
        ep.append((observation, action, reward))
        observation = next_observation
        if done:
            break

    m.update_Q(ep)
    m.epsilon = max((episodes-i)/eps, 0.1)

    print("Final expected returns: {}".format(m.score(env, m.pi, n_samples=10000)))





    



