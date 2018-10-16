import numpy as np
import gym
import time

# Execution
def execute(env, policy, gamma=1.0, render=False):
    """
    Args:
        policy: [S,A] shaped matrix representing the policy
        env: OpenAI gym env
            env.P represents the transition probabilities 
            of the environment
            env.P[s][a] is a list of transition 
            tuples (prob, next_state, reward, done)
            env.nS is a number of states in the environment
            env.nA is the number of actions in the environment
        gamma: Gamma discount factor
        render: boolean to turn rendering on/off
    """

    # intit reward, stepIndex (current iteraton) and start 
    # observation (assuming this is slightly random)
    start = env.reset()
    totalReward = 0
    stepIndex = 0

    # iterate
    while True:

        if render:
            env.render()

        # use latest best action for current state
        start, reward, done, _ = env.step(int(policy(start)))

        # collect reward (with gamma penalties for moves)
        totalReward += (gamma ** stepIndex * reward)

        stepIndex += 1

        if done:
            break

    return totalReward