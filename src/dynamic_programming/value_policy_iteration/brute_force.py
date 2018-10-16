import numpy as np
import time
import gym


# Execution
def execute(env, policy, episodeLength=100, render=False):
    """
    Args:
        policy: [S,A] shaped matrix representing the policy
        env: OpenAI gym env
        render: boolean to turn rendering on/off
    """

    # intit reward and start env
    totalReward = 0
    start = env.reset()

    # iterate through episode
    for t in range(episodeLength):
        if render:
            env.render()

    return totalReward


def gen_random_policy():
    """Generate random policy"""
    return np.random.choice(4, size=16)

if __name__ == "main":
    env = gym.make('FrozenLake-v0')
    policy = gen_random_policy()
    execute(env, policy)
