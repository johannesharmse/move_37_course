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

    # intit reward and start 
    # observation (assuming this is slightly random)
    totalReward = 0
    start = env.reset()

    # iterate through episode
    for t in range(episodeLength):
        
        if render:
            env.render()

        # get action for current state
        action = policy[start]
        # perform state action and see what happens
        start, reward, done, _ = env.step(action)
        # collect reward
        totalReward += reward

        if done:
            break

    return totalReward


# Evaluation
def evaluatePolicy(env, policy, n_episodes=100):
    """Calculate mean score of policy"""
    # init
    totalReward = 0.0

    # run for a number of episodes
    for _ in range(n_episodes):
        # add 
        totalReward += execute(env, policy)
    
    # average reward of input policy
    return totalReward / n_episodes


def gen_random_policy():
    """Generate random policy"""
    return np.random.choice(4, size=16)

if __name__ == "__main__":
    env = gym.make('FrozenLake-v0')
    # number of random policies to consider
    n_policies = 1000
    
    # time complexity
    startTime = time.time()
    
    # generate random policies
    policy_set = [gen_random_policy() for _ in range(n_policies)]

    # score each random policy
    policy_score = [evaluatePolicy(env, p) for p in policy_set]

    # time complexity
    endTime = time.time()

    print("Best policy score = %.2f." % np.max(policy_score))
    print("Time Duration: %.2f seconds" % (endTime - startTime))
