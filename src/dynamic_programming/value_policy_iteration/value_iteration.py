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
        start, reward, done, _ = env.step(int(policy[start]))

        # collect reward (with gamma penalties for moves)
        totalReward += (gamma ** stepIndex * reward)

        stepIndex += 1

        if done:
            break

    return totalReward


# Evaluation
def evaluatePolicy(env, policy, gamma=1.0, n=100):
    """Calculate mean score of policy"""
    
    scores = [execute(env, policy, gamma) for _ in range(n)]

    return np.mean(scores)


def calculatePolicy(v, gamma=1.0):
    """Calculate new policy"""
    # init random policy (zero values for all states)
    policy = np.zeros(env.env.nS)

    for s in range(env.env.nS):
        # random expected total rewards 
        # for given state and possible actions
        q_sa = np.zeros(env.action_space.n)

        # consider all actions
        for a in range(env.action_space.n):
            # consider all next states for an action
            for next_sr in env.env.P[s][a]:
                # probability, next state, reward, done
                p, s_, r, _ = next_sr
                # get probability weighted estimated reward
                q_sa[a] += (p * (r + gamma * v[s_]))
        
        # choose best found action for the state (policy)
        policy[s] = np.argmax(q_sa)

    return policy


def valueIteration(env, gamma=1.0):
    
    # init value-function
    value = np.zeros(env.env.nS)
    # stop if not converging
    max_iterations = 10000
    # converge value
    eps = 1e-20

    # iterate
    for i in range(max_iterations):
        # previous value function
        prev_v = np.copy(value)
        # observe all states
        for s in range(env.env.nS):
            # get estimated rewards for each
            # action for the current state 
            q_sa = [sum([p * (r + prev_v[s_]) 
            for p, s_, r, _ in env.env.P[s][a]]) 
            for a in range(env.env.nA)]
            
            # get highest estimated possible reward for 
            # current state
            value[s] = max(q_sa)

        # stop if converged
        if np.sum(np.fabs(prev_v - value)) <= eps:
            print('Value-iteration converged at # %d.' % (i + 1))
            break

    return value


if __name__ == '__main__':
    
    # discount factor
    gamma = 1.0

    # create env
    env = gym.make('FrozenLake-v0')

    # do value iteration and get optimal values
    optimalValue = valueIteration(env, gamma)

    # time complexity
    startTime = time.time()

    # choose actions (policy) that will achieve
    # our optimal values
    policy = calculatePolicy(optimalValue, gamma)

    # simulate and score
    policy_score = evaluatePolicy(env, policy, gamma, n=1000)

    # time complexity
    endTime = time.time()

    print("Best policy score = %.2f." % policy_score)
    print("Time Duration: %.2f seconds" % (endTime - startTime))