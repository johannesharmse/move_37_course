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


def extractPolicy(v, gamma=1.0):
    """Calculate policy given value-function"""
    # init random policy (zero values for all states)
    policy = np.zeros(env.env.nS)

    for s in range(env.env.nS):
        # random expected total rewards 
        # for given state and possible actions
        q_sa = np.zeros(env.env.nA)

        # consider all actions
        for a in range(env.env.nA):
            # calculate expected reward for a 
            # state-action pair
            q_sa[a] = sum([p * (r + gamma * v[s_]) 
            for p, s_, r, _ in env.env.P[s][a]])
        
        # choose best found action for the state (policy)
        policy[s] = np.argmax(q_sa)

    return policy


def CalcPolicyValue(env, policy, gamma=1.0):
    
    # init value-function
    value = np.zeros(env.env.nS)
    # converge value
    eps = 1e-10

    while True:
        # previous value function
        previousValue = np.copy(value)

        # iterate through states in env
        for states in range(env.env.nS):
            # all possible actions for state
            policy_a = policy[states]

            # calculated expected reward for state
            value[states] = sum([p * (r + gamma * previousValue[s_]) 
            for p, s_, r, _ in env.env.P[states][policy_a]])

        # break if converged
        if np.sum(np.fabs(previousValue - value)) <= eps:
            break

    return value

def policyIteration(env, gamma=1.0):
    """Policy Iteration Algo"""
    # init with random policy
    policy = np.random.choice(env.env.nA, size=(env.env.nS))

    # if not converge
    maxIterations = 1000

    # discount factor
    gamma = 1.0

    for i in range(maxIterations):
        # get policy value
        oldPolicyValue = CalcPolicyValue(env, policy, gamma)
        # get new policy (actions)
        newPolicy = extractPolicy(oldPolicyValue, gamma)

        # check if optimal policy
        if np.all(policy == newPolicy):
            print('Policy Iteration converged at %d.' %(i + 1))
            break
        # update policy for next policy value calculation
        policy = newPolicy

    return policy


if __name__ == '__main__':
    
    # create env
    env = gym.make('FrozenLake-v0')

    # time complexity
    startTime = time.time()

    # get optimal policy
    optimalPolicy = policyIteration(env, gamma=1.0)

    # evaluate and score
    score = evaluatePolicy(env, optimalPolicy, gamma=1.0, n=100)

    # time complexity
    endTime = time.time()

    print("Best policy score = %.2f." % score)
    print("Time Duration: %.2f seconds" % (endTime - startTime))
