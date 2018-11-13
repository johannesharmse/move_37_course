# credit https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Policy%20Gradients/Cartpole/Cartpole%20REINFORCE%20Monte%20Carlo%20Policy%20Gradients.ipynb

import tensorflow as tf
import numpy as np
import gym

def discount_and_normalize_rewards(episode_rewards):
    """Calculate normalized episode reward"""

    discounted_episode_rewards = np.zeros_like(episode_rewards)
    cumulative = 0.0
    for i in reversed(range(len(episode_rewards))):
        cumulative = cumulative * gamma + episode_rewards[i]
        discounted_episode_rewards[i] = cumulative
    
    mean = np.mean(discounted_episode_rewards)
    std = np.std(discounted_episode_rewards)
    discounted_episode_rewards = (discounted_episode_rewards - mean) / (std)
    
    return discounted_episode_rewards

if __name__ == "__main__":
    env = gym.make('LunarLander-v2')

    # n frames
    state_size = 4
    # possible actions
    action_size = env.action_space.n
    
    # max episodes for training
    max_episodes = 500
    learning_rate = 0.01
    # discount rate
    gamma = 0.95




    # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))