# credit https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Policy%20Gradients/Cartpole/Cartpole%20REINFORCE%20Monte%20Carlo%20Policy%20Gradients.ipynb

import tensorflow as tf
import numpy as np
import gym

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