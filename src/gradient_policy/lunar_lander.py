# credit https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Policy%20Gradients/Cartpole/Cartpole%20REINFORCE%20Monte%20Carlo%20Policy%20Gradients.ipynb
# credit https://github.com/dennybritz/reinforcement-learning/blob/master/PolicyGradient/CliffWalk%20REINFORCE%20with%20Baseline%20Solution.ipynb

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

class PolicyEstimator():
    """Policy Function Approximator"""

    def __init__(self, state_size, action_size):
        with tf.name_scope("inputs"):
            input_ = tf.placeholder(tf.float32, [None, state_size], name="input_")
            actions = tf.placeholder(tf.int32, [None, action_size], name="actions")
            discounted_episode_rewards_ = tf.placeholder(tf.float32, [None,], name="discounted_episode_rewards")
            
            # Add this placeholder for having this variable in tensorboard
            mean_reward_ = tf.placeholder(tf.float32 , name="mean_reward")

            with tf.name_scope("fc1"):
                fc1 = tf.contrib.layers.fully_connected(inputs = input_,
                                                        num_outputs = 10,
                                                        activation_fn=tf.nn.relu,
                                                        weights_initializer=tf.contrib.layers.xavier_initializer())

            with tf.name_scope("fc2"):
                fc2 = tf.contrib.layers.fully_connected(inputs = fc1,
                                                        num_outputs = action_size,
                                                        activation_fn= tf.nn.relu,
                                                        weights_initializer=tf.contrib.layers.xavier_initializer())
            
            with tf.name_scope("fc3"):
                fc3 = tf.contrib.layers.fully_connected(inputs = fc2,
                                                        num_outputs = action_size,
                                                        activation_fn= None,
                                                        weights_initializer=tf.contrib.layers.xavier_initializer())

            with tf.name_scope("softmax"):
                action_distribution = tf.nn.softmax(fc3)

            with tf.name_scope("loss"):
                # tf.nn.softmax_cross_entropy_with_logits computes the cross entropy of the result after applying the softmax function
                # If you have single-class labels, where an object can only belong to one class, you might now consider using 
                # tf.nn.sparse_softmax_cross_entropy_with_logits so that you don't have to convert your labels to a dense one-hot array. 
                neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits = fc3, labels = actions)
                loss = tf.reduce_mean(neg_log_prob * discounted_episode_rewards_) 
                
            
            with tf.name_scope("train"):
                train_opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        # tensorboard
        # Setup TensorBoard Writer
        writer = tf.summary.FileWriter("/tensorboard/pg/1")

        ## Losses
        tf.summary.scalar("Loss", loss)

        ## Reward mean
        tf.summary.scalar("Reward_mean", mean_reward_)

        write_op = tf.summary.merge_all()

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