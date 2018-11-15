# credit https://github.com/colinskow/move37/blob/master/actor_critic/lib/common.py

import os
import sys
import time
import numpy as np
import ptan

import tensorflow as tf
# import torch
# import torch.nn as nn

def conv_fn(input_shape):

    input_layer = tf.placeholder(tf.float32, input_shape, 32)
    conv1 = tf.layers.conv2d(
        inputs=input_layer, 
        filters=32, 
        kernel_size=[8,8], 
        strides=4, 
        activation=tf.nn.relu
    )
    conv2 = tf.layers.conv2d(
        inputs=conv1, 
        filters=64, 
        kernel_size=[4,4], 
        strides=2, 
        activation=tf.nn.relu
    )

def policy_fn(input_shape):

    input_layer = tf.placeholder(tf.float32, input_shape, 32)
    fn1 = tf.layers.fully_connected(
        inputs=input_layer, 
        num_outputs= 512,
        activation_fn=tf.nn.relu
    )
    fn2 = tf.layers.fully_connected(
        inputs=fn1, 
        num_outputs= 1,
        activation=None
    )

def value_fn(input_shape):

    input_layer = tf.placeholder(tf.float32, input_shape, 32)
    fn1 = tf.layers.fully_connected(
        inputs=input_layer, 
        num_outputs= 512,
        activation_fn=tf.nn.relu
    )
    fn2 = tf.layers.fully_connected(
        inputs=fn1, 
        num_outputs= 1,
        activation=None
    )

class AtariA2C(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(AtariA2C, self).__init__()
        
        
            
        # self.conv = nn.Sequential(
        #     nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=4, stride=2),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=1),
        #     nn.ReLU()
        # )

        

        conv_out_size = self._get_conv_out(input_shape)
        self.policy = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

        self.value = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.policy(conv_out), self.value(conv_out)


def unpack_batch(batch, net, last_val_gamma, device='cpu'):
    """
    Convert batch into training tensors
    :param batch:
    :param net:
    :return: states variable, actions tensor, reference q values variable
    """
    states = []
    actions = []
    rewards = []
    not_done_idx = []
    last_states = []
    for idx, exp in enumerate(batch):
        # unpack states, actions, and rewards into separate lists
        states.append(np.array(exp.state, copy=False))
        actions.append(int(exp.action))
        rewards.append(exp.reward)
        if exp.last_state is not None:
            # if the episode has not yet ended, save the index and state prime of the transition
            not_done_idx.append(idx)
            last_states.append(np.array(exp.last_state, copy=False))
    states_v = torch.FloatTensor(states).to(device)
    actions_t = torch.LongTensor(actions).to(device)

    # handle rewards
    rewards_np = np.array(rewards, dtype=np.float32)
    # if at least one transition was non-terminal
    if not_done_idx:
        last_states_v = torch.FloatTensor(last_states).to(device)
        # calculate the values of all the state primes from the net
        last_vals_v = net(last_states_v)[1]
        last_vals_np = last_vals_v.data.cpu().numpy()[:, 0]
        # apply the Bellman equation adding GAMMA * V(s') to the reward for all non-terminal states
        # terminal states will contain just the reward received
        rewards_np[not_done_idx] += last_val_gamma * last_vals_np

    # these are the Q(s,a) values we will use to calculate the advantage and value loss
    q_vals_v = torch.FloatTensor(rewards_np).to(device)
    return states_v, actions_t, q_vals_v


def unpack_batch_continuous(batch, net, last_val_gamma, device="cpu"):
    """
    Convert batch into training tensors
    :param batch:
    :param net:
    :return: states variable, actions tensor, reference values variable
    """
    states = []
    actions = []
    rewards = []
    not_done_idx = []
    last_states = []
    for idx, exp in enumerate(batch):
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        if exp.last_state is not None:
            not_done_idx.append(idx)
            last_states.append(exp.last_state)
    states_v = ptan.agent.float32_preprocessor(states).to(device)
    actions_v = torch.FloatTensor(actions).to(device)

    # handle rewards
    rewards_np = np.array(rewards, dtype=np.float32)
    if not_done_idx:
        last_states_v = ptan.agent.float32_preprocessor(last_states).to(device)
        last_vals_v = net(last_states_v)[2]
        last_vals_np = last_vals_v.data.cpu().numpy()[:, 0]
        rewards_np[not_done_idx] += last_val_gamma * last_vals_np

    ref_vals_v = torch.FloatTensor(rewards_np).to(device)
    return states_v, actions_v, ref_vals_v


class RewardTracker:
    def __init__(self, writer, stop_reward):
        self.writer = writer
        self.stop_reward = stop_reward

    def __enter__(self):
        self.ts = time.time()
        self.ts_frame = 0
        self.total_rewards = []
        self.best_mean_reward = None
        return self

    def __exit__(self, *args):
        self.writer.close()

    def reward(self, reward, frame, epsilon=None):
        self.total_rewards.append(reward)
        speed = (frame - self.ts_frame) / (time.time() - self.ts)
        self.ts_frame = frame
        self.ts = time.time()
        mean_reward = np.mean(self.total_rewards[-100:])
        epsilon_str = "" if epsilon is None else ", eps %.2f" % epsilon
        print("%d: done %d games, mean reward %.3f, speed %.2f f/s%s" % (
            frame, len(self.total_rewards), mean_reward, speed, epsilon_str
        ))
        sys.stdout.flush()
        if epsilon is not None:
            self.writer.add_scalar("epsilon", epsilon, frame)
        self.writer.add_scalar("speed", speed, frame)
        self.writer.add_scalar("reward_100", mean_reward, frame)
        self.writer.add_scalar("reward", reward, frame)
        save_checkpoint = False
        if self.best_mean_reward is None or self.best_mean_reward < mean_reward:
            if self.best_mean_reward is not None:
                print("Best mean reward updated %.3f -> %.3f, model saved" % (self.best_mean_reward, mean_reward))
            save_checkpoint = True
            self.best_mean_reward = mean_reward
        if mean_reward > self.stop_reward:
            print("Solved in %d frames!" % frame)
            return True, save_checkpoint
        return False, save_checkpoint


def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path