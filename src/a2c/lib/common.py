# credit https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/8_Actor_Critic_Advantage/AC_CartPole.py

"""
Actor-Critic using TD-error as the Advantage, Reinforcement Learning.
The cart pole example. Policy is oscillated.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
Using:
tensorflow 1.0
gym 0.8.0
"""

import numpy as np
import tensorflow as tf
import gym

np.random.seed(2)
tf.set_random_seed(2)  # reproducible

# Superparameters
OUTPUT_GRAPH = False
MAX_EPISODE = 3000
DISPLAY_REWARD_THRESHOLD = 200  # renders environment if total episode reward is greater then this threshold
MAX_EP_STEPS = 1000   # maximum time step in one episode
RENDER = False  # rendering wastes time
GAMMA = 0.9     # reward discount in TD error
LR_A = 0.001    # learning rate for actor
LR_C = 0.01     # learning rate for critic

env = gym.make('CartPole-v0')
env.seed(1)  # reproducible
env = env.unwrapped

N_F = env.observation_space.shape[0]
N_A = env.action_space.n


class Actor(object):
    """
    Actor. This component is responsible for the 
    action the agent should take next.
    """
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.sess = sess # actor and critic use the same session, but trained seperately

        self.s = tf.placeholder(tf.float32, [1, n_features], "state") # state. can be pixels or whatevs defines the state
        self.a = tf.placeholder(tf.int32, None, "act") # action to take
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error. same as advantage. Positive is good, negative is bad.

        # actor branch of neural net
        # 2 dense layers
        with tf.variable_scope('Actor'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=20,    # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.acts_prob = tf.layers.dense(
                inputs=l1,
                units=n_actions,    # output units
                activation=tf.nn.softmax,   # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='acts_prob'
            )
        
        # expected value based on the action to take
        # we want to maximise the expexted value
        # it has two components - log of the recommended action probability (always negative)
        # and the advantage (postive or negative)
        # with the actor, we essentially just want to correct the action probabilities. not the advantage (critic)
        # thus increase the probabilities that give us large advantage, decrease probabilities of 
        # actions that have large negative advantage.
        # 
        # more extreme action probabilities should correlate with lower td_error (expected value).
        # this means if we have an extreme probability and an extreme td_error
        # we have a lot to learn... however, if we have extreme probabilities and low td_errror, 
        # it's chilled because that's what we expected and thus can't change much (already extreme). 
        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.acts_prob[0, self.a])
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

    def learn(self, s, a, td):
        """
        Update weights based on loss function.
        Return advantage guided (dependent on log prob of action) loss
        if interested
        """
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        """"
        Choose action based on assigned 
        action probabilities
        """
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})   # get probabilities for all actions
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())   # return a int


class Critic(object):
    """
    Critic. Responsible for correctly 
    predicting the value of a state.
    """
    def __init__(self, sess, n_features, lr=0.01):
        self.sess = sess # actor and critic use the same session, but trained seperately

        self.s = tf.placeholder(tf.float32, [1, n_features], "state") # state. can be pixels or whatevs defines the state
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next") #value of next state (used in Bellman equation)
        self.r = tf.placeholder(tf.float32, None, 'r') # reward of current state (used in Bellman equation)

        # critic branch of neural net
        # 1 dense layer
        # linear output
        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=20,  # number of hidden units
                activation=tf.nn.relu,  # None
                # have to be linear to make sure the convergence of actor.
                # But linear approximator seems hardly learns the correct Q.
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.v = tf.layers.dense(
                inputs=l1,
                units=1,  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V'
            )

        # we want to get TD error as close to zero as possible
        # large positive TD error means that it went unexpectedly well
        # large negative TD error means it went unexpectedly bad
        # by getting it as close to zero as possible, we increase 
        # our knowledge around the value of a state
        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + GAMMA * self.v_ - self.v
            self.loss = tf.square(self.td_error)    # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):
        """
        Update weights based on loss function
        Return TD Error of interested
        """
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                          {self.s: s, self.v_: v_, self.r: r})
        return td_error


sess = tf.Session()

actor = Actor(sess, n_features=N_F, n_actions=N_A, lr=LR_A)
critic = Critic(sess, n_features=N_F, lr=LR_C)     # we need a good teacher, so the teacher should learn faster than the actor

# init variablles
sess.run(tf.global_variables_initializer())

# define log path
if OUTPUT_GRAPH:
    tf.summary.FileWriter("logs/", sess.graph)

# loop through episodes
for i_episode in range(MAX_EPISODE):
    # init env and start state
    s = env.reset()
    # step count
    t = 0
    # episode rewards tracker
    track_r = []

    while True:

        # render
        if RENDER: env.render()
        
        # choose action
        a = actor.choose_action(s)

        # perform action. get next state and reward
        s_, r, done, info = env.step(a)

        # dying is bad
        if done: r = -20
        
        # keep track of rewards
        track_r.append(r)

        # train - update critic and actor weights
        td_error = critic.learn(s, r, s_)  # gradient = grad[r + gamma * V(s_) - V(s)]
        actor.learn(s, a, td_error)     # true_gradient = grad[logPi(s,a) * td_error]

        # init next state
        s = s_

        # step count
        t += 1

        # if episode is done or we won
        if done or t >= MAX_EP_STEPS:

            # calculate total episode rewards
            ep_rs_sum = sum(track_r)

            # calculated weighted average episode reward
            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
            
            # worth showing?
            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True  # rendering
            
            print("episode:", i_episode, "  reward:", int(running_reward))
            
            break