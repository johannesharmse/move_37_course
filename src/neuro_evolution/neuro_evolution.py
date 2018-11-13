# adapted from https://github.com/ikergarcia1996/NeuroEvolution-Flappy-Bird/blob/master/Jupyter%20Notebook/Flappy.ipynb

import numpy as np
import cv2
import neat
import gym

n_generations = 1000

class Environment(object):
    def __init__(self, game, width=84, height=84):
        self.game = gym.make(game)
        self.width = width
        self.height = height

    def preprocess(self, screen):
        preprocessed: np.array = cv2.resize(screen, (self.height, self.width))  # 84 * 84 로 변경
        preprocessed = np.dot(preprocessed[..., :3], [0.299, 0.587, 0.114])  # Gray scale 로 변경
        # preprocessed: np.array = preprocessed.transpose((2, 0, 1))  # (C, W, H) 로 변경
        preprocessed: np.array = preprocessed.astype('float32') / 255.

        return preprocessed

    def reset(self):
        """
        :return: observation array
        """
        # print(self.game.reset())
        state = self.game.reset()
        # observation = self.preprocess(observation)
        return state

    def get_screen(self):
        screen = self.game.render('rgb_array')
        screen = self.preprocess(screen)
        return screen

class Agent(object):

    def __init__(self, action_repeat=4, render=False):
        self.env = Environment(game='LunarLander-v2')
        self.action_repeat = action_repeat
        self.reward = 0.

        if render:
            self.env.game.render()

        # self.epsilon = 0.1
        # self.actions = actions
        # self.learning_rate = 0.01
        # self.discount_factor = 0.9
        # self.q_table = defaultdict(lambda: [0., 0., 0., 0.])

    def reward_sum(self, new_reward):
        self.reward += new_reward

    # def get_initial_states(self):
    #     state = self.env.reset()
    #     state = self.env.get_screen()
    #     states = np.stack([state for _ in range(self.action_repeat)], axis=0)

    # def get_action(self, state):
    #     if np.random.rand() < self.epsilon:
    #         action = np.random.choice(self.env.game.action_space)
    #     else:
    #         state_action = self.q_table[state]
    #         a

    # # update q function with sample <s, a, r, s'>
    # def learn(self, state, action, reward, next_state):
    #     currrent_q = self.q_table[state][action]
    #     # using Bellman Optimality Equation
    #     #  to update q function
    #     new_q = reward + self.discount_factor * max(self.q_table[next_state])
    #     self.q_table[state][action] += self.learning_rate * (new_q - currrent_q)

    # get action for the state according to the 
    # q function table
    # agent picks action of epsilon-greedy policy
    # def get_action(self, state):
    #     if np.random.rand() < self.epsilon:
    #         # take random action
    #         action = np.random.choice(self.actions)
    #     else:
    #         # take action according
    #         # to the q function table
    #         state_action = self.q_table[state]
    #         action = self.arg_max(state_action)
        
    #     return action

    # @staticmethod
    # def arg_max(state_action):
    #     max_index_list = []
    #     max_value = state_action[0]
    #     for index, value in enumerate(state_action):
    #         if value > max_value:
    #             max_index_list.clear()
    #             max_value = value
    #             max_index_list.append(index)
    #         elif value == max_value:
    # #             max_index_list.append(index)
        
    #     # random choice of actions that all 
    #     # give max q
    #     return random.choice(max_index_list)

def eval_genomes(genomes, config):
    
    ship = Agent(render=False)

    for genome_id, genome in genomes:
        genome.fitness = 99999
        input_len = 1
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        reward = 0.
        state = list()

        
        for i in range(input_len):
            if i == 0:
                # print(ship.env.reset())
                next_state = ship.env.reset()
            else:
                next_state, next_reward, done, _ = ship.env.game.step(0)
                reward += next_reward

            state.append(next_state)

        state = state[-1]
        ship.reward_sum(reward)
        

        while True:
            nnInput = state
            # print(nnInput)
            output = net.activate(nnInput)
            action = np.argmax(output)
            
            reward = 0.
            state = list()

            for _ in range(input_len):
                next_state, next_reward, done, info = ship.env.game.step(action)
                state.append(next_state)
                reward += next_reward

            state = state[-1]
            ship.reward_sum(reward)

            if done:
                ship.env.game.close()
                break


        genome.fitness = ship.reward


if __name__ == "__main__":
    
    config = neat.Config(neat.DefaultGenome, 
    neat.DefaultReproduction, 
    neat.DefaultSpeciesSet, 
    neat.DefaultStagnation, 
    'LunarLandingNEAT')

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(False))

    winner = p.run(eval_genomes, n_generations)

    print(winner)