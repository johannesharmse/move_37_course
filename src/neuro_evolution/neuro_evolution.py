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
        observation = self.game.reset()
        observation = self.preprocess(observation)
        return observation

    def get_screen(self):
        screen = self.game.render('rgb_array')
        screen = self.preprocess(screen)
        return screen

class Agent(object):

    def __init__(self, action_repeat=4):
        self.env = Environment(game='LunarLander-v2')
        self.action_repeat = action_repeat

    def get_initial_states(self):
        state = self.env.reset()
        state = self.env.get_screen()
        states = np.stack([state for _ in range(self.action_repeat)], axis=0)

def eval_genomes(genomes, config):

    ship = Agent()

    for genome_id, genome in genomes:
        genome.fitness = 99999
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        while True:
            states = ship.get_initial_states()


if __name__ == "__main__":
    
    config = neat.Config(neat.DefaultGenome, 
    neat.DefaultReproduction, 
    neat.DefaultSpeciesSet, 
    neat.DefaultStagnation, 
    'LunarLandingNEAT')

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(False))

    winner = p.run(eval_genomes, n_generations)