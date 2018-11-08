# adapted from https://github.com/ikergarcia1996/NeuroEvolution-Flappy-Bird/blob/master/Jupyter%20Notebook/Flappy.ipynb

import neat
import gym

n_generations = 1000
env = 

class Environment(object):
    def __init__(self, game, width=84, height=84):
        self.game = gym.make(game)
        self.width = width
        self.height = height

class Agent(object):

    def __init__(self):
        self.env = Environment(game='LunarLander-v2')

    def get_initial_states(self):
        state = self.env.reset()
        state = self.env.get_screen()
        states = np.stack([state for _ in range(self.action_repeat)], axis=0)

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 99999
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        while True:
            states = self.get_initial_states()



config = neat.Config(neat.DefaultGenome, 
neat.DefaultReproduction, 
neat.DefaultSpeciesSet, 
neat.DefaultStagnation, 
'LunarLandingNEAT')

p = neat.Population(config)

p.add_reporter(neat.StdOutReporter(False))

winner = p.run(eval_genomes, n_generations)