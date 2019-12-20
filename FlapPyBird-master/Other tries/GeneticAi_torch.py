import sys
import numpy as np

import torch
import multiprocessing as mp

import torch.nn as nn
import torch.nn.functional as F
from Game import flappy_AI_Supervises

from torch.autograd import Variable
cuda = torch.cuda.is_available()
torch.set_grad_enabled(False)


class Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 2)
        self.fc2 = nn.Linear(2, 2)
        self.fc3 = nn.Linear(2, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.sigmoid(x)
        # -----
        #x = self.fc2(x)
        #x = F.sigmoid(x)
        #-------
        x = self.fc3(x)
        x = F.sigmoid(x)
        return x


# -- Return chromosomes of 2 best players
# -- List of torch models, we select n-best of them
def select(chromosomes):
    return chromosomes[0], chromosomes[1]


# -- Gets 2 childs by mixing parameters from each parent
def crossover(parent1, parent2):
    child1 = Net(2,1)
    child2 = Net(2,1)

    weights1 = parent1.fc1.weight.data
    weights2 = parent2.fc1.weight.data
    cross_point = np.random.randint(1, len(weights1 - 1))
    child1.fc1.weight.data = torch.cat((weights1[0:cross_point], weights2[cross_point:]),0)
    child2.fc1.weight.data = torch.cat((weights2[0:cross_point], weights1[cross_point:]),0)

    weights1 = parent1.fc2.weight.data
    weights2 = parent2.fc2.weight.data
    cross_point = np.random.randint(1, len(weights1[0] - 1))
    child1.fc2.weight.data = torch.reshape(torch.cat((weights1[0][0:cross_point], weights2[0][cross_point:]), 0),(1,2))
    child2.fc2.weight.data = torch.reshape(torch.cat((weights2[0][0:cross_point], weights1[0][cross_point:]), 0),(1,2))



    return child1, child2


def mutate1(child1, child2):
    model = child1
    for name, param in model.named_parameters():
        print(name, param.size())

    for layer in model._modules.keys():
        dimension = model._modules[layer].weight.data.shape()


        for neuron in model._modules[layer].weight.data:
            print()
            return




def mutate(child1, child2):

    fc1_weights = []
    w = []
    for param in child1.fc1.weight.data:
        for w_params in param:
            prob = np.random.rand()
            if prob > 0.8:
                w_params += (np.random.randn() - 0.5)/2
            w.append(w_params)

        fc1_weights.append(w)
        w = []

    child1.fc1.weight.data = torch.FloatTensor(fc1_weights)

    fc2_weights = []
    w = []
    for param in child1.fc2.weight.data:
        for w_params in param:
            prob = np.random.rand()
            if prob > 0.8:
                w_params += (np.random.randn() - 0.5)/2
            w.append(w_params)

        fc2_weights.append(w)
        w = []

    child1.fc2.weight.data = torch.FloatTensor(fc2_weights)

    ######################

    fc1_weights = []
    w = []
    for param in child2.fc1.weight.data:
        for w_params in param:
            prob = np.random.rand()
            if prob > 0.8:
                w_params += (np.random.randn() - 0.5)/2
            w.append(w_params)

        fc1_weights.append(w)
        w = []

    child2.fc1.weight.data = torch.FloatTensor(fc1_weights)

    fc2_weights = []
    w = []
    for param in child2.fc2.weight.data:
        for w_params in param:
            prob = np.random.rand()
            if prob > 0.8:
                w_params += (np.random.randn() - 0.5)/2
            w.append(w_params)

        fc2_weights.append(w)
        w = []

    child2.fc2.weight.data = torch.FloatTensor(fc2_weights)


    return child1, child2

def evolution(chromosomes, child1, child2):
    chromosomes[-1] = child1
    chromosomes[-2] = child2
    return chromosomes

def random_evolution_option2(chromosomes,prob_random):
    for i in range(2,len(chromosomes) - 2):
        prob = np.random.rand()
        if prob > (1-prob_random):
          chromosomes[i] = Net(2,1)
    return chromosomes


def random_evolution_option1(N_POPULATION,new_population,prob_random):
    n_random = int(prob_random*N_POPULATION)

    for i in range(n_random):
        new_population.append(Net(2,1))
    return new_population


def elitism(chromosomes,percent):
    new_population = []
    n_intact = int(percent*len(chromosomes))
    for i in range(n_intact):
        new_population.append(chromosomes[i])

    return new_population

def child_generator(parent1, parent2, new_population, n_childs):

    prob_mutate = 0.1

    for i in range(n_childs):
      child1 = Net(2,1)

      # WE UPDATE FIRST LAYER
      weights1 = parent1.fc1.weight.data
      weights2 = parent2.fc1.weight.data

      row,col = weights1.shape

      for j in range(row):
          for k in range(col):
            if np.random.rand() < 0.5:
                weights1[j][k] = weights2[j,k]
            if np.random.rand() <  0.1: #RANDOM MUTATION
                weights1[j][k] += np.random.randn() * 0.5 * 2 - 0.5

      child1.fc1.weight.data = weights1

      # WE UPDATE SECOND LAYER
      weights1 = parent1.fc2.weight.data
      weights2 = parent2.fc2.weight.data

      row,col = weights1.shape

      for j in range(row):
          for k in range(col):
            if np.random.rand() < 0.5:
                weights1[j][k] = weights2[j,k]
            if np.random.rand() <  0.1: #RANDOM MUTATION
                weights1[j][k] += np.random.randn() * 0.5 * 2 - 0.5


      child1.fc2.weight.data = weights1

      new_population.append(child1)

    return new_population


def train(chromosomes): #chromosomes are the models sorted by points

    #option 1
    ##################################################################

    #N_POPULAION = len(chromosomes)
    #new_population = elitism(chromosomes,0.2)
    #new_population = random_evolution_option1(N_POPULAION,new_population,0.1)

    #n_childs = N_POPULAION - len(new_population)

    #parent1, parent2 = select(chromosomes)
    #new_population = child_generator(parent1, parent2, new_population, n_childs)


    ####################################################################
    # option2


    parent1, parent2 = select(chromosomes)
    child1, child2 = crossover(parent1, parent2)
    child1, child2 = mutate(child1, child2)
    new_population = evolution(chromosomes, child1, child2)
    new_population = random_evolution_option2(new_population,0.2)

    return new_population


def initialize(n_population, n_param):
    chromosomes_arr = []
    for i in range(n_population):
        model = Net(n_param, 1)
        chromosomes_arr.append(model)

    return chromosomes_arr


def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def predict(chromosomes, parameters):


    y = chromosomes.forward(torch.from_numpy(np.array(parameters)).float())


    if y > 0.5:
        return True
    else:
        return False


class Player():
    def __init__(self, model, reward, state, done):
        self.state = state
        self.model = model
        self.reward = reward
        self.isdone = done



def main():
    #PARAMETERS
    n_players = 50
    n_generations = 10000

    ######################

    input_queue = mp.Queue()
    output_queue = mp.Queue()

    p = mp.Process(target=flappy_AI_Supervises.main, args=(input_queue, output_queue))
    p.start()


    crash_list = [False]* n_players

    new_population = []

    for i in range(n_players):
        new_population.append(Net(2,1))

    for generation  in range(n_generations):

        players = []

        for i in range(n_players):
            players.append(Player(new_population[i], 0, None, False))

        input_queue.put([n_generations, n_generations, n_players])  # This starts next episode
        #input_queue.put([True, 0])  # This starts next episode
        #output_queue.get()  # Gets blank response to confirm episode started

        for id_player in range(n_players):
            input_queue.put([False,id_player])  # Input initial action as no flap

        for n in range(n_players):
            state, reward, done, id_player = output_queue.get()  # Get initial state
            players[id_player].state = state
            players[id_player].reward = reward
            crash_list[id_player] = done

        alive_players = n_players

        while any(x == False for x in crash_list):
            for id_player,player in enumerate(players):
                if not crash_list[id_player]:
                    choice = predict(player.model,player.state)

                    #choice = np.random.choice([0,1],p=(0.9,0.1))

                    input_queue.put([choice, id_player])

            #input()
            for id_player in range(alive_players):
                state, reward, done, id_player1 = output_queue.get()
                players[id_player].state = state
                players[id_player].reward = reward

                if done:
                    crash_list[id_player1] = True
                    alive_players -= 1

        players.sort(key=lambda x: x.reward, reverse=True)
        chromosomes_2_train = []

        for player in players:
            print(player.reward)
            chromosomes_2_train.append(player.model)

        #input()
        new_population = train(chromosomes_2_train)


if __name__ == "__main__":
    # -- Matrix with the chromosomes of each player sorted
    main()
    print()
