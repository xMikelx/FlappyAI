import sys
import numpy as np

import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
cuda = torch.cuda.is_available()
torch.set_grad_enabled(False)


class Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 2)
        self.fc2 = nn.Linear(2, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.sigmoid(x)
        # -----
        x = self.fc2(x)
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


def mutate(child1, child2):

    fc1_weights = []
    w = []
    for param in child1.fc1.weight.data:
        for w_params in param:
            prob = np.random.rand()
            if prob > 0.8:
                w_params += np.random.randn() * 0.5 * 2 - 0.5
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
                w_params += np.random.randn() * 0.5 * 2 - 0.5
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
                w_params += np.random.randn() * 0.5 * 2 - 0.5
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
                w_params += np.random.randn() * 0.5 * 2 - 0.5
            w.append(w_params)

        fc2_weights.append(w)
        w = []

    child2.fc2.weight.data = torch.FloatTensor(fc2_weights)


    return child1, child2

def evolution(chromosomes, child1, child2):
    chromosomes[-1] = child1
    chromosomes[-2] = child2
    return chromosomes


def random_evolution(N_POPULATION,new_population,prob_random):
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
    N_POPULAION = len(chromosomes)
    new_population = elitism(chromosomes,0.2)
    new_population = random_evolution(N_POPULAION,new_population,0.1)

    n_childs = N_POPULAION - len(new_population)

    parent1, parent2 = select(chromosomes)
    new_population = child_generator(parent1, parent2, new_population, n_childs)

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


    # -- velx ¨= 4, pip_y ¨= 512, player_heigth = 512, dist = 288
    y = chromosomes.forward(torch.from_numpy(np.array(parameters)).float())

    prob = np.random.normal(0.5, 0.1, 1)
    #print(parameters)
    #print(y)

    if y > 0.5:
        return 1
    else:
        return 0

    '''
    if y > 0.6:  # -- We supose al the weigths starts with 0.5 mean
        return 1
    elif y < 0.4 :
        return 0
    else:
        prob = np.random.rand()
        if prob  > 0.7:
            return 1
        else:
            return 0
    '''

if __name__ == "__main__":
    # -- Matrix with the chromosomes of each player sorted
    print()
