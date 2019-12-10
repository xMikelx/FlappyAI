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
        x = F.relu(x)
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


def random_evolution(chromosomes):
    for i in range(2,len(chromosomes) - 2):
        prob = np.random.rand()
        if prob > 0.5:
          chromosomes[i] = Net(2,1)
    return chromosomes

def train(chromosomes):
    parent1, parent2 = select(chromosomes)
    child1, child2 = crossover(parent1, parent2)
    child1, child2 = mutate(child1, child2)
    new_population = evolution(chromosomes, child1, child2)
    new_population = random_evolution(new_population)


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
    print(parameters)
    print(y)

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
