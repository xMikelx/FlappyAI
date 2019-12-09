import sys
import numpy as np


# -- Return chromosomes of 2 best players
def select(chromosomes):
    return chromosomes[0], chromosomes[1]


# -- Gets 2 childs by mixing parameters from each parent
def crossover(parent1, parent2):
    cross_point = np.random.randint(1, len(parent1)-1)
    child1 = np.concatenate((np.array(parent1[0:cross_point]), np.array(parent2[cross_point:])))
    child2 = np.concatenate((np.array(parent2[0:cross_point]), np.array(parent1[cross_point:])))
    return child1, child2


def mutate(child1, child2):
    mutate_point = np.random.randint(0, len(child1))
    child1[mutate_point] = np.random.uniform()

    mutate_point = np.random.randint(0, len(child2))
    child2[mutate_point] = np.random.uniform()

    return child1, child2

def evolution(chromosomes, child1, child2):
    chromosomes[-1] = child1
    chromosomes[-2] = child2
    return chromosomes


def train(chromosomes):
    parent1, parent2 = select(chromosomes)
    child1, child2 = crossover(parent1, parent2)
    child1, child2 = mutate(child1, child2)
    new_population = evolution(chromosomes, child1, child2)
    return new_population


def initialize(n_population,n_param):
    chromosomes_arr = []
    for i in range(n_population):
        chromosomes_arr.append(np.random.uniform(size=n_param))

    return chromosomes_arr

def sigmoid(X):
   return 1/(1+np.exp(-X))

def predict(chromosomes, parameters):

    W1, W2, W3, W4, bias = chromosomes
    velx, pip_y, player_y, dist_pip_player = parameters

    # -- velx ¨= 4, pip_y ¨= 512, player_heigth = 512, dist = 288
    print(velx, pip_y, player_y, dist_pip_player)
    y = sigmoid(velx/4 * W1 + pip_y/482 * W2 + player_y/400 * W3 + dist_pip_player/400 * W4 + bias)
    print(y)
    if y > 0.95:  # -- We supose al the weigths starts with 0.5 mean
        return 1
    else:
        return 0


if __name__ == "__main__":
    # -- Matrix with the chromosomes of each player sorted
    print()
