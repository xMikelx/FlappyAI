import sys
import numpy as np

#-- Return chromosomes of 2 best players
def select(chromosomes):
    return chromosomes[0] , chromosomes[1]

#-- Gets 2 childs by mixing parameters from each parent
def crossover(parent1, parent2):
    cross_point = np.random.randint(1, 3)
    child1 = np.concatenate(parent1[0:cross_point], parent2[cross_point:])
    child2 = np.concatenate(parent2[0:cross_point], parent1[cross_point:])
    return child1, child2

def mutate(child1, child2):
    mutate_point = np.random.randint(0, 4)
    child1[mutate_point] = np.random.uniform()

    mutate_point = np.random.randint(0, 4)
    child2[mutate_point] = np.random.uniform()

    return child1, child2

def evolution(chromosomes,child1,child2):
    chromosomes[-1] = child1
    chromosomes[-2] = child2
    return chromosomes

def main(chromosomes):
    parent1, parent2 = select(chromosomes)
    child1, child2 = crossover(parent1, parent2)
    child1, child2 = mutate(child1,child2)
    new_population = evolution(chromosomes,child1,child2)
    return new_population


if __name__ == "__main__":
    #-- Matrix with the chromosomes of each player sorted
    chromosomes = sys.argv[1]
    main(chromosomes)