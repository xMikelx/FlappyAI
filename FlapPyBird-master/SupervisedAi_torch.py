import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
import torch.optim as optim
import torch

from Game import flappy_AI_Supervises

from torch.autograd import Variable
cuda = torch.cuda.is_available()


class Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 10)
        self.fc3 = nn.Linear(10, output_dim)


    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        # -----
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.sigmoid(x)
        return x

def discount_rewards(r, gamma):
    # This function performs discounting of rewards by going back
    # and punishing or rewarding based upon final outcome of episode
    disc_r = np.zeros_like(r, dtype=float)
    running_sum = 0
    for t in reversed(range(0, len(r))):
        if r[t] == -1:  # If the reward is -1...
            running_sum = 0  # ...then reset sum, since it's a game boundary
        running_sum = running_sum * gamma + r[t]
        disc_r[t] = running_sum

    # Here we normalise with respect to mean and standard deviation:
    discounted_rewards = (disc_r - disc_r.mean()) / (disc_r.std() + np.finfo(float).eps)
    # Note that we add eps in the rare case that the std is 0
    return discounted_rewards


def train(agent, opt, generation_reward,generation_logprob): #chromosomes are the models sorted by points

    gamma = 0.99  # Discount factor when accumulating future rewards
    opt.zero_grad()  # Zero gradients to clear existing data

    discounted_rewards = discount_rewards(generation_reward, gamma)

    for i in range(len(generation_logprob)):
        loss = -generation_logprob[i] * discounted_rewards[i]  # Calculate negative log likelihood loss, scaled by reward
        loss.backward()  # Backpropagate to calculate gradients

    print('Updating network weights.')
    opt.step()  # Update network weights using above accumulated gradients

    return agent,opt


def initialize(n_param):
    model = Net(n_param, 1)

    return model

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def predict(agent,parameters):

    parameters = parameters[0,:]
    # -- velx ¨= 4, pip_y ¨= 512, player_heigth = 512, dist = 288
    y = agent.forward(torch.from_numpy(np.array(parameters)).float())

    prob_dist = Bernoulli(y)  # Generate Bernoulli distribution with given probability
    action = prob_dist.sample()  # Sample action from probability distribution
    log_prob = prob_dist.log_prob(action)

    if y > 0.5:
        return 1,log_prob
    else:
        return 0,log_prob


def main():
    playerModel = initialize(20)  # 1 player 12 inputs

    learning_rate = 1e-4  # Learning rate used to scale gradient updates
    opt = optim.Adam(playerModel.parameters(), lr=learning_rate)

    flappy_AI_Supervises.main()


if __name__ == "__main__":
    # -- Matrix with the chromosomes of each player sorted

    main()
    print()
