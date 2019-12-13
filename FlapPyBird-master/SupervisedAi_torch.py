import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
import torch.optim as optim
import torch
import multiprocessing as mp

from Game import flappy_AI_Supervises

from torch.autograd import Variable
cuda = torch.cuda.is_available()


class Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 10)
        self.fc2 = nn.Linear(10, 10)
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

    print(log_prob)

    if y > 0.5:
        return 1,log_prob
    else:
        return 0,log_prob




def main():

    #PARAMETERS
    learning_rate = 1e-4  # Learning rate used to scale gradient updates
    n_games = 25 #games before updating model
    n_generations = 10000 #Total times we update model
    gamma = 0.99  # Discount factor when accumulating future rewards
    n_players = 1
    ###########################

    playerModel = initialize(20)  # 1 player 12 inputs


    opt = optim.Adam(playerModel.parameters(), lr=learning_rate)

    input_queue = mp.Queue()
    output_queue = mp.Queue()

    batch_final_rewards=[]
    best_batch_median = 0  # Variable to determine best model so far, for saving

    # Start game on a separate process:
    p = mp.Process(target=flappy_AI_Supervises.main, args=(input_queue, output_queue))
    p.start()

    for generation in range(n_generations):
        batch_log_prob = []
        batch_rewards = []

        for episode in range(n_games):  # Start episode at 1 for easier batch management with % later
            input_queue.put([episode,generation,n_players])  # This starts next episode
            input_queue.put([True,0])  # This starts next episode
            output_queue.get()  # Gets blank response to confirm episode started
            input_queue.put([False,0])  # Input initial action as no flap
            state, reward, done,id_player = output_queue.get()  # Get initial state

            episode_steps = 0  # Number of steps taken in current episode
            episode_reward = 0  # Amount of reward obtained from current episode


            while not done:
                flap_probability = playerModel(state)[0]  # Forward pass state through network to get flap probability

                prob_dist = Bernoulli(flap_probability)  # Generate Bernoulli distribution with given probability
                action = prob_dist.sample()  # Sample action from probability distribution
                log_prob = prob_dist.log_prob(action)  # Store log probability of action


                if action == 1:
                    input_queue.put([True,0])  # If action is 1, input True
                else:
                    input_queue.put([False,0])  # Otherwise, False

                state, reward, done,id_player = output_queue.get()  # Get resulting state and reward from above action

                batch_log_prob.append(log_prob)  # Store the log probability for loss calculation
                batch_rewards.append(reward)  # Store the reward obtained as a result of action

                episode_reward += reward  # Increase current episode's reward counter
                episode_steps += 1  # Increase number of steps taken on current episode


            #input_queue.put(True)  # Reset the game.

            batch_final_rewards.append(episode_reward)


            print('Batch {}, Episode {}  || Reward: {:.1f} || Steps: {} '.format(generation, episode, episode_reward,
                                                                             episode_steps))

        discounted_rewards = discount_rewards(batch_rewards, gamma)  # Discount rewards with discount factor gamma

        opt.zero_grad()  # Zero gradients to clear existing data
        for i in range(len(batch_log_prob)):
          loss = -batch_log_prob[i] * discounted_rewards[i]  # Calculate negative log likelihood loss, scaled by reward
          loss.backward()  # Backpropagate to calculate gradients

        print('Updating network weights.')
        opt.step()  # Update network weights using above accumulated gradients

        batch_median = np.median(batch_final_rewards)
        # If current model has best median performance, save:
        if batch_median > best_batch_median:
            print(
                'New best batch median {} (previously {}), saving network weights.'.format(batch_median, best_batch_median))
            best_batch_median = batch_median

            state = {
                'state_dict': playerModel.state_dict(),
                'optimizer': opt.state_dict(),
            }
            torch.save(state, 'model/trained-model.pt')

            # Load using:
            # state = torch.load(filepath)
            # agent.load_state_dict(state['state_dict']), opt.load_state_dict(state['optimizer'])

        else:
            print('Batch Median Reward: {}'.format(batch_median))





if __name__ == "__main__":
    main()
