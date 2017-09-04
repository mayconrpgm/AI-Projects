# AI for Self Driving Car

# Importing the libraries

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

# Creating the architecture of the Neural Network

class Network(nn.Module):
    #inputSize: Number of neurons on the input layer
    #numberOfActions: Number of actions to be performed (the number of neurons on the output layer)
    def __init__(self, inputSize, numberOfActions):
        super(Network, self).__init__()
        self.inputSize = inputSize
        self.numberOfActions = numberOfActions
        numberOfHiddenNeurons = 30
        #Estabilishing the connection between the input layer and the hidden layer
        self.fullConnection1 = nn.Linear(inputSize, numberOfHiddenNeurons)
        #Estabilishing the connection between the hidden layer and the output layer
        self.fullConnection2 = nn.Linear(numberOfHiddenNeurons, numberOfActions)
        
    def forward(self, state):
        #F.relu: the rectifier activation function (https://qph.ec.quoracdn.net/main-qimg-b0a1423dbff251a5d46117cc72943d2b)
        x = F.relu(self.fullConnection1(state))
        qValues = self.fullConnection2(x)
        return qValues
    
# Implementing Experience Replay

class ReplayMemory(object):
    
    #capacity: maximun number of transition events that will be stored
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
    
    #adds an event to the memory pile
    def push(self, event):
        self.memory.append(event)
        #if there are more events than the estipulated capacity, remove the oldest event
        if len(self.memory) > self.capacity:
            del self.memory[0]
    
    #extracts a sample of events from the memory pile
    def sample(self, batchSize):
        #what the zip function does:
        #if list ((1,2,3),(4,5,6)), then zip(*list) = ((1,4),(2,5),(3,6))
        samples = zip(*random.sample(self.memory, batchSize))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)
        
# Implementing Deep Q Learning
        
class Dqn():
    #gamma: the discount coefficient on the q-learning function
    def __init__(self, inputSize, numberOfActions, gamma):
        self.gamma = gamma
        #A reward window of the last x amount of rewards.
        #The mean of the rewards will measure the performance of the AI
        self.rewardWindow = []
        self.model = Network(inputSize, numberOfActions)
        
        capacity = 100000
        self.memory = ReplayMemory(capacity)
        #using the Adam optimizer from pytorch. Used for the stochastic gradient descent
        learningRate = 0.001
        self.optimizer = optim.Adam(self.model.parameters(), lr = learningRate)
        #creating a tensor
        #.unsqueeze: inserts a fake dimension at the start of the tensor, making it bi-dimensional
        #x = torch.Tensor([1,2,3,4])    
        # 1
        # 2
        # 3
        # 4
        #[torch.FloatTensor of size 4]
        #x.unsqueeze(0)
        # 1  2  3  4
        #[torch.FloatTensor of size 1x4]
        self.lastState = torch.Tensor(inputSize).unsqueeze(0)
        self.lastAction = 0
        self.lastReward = 0
        
    def selectAction(self, state):
        #The temperature will define how sure the neural network will be when selecting an action
        temperature = 100
        probabilities = F.softmax(self.model(Variable(state, volatile = True)) * temperature)
        # softmax([1,2,3]) = [0.04, 0.11, 0.85] => softmax([1,2,3] * 3) = [0,0.02,0.98]
        #get a random action based on the probabilities distribution
        action = probabilities.multinomial()
        return action.data[0,0]

    def learn(self, batchState, batchNextState, batchReward, batchAction):
        #prediction of the neural network for each state
        outputs = self.model(batchState).gather(1, batchAction.unsqueeze(1)).squeeze(1)
        nextOutputs = self.model(batchNextState).detach().max(1)[0]
        target = self.gamma * nextOutputs + batchReward        
        #F.smooth_l1_loss => Huber Loss
        temporalDifferenceLoss = F.smooth_l1_loss(outputs, target)
        #applying stochastic gradient descent
        #the optimizer must be reinitialized for every iteration by using the zero_grad() method
        self.optimizer.zero_grad()
        temporalDifferenceLoss.backward(retain_variables = True)
        self.optimizer.step()
        
    def update(self, reward, newSignal):
        newState = torch.Tensor(newSignal).float().unsqueeze(0)
        self.memory.push((self.lastState, newState, torch.LongTensor([int(self.lastAction)]), torch.Tensor([self.lastReward])))
        action = self.selectAction(newState)
        
        #the learningSample defines after how many state transitions the neural network will 
        #start to learn from the samples extracted from the memory
        learningSample = 100
        if len(self.memory.memory) > learningSample:
            batchState, batchNextState, batchAction, batchReward = self.memory.sample(learningSample)
            self.learn(batchState, batchNextState, batchReward, batchAction)

        self.lastAction = action
        self.lastState = newState
        self.lastReward = reward
        self.rewardWindow.append(reward)

        rewardWindowSize = 1000
        if len(self.rewardWindow) > rewardWindowSize:
            del self.rewardWindow[0]
        
        return action
        
    def score(self):
        #returns the average of the rewards in the reward windows
        return sum(self.rewardWindow) / (len(self.rewardWindow) + 1.)
    
    def save(self):
        #saves the current weights of the neural network and the optimizer into a file
        torch.save({'state_dict': self.model.state_dict(), 
                    'optimizer': self.optimizer.state_dict()
                    }, 'last_brain.pth')
                    
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoing...")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done!")
        else:
            print("no checkpoint found...")