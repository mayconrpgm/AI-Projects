# AI for Doom



# Importing the libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# Importing the packages for OpenAI and Doom
import gym
from gym.wrappers import SkipWrapper
from ppaquette_gym_doom.wrappers.action_space import ToDiscrete

# Importing the other Python files
import experience_replay, image_preprocessing



# Part 1 - Building the AI

# Making the brain
# Convolutional Neural Network
class CNN(nn.Module):
    #constructor
    def __init__(self, number_actions):
        super(CNN, self).__init__()
        
        #convolutional layers
        #Conv2d - Applies the convolution considering a 2d image
        #in_channels - Number of input images. On the first layer is the number of color channels on the image, 
        #1 channel is a black and white image, 3 channels is a colored image  
        #out_channels - Number of output images, number of features you want to detect.
        #A commom practice is to start with 32 feature detectors
        #kernel_size - Size of the feature detector matrix, usually start with 2x2, 3x3 or 5x5
        self.convolution1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 5)
        self.convolution2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3)
        self.convolution3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 2)
        
        #full connected layers
        #Linear - Creates a full connection layer
        #in_features - number of input neurons, for the CNN, on the input layers the number of neurons is equal to the number of pixels
        #after flattening the images comming from the convolutional layer
        #out_features - number of neurons on the output of the layer
        self.full_connection1 = nn.Linear(in_features = self.count_neurons((1, 80, 80)), out_features = 40)
        self.full_connection2 = nn.Linear(in_features = 40, out_features = number_actions)
        
    def count_neurons(self, image_dim):
        #image_dim is a tuple that contains the number of channels of the image, the width and the height
        x = Variable(torch.rand(1, *image_dim))
        #the "*" in the image_dim parameter implies that each element of the tuple will be passed to the function as a single parameter
        #Applying max-pooling to the convoluted images and then using ReLU to keep only the positive values
        #max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False)
        x = F.relu(F.max_pool2d(self.convolution1(x), kernel_size=3, stride=2))
        x = F.relu(F.max_pool2d(self.convolution2(x), kernel_size=3, stride=2))
        x = F.relu(F.max_pool2d(self.convolution3(x), kernel_size=3, stride=2))
        
        #Creating the Flattening layer
        return x.data.view(1, -1).size(1)
        
    def forward(self, x):
        #1st convolution
        x = F.relu(F.max_pool2d(self.convolution1(x), kernel_size=3, stride=2))
        #2nd convolution
        x = F.relu(F.max_pool2d(self.convolution2(x), kernel_size=3, stride=2))
        #3rd convolution
        x = F.relu(F.max_pool2d(self.convolution3(x), kernel_size=3, stride=2))        
        #Flatenning
        x = x.view(x.size(0), -1)
        #Connection from the input layer of the hidden layer
        x = F.relu(self.fc1(x))
        #Connection to the hidden layer to the output layer
        x = self.fc2(x)
        
        return x

# Making the body



# Making the AI





# Part 2 - Training the AI with Deep Convolutional Q-Learning

# Getting the Doom environment


# Building an AI


# Setting up Experience Replay

    
# Implementing Eligibility Trace


# Making the moving average on 100 steps


# Training the AI


# Closing the Doom environment

