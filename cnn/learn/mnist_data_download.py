"""
Download MNIST dataset
https://github.com/gradient-ai/LeNet5-Tutorial/blob/main/test.py
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

batch_size = 64
num_classes = 10
learning_rate = 0.001
num_epochs = 10


#Loading the dataset and preprocessing
torchvision.datasets.MNIST(root = './data_mnist',train = True,transform = transforms.Compose([transforms.Resize((32,32)),
    transforms.ToTensor(),transforms.Normalize(mean = (0.1307,), std = (0.3081,))]),download = True)


torchvision.datasets.MNIST(root = './data_mnist',train = False,transform = transforms.Compose([transforms.Resize((32,32)),
    transforms.ToTensor(),transforms.Normalize(mean = (0.1325,), std = (0.3105,))]),)