"""
 Using the LeNet 5 Model that performed well for MNIST and modified for CIFAR-10 slightly
 Basically does not work so well for color
"""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import sys
import logging as log

log.basicConfig(format='%(asctime)s %(message)s', level=log.DEBUG)


#Defining the convolutional neural network
# from https://blog.paperspace.com/writing-lenet5-from-scratch-in-python/
class LeNet5(nn.Module):
    def __init__(self, num_classes):
        super(LeNet5, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Linear(400, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out

# Define relevant variables for the ML task
batch_size = 64
num_classes = 10
learning_rate = 0.001
num_epochs = 30 # actual 20 epochs

# Device will determine whether to run the training on GPU or CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    deviceid = torch.cuda.current_device()
    log.info(f"Gpu device {torch.cuda.get_device_name(deviceid)}")

# Load the data

# Use transforms.compose method to reformat images for modeling,
# and save to variable all_transforms for later use
# mean calculated like https://stackoverflow.com/a/69750247/429476
all_transforms = transforms.Compose([transforms.Resize((32,32)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                          std=[0.2470, 0.2435, 0.2616])
                                     ])
# Create Training dataset
train_dataset = torchvision.datasets.CIFAR10(root = './data',
                                             train = True,
                                             transform = all_transforms,
                                             download = True)

# Create Testing dataset
test_dataset = torchvision.datasets.CIFAR10(root = './data',
                                            train = False,
                                            transform = all_transforms,
                                            download=True)

# Instantiate loader objects to facilitate processing
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = batch_size,
                                           shuffle = True)


test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                           batch_size = batch_size,
                                           shuffle = True)



model = LeNet5(10).to(device)


# initialize our optimizer and loss function
opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
lossFn = nn.CrossEntropyLoss()


for x, y in train_loader:
  log.info(f"Shape of X [N, C, H, W]: {x.shape}")
  log.info(f"Shape of y: {y.shape} {y.dtype}")
  # test one flow
  #pred = model(x)
  #loss = lossFn(pred, y)
  break
total_step = len(train_loader)
# loop over our epochs
for epoch in range(0, num_epochs):
  # set the model in training mode
  model.train()
  # initialize the total training and validation loss
  totalTrainLoss = 0
  totalValLoss = 0
  # initialize the number of correct predictions in the training
  # and validation step
  trainCorrect = 0
  valCorrect = 0
	# loop over the training set
  for i, (x, y) in enumerate(train_loader):
    # send the input to the device
    (x, y) = (x.to(device), y.to(device))
    # perform a forward pass and calculate the training loss
    pred = model(x)
    loss = lossFn(pred, y)
    # zero out the gradients, perform the backpropagation step,
    # and update the weights
    opt.zero_grad()
    loss.backward()
    opt.step()
    if (i+1) % 400 == 0:
          print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                      .format(epoch+1, num_epochs, i+1, total_step, loss.item()))


# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
  
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

"""
Okay it seem LeNet5 for CIFAR10 accuracy is only about 60 percent
https://towardsdatascience.com/convolutional-neural-network-champions-part-1-lenet-5-7a8d6eb98df6
Epoch [16/30], Step [400/782], Loss: 0.9487
Epoch [17/30], Step [400/782], Loss: 0.8363
Epoch [18/30], Step [400/782], Loss: 0.7358
Epoch [19/30], Step [400/782], Loss: 0.6749
Epoch [20/30], Step [400/782], Loss: 0.6875
Epoch [21/30], Step [400/782], Loss: 0.6840
Epoch [22/30], Step [400/782], Loss: 0.7196
Epoch [23/30], Step [400/782], Loss: 0.5475
Epoch [24/30], Step [400/782], Loss: 0.7285
Epoch [25/30], Step [400/782], Loss: 0.5338
Epoch [26/30], Step [400/782], Loss: 0.6070
Epoch [27/30], Step [400/782], Loss: 0.6122
Epoch [28/30], Step [400/782], Loss: 0.6509
Epoch [29/30], Step [400/782], Loss: 0.5590
Epoch [30/30], Step [400/782], Loss: 0.6796
Accuracy of the network on the 10000 test images: 62.6 %

"""