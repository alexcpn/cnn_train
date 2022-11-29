"""
 Using the CIFAR-10 dataset on a custom CNN
 Explanation and equations here https://alexcpn.medium.com/cnn-from-scratch-b97057d8cef4
"""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import logging as log
import alexnet
import mycnn

log.basicConfig(format='%(asctime)s %(message)s', level=log.INFO)


#-------------------------------------------------------------------------------------------------------
# Code
#-------------------------------------------------------------------------------------------------------

# Define relevant variables for the ML task
batch_size = 64
num_classes = 10
learning_rate = 0.001
num_epochs = 20 # actual 20 epochs

# Device will determine whether to run the training on GPU or CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    deviceid = torch.cuda.current_device()
    log.info(f"Gpu device {torch.cuda.get_device_name(deviceid)}")


#-------------------------------------------------------------------------------------------------------
# Load the model
#-------------------------------------------------------------------------------------------------------

# Alexnet model works well for CIFAR-10 when input is scaled to 227x227 (from 32x32)
#model = alexnet.AlexNet().to(device)
#resize = transforms.Resize((227, 227))

model = mycnn.MyCNN().to(device)
resize = transforms.Resize((32, 32))

#-------------------------------------------------------------------------------------------------------
# Load the data
#-------------------------------------------------------------------------------------------------------
# Use transforms.compose method to reformat images for modeling,
# and save to variable all_transforms for later use
# mean calculated like https://stackoverflow.com/a/69750247/429476
all_transforms = transforms.Compose([resize,
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                          std=[0.2470, 0.2435, 0.2616])
                                     #transforms.Normalize(mean=[0.1307], # for MNIST - one channel
                                     #                     std=[0.3081,]) # for MNIST - one channel
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
Ouput on MNIST dataset

Epoch [18/20], Step [800/938], Loss: 0.2882
Epoch [19/20], Step [400/938], Loss: 0.1440
Epoch [19/20], Step [800/938], Loss: 0.1439
Epoch [20/20], Step [400/938], Loss: 0.1450
Epoch [20/20], Step [800/938], Loss: 0.1349
Accuracy of the network on the 10000 test images: 88.7 %

Ouput of CIFAT10 dataset

Epoch [17/20], Step [400/782], Loss: 0.0120
Epoch [18/20], Step [400/782], Loss: 0.1071
Epoch [19/20], Step [400/782], Loss: 0.0254
Epoch [20/20], Step [400/782], Loss: 0.0167
Accuracy of the network on the 10000 test images: 50.74 % --> Bad

With
learning_rate = 0.0001
num_epochs = 40 

Epoch [37/40], Step [400/782], Loss: 2.0008
Epoch [38/40], Step [400/782], Loss: 2.1031
Epoch [39/40], Step [400/782], Loss: 2.0204
Epoch [40/40], Step [400/782], Loss: 2.0130
Accuracy of the network on the 10000 test images: 39.99 %

With BacthNorm and AveragePooling

Epoch [15/20], Step [400/782], Loss: 1.9575
Epoch [16/20], Step [400/782], Loss: 1.9851
Epoch [17/20], Step [400/782], Loss: 2.0514
Epoch [18/20], Step [400/782], Loss: 1.9648
Epoch [19/20], Step [400/782], Loss: 1.9385
Epoch [20/20], Step [400/782], Loss: 2.0371
Accuracy of the network on the 10000 test images: 46.93 %

"""

## AlexNet - Works !!

"""
Epoch [15/20], Step [400/782], Loss: 0.1792
Epoch [16/20], Step [400/782], Loss: 0.1993
Epoch [17/20], Step [400/782], Loss: 0.1874
Epoch [18/20], Step [400/782], Loss: 0.2119
Epoch [19/20], Step [400/782], Loss: 0.1802
Epoch [20/20], Step [400/782], Loss: 0.2543
Accuracy of the network on the 10000 test images: 83.2 %
"""