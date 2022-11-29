"""
Utility to check Precision and Recall of a  trained model
Author - Alex Punnen 
"""
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import logging as log
from models import resnet, alexnet, mycnn, mycnn2
import os
import sklearn.metrics as skmc #this has confusion matrix but need to give all in a shot ?




log.basicConfig(format="%(asctime)s %(message)s", level=log.INFO)

torch.cuda.empty_cache()

# -------------------------------------------------------------------------------------------------------
# Code
# -------------------------------------------------------------------------------------------------------

# Device will determine whether to run the training on GPU or CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    deviceid = torch.cuda.current_device()
    log.info(f"Gpu device {torch.cuda.get_device_name(deviceid)}")


# -------------------------------------------------------------------------------------------------------
# Select the model you want to train
# -------------------------------------------------------------------------------------------------------


# Choose a saved Model - assign the name you want to test with
# (assuming that you have trained the models)
modelname = "resnet50"
    
if modelname == "mycnn":
    model = mycnn.MyCNN()
    path =  "mycnn_11:49_October302022.pth" 
    resize_to = transforms.Resize((150, 150))
if modelname == "mycnn2":
    model = mycnn2.MyCNN2()
    path ="mycnn2_16:43_October182022.pth"
    resize_to = transforms.Resize((227, 227))
if modelname == "alexnet":
    model = alexnet.AlexNet()
    path = "./alexnet_15:08_August082022.pth"
    resize_to = transforms.Resize((227, 227))
if modelname == "resnet50":
    model = resnet.ResNet50(img_channel=3, num_classes=10)
    path = "./RestNet50_11:43_October072022.pth"   # trained with more dog images from imagenet
    path ="./RestNet50_11:45_November072022.pth"
    resize_to = transforms.Resize((227, 227))

path = "cnn/saved_models/" +path
model.load_state_dict(torch.load(path))
model.eval()

# -------------------------------------------------------------------------------------------------------
# Load the data from image folder
# -------------------------------------------------------------------------------------------------------

data_dir = "./imagenette2-320"


train_dir = os.path.join(data_dir, "train")
normalize_transform = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)


train_transforms = transforms.Compose(
    [resize_to, 
    transforms.RandomHorizontalFlip(),
    transforms.RandomGrayscale(),
    transforms.ToTensor(), normalize_transform]
)

val_dir = os.path.join(data_dir, "val")

val_transforms = transforms.Compose(
    [resize_to, transforms.ToTensor(), normalize_transform]
)


train_dataset = torchvision.datasets.ImageFolder(train_dir, train_transforms)

val_dataset = torchvision.datasets.ImageFolder(val_dir, val_transforms)

#-----------------------------------------------------------------------------------------------------
# Order the categories as per how Dataloader loads it
#-----------------------------------------------------------------------------------------------------

foldername_to_class = { 'dogs50A-train' : "dog",
                        'n01440764': "tench",
                        'n02979186': "cassette player", 
                        'n03000684': "chain saw",
                        'n03028079': "church",
                        'n03394916': "French horn",
                        'n03417042': "garbage truck",
                        'n03425413': "gas pump",
                        'n03445777':  "golf ball",
                        'n03888257': "parachute" }

# Imagenette classes - labels for better description
categories_ref = [
    "English springer",
    "tench",
    "cassette player",
    "chain saw",
    "church",
    "French horn",
    "garbage truck",
    "gas pump",
    "golf ball",
    "parachute",
]

# sort as value to fit the directory order to labels to be sure
print("Image to Folder Index",train_dataset.class_to_idx)
sorted_vals = dict(sorted(train_dataset.class_to_idx.items(), key=lambda item: item[1]))
categories =[]
for key in sorted_vals:
    classname = foldername_to_class[key]
    categories.append(classname)

print("Categories",categories)
# -------------------------------------------------------------------------------------------------------
# Initialise the data loaders
# -------------------------------------------------------------------------------------------------------

workers = 2
pin_memory = True
batch_size = 64

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True, #IMPORTANT otherwise the data is not shuffled
    num_workers=workers,
    pin_memory=pin_memory,
    sampler=None,
)

test_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=workers,
    pin_memory=pin_memory,
)


# -------------------------------------------------------------------------------------------------------
#  Test the model - Find accuracy and per class
# -------------------------------------------------------------------------------------------------------

print("Image to Folder Index",train_dataset.class_to_idx)

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    model.to("cuda")

confusion_matrix = np.zeros((len(categories),len(categories)))

# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    model.eval() #IMPORTANT set model to eval mode before inference
    # correct = 0
    # total = 0


    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        # ------------------------------------------------------------------------------------------
        # Predict for the batch of images
        # ------------------------------------------------------------------------------------------
        outputs = model(images)  #Outputs= torch.Size([64, 10]) Probability of each of the 10 classes
        _, predicted = torch.max(outputs.data, 1) # get the class with the highest Probability out Given 1 per image # predicted= torch.Size([64])
        # total += labels.size(0) #labels= torch.Size([64])  This is the truth value per image - the right class
        # correct += (predicted == labels).float().sum().item()  # Find which are correctly classified
        
        # Below illustrates the above Torch Tensor semantics
        # >>> import torch
        # >>> some_integers = torch.tensor((2, 3, 5, 7, 11, 13, 17, 19))
        # >>> some_integers3 = torch.tensor((12, 3, 5, 7, 11, 13, 17, 19))
        # >>> (some_integers ==some_integers3)*(some_integers == 3)
        # tensor([False,  True, False, False, False, False, False, False])
        # >>> ((some_integers ==some_integers3)*(some_integers >12)).sum().item()
        # 3
        
        # ------------------------------------------------------------------------------------------
        #  Lets check also which classes are wrongly predicted with other classes  to create a MultiClass confusion matrix
        # ------------------------------------------------------------------------------------------

        mask=(predicted != labels) # Wrongly predicted
        wrong_predicted =torch.masked_select(predicted,mask)
        wrong_labels =torch.masked_select(labels,mask)
        wrongly_zipped = zip(wrong_labels,wrong_predicted)

        mask=(predicted == labels) # Rightly predicted
        rightly_predicted =torch.masked_select(predicted,mask)
        right_labels =rightly_predicted #same torch.masked_select(labels,mask)
        rightly_zipped = zip(right_labels,rightly_predicted)
        
        # Note that this is for a single batch - add to the list associated with class
        for _,j in enumerate(wrongly_zipped):
            k = j[0].item() # label
            l = j[1].item() # predicted
            confusion_matrix[k][l] +=1
       
        # Note that this is for a single batch - add to the list associated with class
        for _,j in enumerate(rightly_zipped):
            k = j[0].item() # label
            l = j[1].item() # predicted
            confusion_matrix[k][l] +=1
    
    #print("Confusion Matrix1=\n",confusion_matrix)
    # ------------------------------------------------------------------------------------------
    # Print Confusion matrix in Pretty print format
    # ------------------------------------------------------------------------------------------
    print(categories)
    for i in range(len(categories)):
        for j in range(len(categories)):
            print(f"\t{confusion_matrix[i][j]}",end='')
        print(f"\t{categories[i]}\n",end='')
    # ------------------------------------------------------------------------------------------
    # Calculate Accuracy per class
    # ------------------------------------------------------------------------------------------
    print("---------------------------------------")
    print(f"Accuracy/precision from confusion matrix is {round(confusion_matrix.trace()/confusion_matrix.sum(),2)}")
    print("---------------------------------------")
    for i in range(len(categories)):
        print(f"---Accuracy for class {categories[i]} = {round(confusion_matrix[i][i]/confusion_matrix[i].sum(),2)}")
    
    # ---------------------------------------------------
    # Plot this in a good figure
    # ---------------------------------------------------
        

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title('Confusion Matrix', fontsize=18)
    ax.matshow(confusion_matrix, cmap=plt.cm.Blues, alpha=0.7)
    ax.set_xticklabels([''] + categories,rotation=90)
    ax.set_yticklabels([''] + categories)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(x=j, y=i,s=int(confusion_matrix[i, j]), va='center', ha='center', size='xx-small')
            if ( i==j):
                acc = round(confusion_matrix[i][i]/confusion_matrix[i].sum(),2)
                ax.text(x=len(categories)+1, y=i,s=acc, va='center', ha='center', size='xx-small')
    plt.savefig("confusion_matrix_"+modelname +".jpg")

    # correct = 0
    # total = 0
    # for images, labels in train_loader:
    #     images = images.to(device)
    #     labels = labels.to(device)
    #     outputs = model(images)
    #     _, predicted = torch.max(outputs.data, 1)
    #     total += labels.size(0)
    #     correct += (predicted == labels).float().sum().item()
    # # this is not really not needed- but just to cross check if what we calculated during training is accurate
    # print(
    #     "Accuracy of the network on the {} Train images: {} %".format(
    #         total, 100 * correct / total
    #     )
    # )

"""
Output 
['tench', 'English springer', 'cassette player', 'chain saw', 'church', 'French horn', 'garbage truck', 'gas pump', 'golf ball', 'parachute']
        333.0   17.0    7.0     12.0    0.0     1.0     6.0     0.0     8.0     3.0     tench
        10.0    314.0   4.0     41.0    3.0     1.0     10.0    0.0     7.0     5.0     English springer
        2.0     12.0    305.0   16.0    0.0     0.0     15.0    4.0     2.0     1.0     cassette player
        12.0    25.0    21.0    254.0   0.0     5.0     53.0    1.0     4.0     11.0    chain saw
        3.0     7.0     11.0    14.0    308.0   5.0     45.0    8.0     5.0     3.0     church
        14.0    36.0    49.0    50.0    3.0     207.0   24.0    7.0     2.0     2.0     French horn
        1.0     3.0     11.0    12.0    2.0     2.0     353.0   2.0     1.0     2.0     garbage truck
        1.0     9.0     71.0    28.0    7.0     1.0     85.0    211.0   1.0     5.0     gas pump
        8.0     15.0    16.0    24.0    3.0     2.0     13.0    2.0     291.0   25.0    golf ball
        4.0     6.0     3.0     20.0    6.0     1.0     13.0    3.0     8.0     326.0   parachute
---------------------------------------
Accuracy/precision from confusion matrix is 0.74
---------------------------------------
---Accuracy for class tench = 0.86
---Accuracy for class English springer = 0.79
---Accuracy for class cassette player = 0.85
---Accuracy for class chain saw = 0.66
---Accuracy for class church = 0.75
---Accuracy for class French horn = 0.53
---Accuracy for class garbage truck = 0.91
---Accuracy for class gas pump = 0.5
---Accuracy for class golf ball = 0.73
---Accuracy for class parachute = 0.84
"""