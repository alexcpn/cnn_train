"""
Pre trained  model from tutorial modified from
https://pytorch.org/hub/pytorch_vision_alexnet/
And for imagenette small dataset
Imagenet (tench, English springer, cassette player, chain saw, church, French horn, garbage truck, gas pump, golf ball, parachute).
https://github.com/fastai/imagenette

Load the Pre-trained models generated from test4_cnn_imagenet_small.py in the same folder
"""

from importlib.resources import path
from PIL import Image
from torchvision import transforms,datasets
import torch
from models import resnet, alexnet, mycnn, mycnn2
import os

test_images = ['test-tench.jpg','fish_boy.jpg','test-church.jpg','test-garbagetruck.jpg','test-truck.jpg','test-dog.jpg','train_dog.png',
"test-englishspringer.jpg","test_dogcartoon.jpg","test_chaingsaw.jpg","test_chainsawtrain.jpg","test_frenchhorn.jpg",
"test_frenchhorntrain.jpg","test-golfball.jpg"]


data_dir = "./imagenette2-320"
train_dir = os.path.join(data_dir, "train")
train_dataset = datasets.ImageFolder(train_dir,[])

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

# # Imagenette classes - labels for better description
# categories_ref = [
#     "English springer",
#     "tench",
#     "cassette player",
#     "chain saw",
#     "church",
#     "French horn",
#     "garbage truck",
#     "gas pump",
#     "golf ball",
#     "parachute",
# ]

# sort as value to fit the directory order to labels to be sure
print("Image to Folder Index",train_dataset.class_to_idx)
sorted_vals = dict(sorted(train_dataset.class_to_idx.items(), key=lambda item: item[1]))
categories =[]
for key in sorted_vals:
    classname = foldername_to_class[key]
    categories.append(classname)

print("Categories",categories)

# Choose a saved Model - assign the name you want to test with
# (assuming that you have trained the models)
modelname = "resnet50"

if modelname == "mycnn":
    model = mycnn.MyCNN()
    path = "mycnn_18:07_October142022.pth" 
    resize_to = transforms.Resize((227, 227))
if modelname == "mycnn2":
    model = mycnn2.MyCNN2()
    path ="mycnn2_16:43_October182022.pth"
    resize_to = transforms.Resize((227, 227))
if modelname == "alexnet":
    model = alexnet.AlexNet()
    path = "alexnet_15:08_August082022.pth"
    resize_to = transforms.Resize((227, 227))
if modelname == "resnet50":
    model = resnet.ResNet50(img_channel=3, num_classes=10)
    path ="RestNet50_11:45_November072022.pth"
    resize_to = transforms.Resize((227, 227))

path = "cnn/saved_models/" +path
model.load_state_dict(torch.load(path))
model.eval()

for filename in test_images:
    input_image = Image.open('./test-images/'+filename)
    preprocess = transforms.Compose(
        [
            resize_to,
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # IMPORTANT: normalize for pretrained models
        ]
    )
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to("cuda")
        model.to("cuda")

    with torch.no_grad():
        output = model(input_batch)
    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    # print(probabilities)
    print(f"Detecting for class {filename} model {modelname}")
    print("--------------------------------")
    # Show top categories per image
    top5_prob, top5_catid = torch.topk(probabilities, 2)
    for i in range(top5_prob.size(0)):
        print(categories[top5_catid[i]], top5_prob[i].item())
    print("--------------------------------")

