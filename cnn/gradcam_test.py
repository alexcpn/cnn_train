"""
Using Grad CAM libray to visualize the gradients of the last layer
and to see if the model has learned the features of the images
"""

from pytorch_grad_cam import (
    GradCAM,
    HiResCAM,
    ScoreCAM,
    GradCAMPlusPlus,
    AblationCAM,
    XGradCAM,
    EigenCAM,
    FullGrad,
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


from importlib.resources import path
from PIL import Image
from torchvision import transforms, datasets
import torch
from models import resnet, alexnet, mycnn, mycnn2
import numpy as np
import os
import torch.nn as nn


#-----------------------------------------------------------------------------------------------------
# Order the categories as per how Dataloader loads it
#-----------------------------------------------------------------------------------------------------
data_dir = "./imagenette2-320"
train_dir = os.path.join(data_dir, "train")
train_dataset = datasets.ImageFolder(train_dir,[])

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


# sort as value to fit the directory order to labels to be sure
print("Image to Folder Index",train_dataset.class_to_idx)
sorted_vals = dict(sorted(train_dataset.class_to_idx.items(), key=lambda item: item[1]))
categories =[]
for key in sorted_vals:
    classname = foldername_to_class[key]
    categories.append(classname)

print("Categories",categories)

# Choose a saved Model - comment out the rest
modelname = "resnet50"

# Choose a saved Model - assign the name you want to test with
# (assuming that you have trained the models)
#modelname = "resnet50"
resize_size =(1,1)
if modelname == "mycnn":
    resize_size = (227, 227)
    model = mycnn.MyCNN()
    path = "mycnn_18:07_October142022.pth"
    path ="mycnn_13:01_October272022.pth" #dual class only
    resize_to = transforms.Resize(resize_size)
if modelname == "mycnn2":
    resize_size = (227, 227)
    model = mycnn2.MyCNN2()
    path ="mycnn2_16:43_October182022.pth"
    resize_to = transforms.Resize(resize_size)
if modelname == "alexnet":
    resize_size = (227,227)
    model = alexnet.AlexNet()
    path = "alexnet_20:56_October102022.pth"
    resize_to = transforms.Resize(resize_size)
if modelname == "resnet50":
    model = resnet.ResNet50(img_channel=3, num_classes=10)
    resize_size =(227,227)
    #path = "./RestNet50_12:26_August082022.pth" # without augumentation
    path = "RestNet50_13:49_September102022.pth" #with augumentation
    path = "RestNet50_16:54_October062022.pth" #with cartoon dogs
    path = "RestNet50_11:43_October072022.pth"   # trained with more dog images from imagenet
    path ="RestNet50_11:45_November072022.pth" #227*227
    resize_to = transforms.Resize(resize_size)

path = "cnn/saved_models/" +path
model.load_state_dict(torch.load(path))
model.eval()


def flatten_model(modules):
    def flatten_list(_2d_list):
        flat_list = []
        # Iterate through the outer list
        for element in _2d_list:
            if type(element) is list:
                # If the element is of type list, iterate through the sublist
                for item in element:
                    flat_list.append(item)
            else:
                flat_list.append(element)
        return flat_list

    ret = []
    try:
        for _, n in modules:
            ret.append(loopthrough(n))
    except:
        try:
            if str(modules._modules.items()) == "odict_items([])":
                ret.append(modules)
            else:
                for _, n in modules._modules.items():
                    ret.append(loopthrough(n))
        except:
            ret.append(modules)
    return flatten_list(ret)


# Set the target for the CAM layer; Add all the layers in the model
target_layers =[]
module_list =[module for module in model.modules()]


flatted_list= flatten_model(module_list)


print("------------------------")
for count, value in enumerate(flatted_list):
    
    if isinstance(value, (nn.Conv2d,nn.AvgPool2d,nn.BatchNorm2d)):
    #if isinstance(value, (nn.Conv2d)):
        print(count, value)
        target_layers.append(value)

print("------------------------")


# Alternative is to add specific layers and check
# if modelname=='resnet50':
#     #target_layers = [module_list[142],module_list[143],module_list[144],module_list[145],module_list[146],module_list[147]]
#     target_layers = [module_list[35],module_list[36],module_list[37],module_list[38],module_list[39],module_list[40]]
# if modelname=='mycnn':
#     target_layers = [module_list[11],module_list[8],module_list[5],module_list[2],module_list[4],module_list[7],module_list[10],module_list[13]] # CNN and Avg pooling
#     #target_layers = [module_list[11],module_list[8],module_list[5],module_list[2]] # CNN only

    
# Construct the CAM object once, and then re-use it on many images:
cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

# Load the images

test_images = ['test-tench.jpg','fish_boy.jpg','test-church.jpg','test-garbagetruck.jpg','test-truck.jpg','test-dog.jpg','train_dog.png',
"test-englishspringer.jpg","test_dogcartoon.jpg","test_chaingsaw.jpg","test_chainsawtrain.jpg","test_frenchhorn.jpg",
"test_frenchhorntrain.jpg","test-golfball.jpg"]

for filename in test_images:
    input_image = Image.open('./test-images/'+filename)

    preprocess = transforms.Compose(
        [
            resize_to,
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # IMPORTANT: normalize for pre-trained models
        ]
    )
    input_tensor = preprocess(input_image)
    print("Input Tensor Shape:", input_tensor.shape)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to("cuda")
        model.to("cuda")
        input_tensor = input_tensor.to("cuda")

    # We have to specify the target we want to generate
    # the Class Activation Maps for.
    # If targets is None, the highest scoring category
    # will be used for every image in the batch.
    # Here we use ClassifierOutputTarget, but you can define your own custom targets

    # That are, for example, combinations of categories, or specific outputs in a non standard model.
    targets = [ClassifierOutputTarget(6)] #0 for finch ?

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_batch, targets=None,aug_smooth=True)
    #print( "len grayscale_cam",len(grayscale_cam),grayscale_cam.shape)

    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    # from PIL import Image
    # im = Image.fromarray(grayscale_cam)
    # if im.mode != 'RGB':
    #     im = im.convert('L')
    # im.save("grayscale_cam.jpeg"

    img=np.array(input_image.resize(resize_size),np.float32)
    img = img.reshape(img.shape[1],img.shape[0],img.shape[2])
    #print("img shape",img.shape,img.max())
    img = img/255
    visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    #cam_images = [show_cam_on_image(img, grayscale, use_rgb=True) for img, grayscale in zip(input_image, grayscale_cam)]
    visualization = Image.fromarray(visualization)
    out_file_name ="cnn/gradcam_out/" +modelname+ "_" + os.path.basename(filename)
    visualization.save(out_file_name)
    #print("Visualization saved- now trying to show (GUI mode)")
    im = Image.open(out_file_name)
    im.show()
    




