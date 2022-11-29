# Test's with Convolutional Neural Network

This contains a simple custom CNN model and other simpler and older models like AlexNet and ResNet50 using PyTorch.

This repo is to understand each model in depth. Also understanding CNN's via training and testing these models with different images.

## Training

Training is happening in [cnn/train_cnn.py](cnn/train_cnn.py) script

To train a small subset of Imagenet is used called Imagenette; It is a subset of 10 easily classified classes from 
Imagenet (tench, English springer, cassette player, chain saw, church, French horn, garbage truck, gas pump, golf ball, parachute).

## Training Data 

 https://github.com/fastai/imagenette

Also uploaded in [S3 in Blackblaze](https://tree-sac0-0008.secure.backblaze.com/b2_browse_files2.htm) but it may get deleted

### Training Infrastructure

I am running the training and inference in GPU in a RTX3060 Laptop (Acer Nitro5) with PopOS (from System32) and 16 GB memory.Any similar or better infrastructure should suffice. Training time for this small data set is about 10 to 20 minutes

### Training

```
/usr/bin/python3 /home/alex/coding/cnn_2/cnn/train_cnn.py
```
Trained models are save in [saved_models](cnn/saved_models/) folder
## Testing 

Testing is happening in [cnn/test_cnn.py](cnn/test_cnn.py)

```
 /usr/bin/python3 /home/alex/coding/cnn_2/cnn/test_cnn.py
 ```

The trained models are tested with some test images pulled from the internet.

## Model Accuracy and Multilabel Confusion matrix

```
usr/bin/python3 /home/alex/coding/cnn_2/cnn/model_accuracy.py
```

## Model Explain-ability GradCam

Gradcam helps one visualize which parts of the images are important for the CNN when it classifies an object with high probability. After testing a model, you can use this to visualize and debug the test results

```
/usr/bin/python3 /home/alex/coding/cnn_2/cnn/gradcam_test.py
```

Output images are stored in  [gradcam_out](cnn/gradcam_out/) folder.

Example output for classification of FrenchHorn by the ResNet50 model here 

![](https://i.imgur.com/vhxaB2d.png)



 Related repo: https://github.com/alexcpn/cnn_in_python

 
 