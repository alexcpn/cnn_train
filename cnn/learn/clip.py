
'''
Clip Board 
'''



np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
# Generate a random image
image_size = 32 
image_depth = 3
image = np.random.rand(image_size, image_size)
# to mimic RGB channel
image = np.stack([image,image,image], axis=image_depth-1) # 0 to 2
image = np.moveaxis(image, [2, 0], [0, 2])
print("Image Shape=",image.shape)
input_tensor = torch.from_numpy(image)

# Run this manually to figure the shape of the output
m =nn.Sequential(
  nn.Conv2d(in_channels=3,out_channels=6,kernel_size=5,stride=1), #[6,28,28]
  nn.ReLU(),
  nn.Conv2d(in_channels=6,out_channels=1,kernel_size=5,stride=1), #[1.24.24]
  nn.ReLU(),
  nn.Conv2d(in_channels=1,out_channels=10,kernel_size=5,stride=1), #[10,20,20] #changed channels to 10
   nn.ReLU()
  )
output = m(input_tensor.float())
print("Output Shape=",output.shape)
output = torch.flatten(output,start_dim=1,end_dim=-1)
print("Output Shape=",output.shape) #([10, 400])
layer = nn.Linear(400,10)
output = layer(output)
print("Output Shape=",output.shape) #[10, 10])
layer = nn.Linear(10,1)
output = layer(output)
print("Output Shape=",output.shape) #([10, 1])
logSoftmax = nn.LogSoftmax(dim=1)
output = logSoftmax(output)
print("Final Output Shape=",output.shape) #([10, 1])
print('-----------------------------------------------------')
lossFn = nn.CrossEntropyLoss()



# Run this through the model - CPU

model = model.MyCNN()
x = input_tensor.float()
ouput = model(x)
print("Model Output Shape=",output.shape) 
print('-----------------------------------------------------')


# Run this through the model - GPU

mymodel = mymodel.MyCNN()
x = input_tensor.float().to(device)
mymodel = mymodel.to(device)
ouput = mymodel(x)
print("Model Output Shape=",output.shape) 
print('-----------------------------------------------------')