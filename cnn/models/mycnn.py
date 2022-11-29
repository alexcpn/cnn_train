# My CNN Model
# Author Alex Punnen
# https://alexcpn.medium.com/cnn-from-scratch-b97057d8cef4
# To get the filter use this https://docs.google.com/spreadsheets/d/1tsi4Yl2TwrPg5Ter8P_G30tFLSGQ1i29jqFagxNFa4A/edit?usp=sharing

import torch.nn as nn
import logging as log

log.basicConfig(format='%(asctime)s %(message)s', level=log.INFO)


class MyCNN(nn.Module):
    def __init__(self):
        self.output_cnn = 10000# 178084 
        super(MyCNN, self).__init__()
        self.cnn_stack = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=6,kernel_size=5,stride=1), # In[3,32,32]
            nn.BatchNorm2d(6), 
            nn.ReLU(),
            nn.AvgPool2d(kernel_size = 3, stride = 1),
            nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5,stride=1), #[6,28,28]
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size = 3, stride = 1),
            nn.Conv2d(in_channels=16,out_channels=8,kernel_size=5,stride=1), #[16,24,24] [32,20,20] [C, H,W] 
            # Note when images are added as a batch the size of the output is [N, C, H, W], where N is the batch size ex [1,10,20,20]
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size = 3, stride = 2),
            nn.Conv2d(in_channels=8,out_channels=4,kernel_size=5,stride=1), # [32,20,20]  [C, H,W] 
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size = 3, stride = 2)
            # out #[16,16,16]
            # Adding more layers to keep the FC layers small (out of memory)

        )
     
        self.linear_stack = nn.Sequential(
            nn.Linear(self.output_cnn,1000),
            nn.ReLU(),
            nn.Linear(1000,10)
            
        )
        self.flatten = nn.Flatten(start_dim=1,end_dim=-1)
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        logits = self.cnn_stack(x)
        log.debug("Shape of logits: %s", logits.shape)
        logits = self.flatten(logits) # Note flatten will flatten previous layer output to [N, C*H*W] ex [1,4000]
        log.debug("Shape of logits after flatten: %s", logits.shape) # [N, C*H*W]
        logits = self.linear_stack(logits)
        log.debug("Shape of logits after linear stack: %s", logits.shape) # [N,10]
        #logits = self.softmax(logits) #IMPORTANT: Softmax  is already there in CrossEntropyLoss
        #log.debug("Shape of logits after logSoftmax: %s", logits.shape) #batchsize, 10
        return logits

    def get_output(self,input_width:int,input_height:int,stride:int,kernel_size:int,out_channels:int,padding=0)-> tuple[int,int,int]:
            """
            get the output_width,output_height and output_depth after a convolution of input 
            param input_width:  input_width
            param input_height:  input_height
            param stride:  stride of the CNN
            param kernel_size:  kernel_size or filter size of the CNN
            param out_channels:  out_channels specified in the CNN layer
            param padding:  padding defaults to 0
            returns (output_width,output_height,output_depth)
            """
            # Formula (W) = (W-F + 2P)/S +1
            # Formula (H) = (H-F + 2P)/S +1
            output_width = (input_width - kernel_size + 2*padding)/stride +1
            output_height = (input_height - kernel_size + 2*padding)/stride +1
            output_depth = out_channels
            return (int(output_width),int(output_height),output_depth)
        