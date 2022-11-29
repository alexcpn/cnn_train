# My CNN Model
# https://alexcpn.medium.com/cnn-from-scratch-b97057d8cef4
# To get the filter use this https://docs.google.com/spreadsheets/d/1tsi4Yl2TwrPg5Ter8P_G30tFLSGQ1i29jqFagxNFa4A/edit?usp=sharing

import torch.nn as nn
import logging as log

log.basicConfig(format='%(asctime)s %(message)s', level=log.DEBUG)


class MyCNN2(nn.Module):


    def make_block(self,input_depth,out_channel,kernel_size,stride):
        cnn1 =  nn.Conv2d(in_channels=input_depth,out_channels=int(out_channel/2),kernel_size=kernel_size,stride=stride)
        #bn1 = nn.BatchNorm2d(int(out_channel/2))
        relu1 =  nn.ReLU()
        pool2 = nn.AvgPool2d(kernel_size = 3, stride = 1)
        cnn2 =  nn.Conv2d(in_channels=int(out_channel/2),out_channels=out_channel,kernel_size=kernel_size,stride=stride)
        #bn2 = nn.BatchNorm2d(out_channel)
        relu2 =  nn.ReLU()
        pool3 = nn.AvgPool2d(kernel_size = 3, stride = 2)
        cnn3 =  nn.Conv2d(in_channels=out_channel,out_channels=int(out_channel/2),kernel_size=kernel_size,stride=stride)
        #bn3 = nn.BatchNorm2d(int(out_channel/2))
        relu3 =  nn.ReLU()
        nn_stack = nn.Sequential(cnn1,relu1,pool2,cnn2,relu2,pool3,cnn3,relu3)
        return nn_stack

    def __init__(self):
        self.output_cnn = 19095# 178084 
        super(MyCNN2, self).__init__()
        self.input_width = 227
        self.input_height = 227
        self.in_channel =3
        out_channel =1
        

        self.flatten = nn.Flatten(start_dim=1,end_dim=-1)
        kernel_size = 6
        stride =1
        self.cnn_stack1 = self.make_block(self.in_channel,out_channel*16,kernel_size,stride)
        self.linear_stack = nn.Linear(83232,1152)
        
        kernel_size = 4
        stride =2 
        self.cnn_stack2 = self.make_block(self.in_channel,out_channel*16,kernel_size,stride)
        self.linear_stack2 = nn.Linear(1152,1000)
        
        kernel_size =2
        stride =4
        self.cnn_stack3 = self.make_block(self.in_channel,out_channel*16,kernel_size,stride)
        self.linear_stack3 = nn.Linear(32,100)
        
            

        self.linear_stack4 = nn.Linear(100,10)
        self.linear_stack5 = nn.Linear(1000,100)

    def forward(self, x):
        logits1 = self.cnn_stack1(x)
        logits1 = self.flatten(logits1)
        logits1 = self.linear_stack(logits1)
        
        logits2 = self.cnn_stack2(x)
        logits2 = self.flatten(logits2)
        logits2 = self.linear_stack2(logits2+logits1)

        logits3 = self.cnn_stack3(x)
        logits3 = self.flatten(logits3)
        logits3 = self.linear_stack3(logits3)
       
        logits =self.linear_stack5(logits2)
        logits =self.linear_stack4(logits3+logits)
        log.debug("Shape of logits after linear stack: %s", logits.shape) # [N,10]
        
        
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
    