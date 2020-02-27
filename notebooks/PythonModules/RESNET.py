import torch
import torch.nn as nn
from pathlib import Path
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
from math import *
from random import gauss,seed
import torch.nn.functional as F
import math

class DensityEstimator(nn.Module):
    def __init__(self,dimer_atoms,output_dim):
        super(DensityEstimator, self).__init__()
        self.dimer_atoms = dimer_atoms
        self.hidden_dim = 16
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.fc1 = nn.Linear(self.dimer_atoms, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim*4)
        self.fc3 = nn.Linear(self.hidden_dim*4, self.hidden_dim*4*4)
        #self.lstm = nn.LSTM(hidden_dim, output_dim)
        #self.fc3 = nn.Linear(self.hidden_dim*4*4, hidden_dim*4)
        self.fc4 = nn.Linear(self.hidden_dim*4*4, output_dim)
    
    def forward(self,x):
        out = self.fc1(x)
        out = self.relu(out)
        #out,_ = self.lstm(out.view(self.boxes,1,-1))
        out = self.relu(self.fc2(out))
        out = self.relu(self.fc3(out) )
        out = self.relu(self.fc4(out))
        out = 30.2*torch.softmax(out,1)
        return out


class Residual_Block(nn.Module):
    def __init__(self):
        super(Residual_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.in1 = nn.InstanceNorm2d(64, affine=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.in2 = nn.InstanceNorm2d(64, affine=True)

    def forward(self, x):
        identity_data = x
        output = self.in1( self.relu(self.conv1(x)) )
        output = self.in2( self.conv2(output) )
        output = torch.add(output,identity_data)
        return output


class RESNET(nn.Module):
    def __init__(self,numberOfLayers= 18):
        super(RESNET, self).__init__()
        self.conv_input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.residual = self.make_layer(Residual_Block, numberOfLayers)
        self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_mid = nn.InstanceNorm2d(64, affine=True)

        self.upscale2x = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=5, stride=1, padding=2, bias= False)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.conv_input(x))
        residual = out
        out = self.residual(out)
        #out= self.bn_mid(self.conv_mid(out))
        #print(out.shape)
        out = torch.add(out,residual)
        #print(out.shape)
        out = self.upscale2x(out)
        #print(out.shape)
        out = self.conv_output(out)
        return out
        return torch.relu(out)
        t = torch.softmax( torch.reshape(out,(out.shape[0],1,out.shape[2]*out.shape[2])), 2 )
        print(torch.reshape(out,(out.shape[0],out.shape[1],out.shape[2],out.shape[3])).shape )
        #print(torch.softmax( torch.flatten(out[:,:,]))
        return torch.reshape(out,(out.shape[0],out.shape[1],out.shape[2],out.shape[3]))
        return out
