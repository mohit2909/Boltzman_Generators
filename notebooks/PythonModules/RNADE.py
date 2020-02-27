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

class RNADE(nn.Module):
    def __init__(self,dimer_atoms,solvent_atoms):
        super(RNADE, self).__init__()
        self.dimer_atoms = dimer_atoms
        self.solvent_atoms = solvent_atoms
        self.total_dims = self.dimer_atoms + self.solvent_atoms
        self.D = self.total_dims
        self.H = 64
        self.relu = nn.LeakyReLU(0.2, inplace = True)
        self.params = nn.ParameterDict({
            "V" : nn.Parameter(torch.randn(self.D, self.H)),
            "b" : nn.Parameter(torch.zeros(self.D)),
            "V2" : nn.Parameter(torch.randn(self.D, self.H)),
            "b2" : nn.Parameter(torch.zeros(self.D)),
            "W" : nn.Parameter(torch.randn(self.H, self.D)),
            "c" : nn.Parameter(torch.zeros(1, self.H)),
            "P" : nn.Parameter(torch.randn(self.D, self.H)),
        })
        nn.init.xavier_normal_(self.params["P"])
        nn.init.xavier_normal_(self.params["V"])
        nn.init.xavier_normal_(self.params["V2"])
        nn.init.xavier_normal_(self.params["W"])
        
    def forward(self, x):
        ai = self.params["c"].expand(x.size(0), -1)   #B x H
        p = self.params["P"].expand(x.size(0), -1)
        print(ai.size())
        a1=[]
        m1 = []
        for d in range(self.D):
            h_i = se.relu(ai*p[:,d,:]) #B x H
            alpha1 = torch.sigmoid( h_i.mm(self.params["V"][d:d+1,:].t() ) + self.params["b"][d:d+1] )*2  + pow(10,-1) + 0.5#  BxH *  Hx1  
            mean1 = h_i.mm(self.params["V2"][d:d+1,:].t() ) + self.params["b2"][d:d+1]
            a1.append(alpha1)
            m1.append(mean1)
                
            ai = x[:, d:d+1].mm(self.params["W"][:, d:d+1].t() ) + ai #Bx1 * 1xH =  BxH
        
        a1 = torch.cat(a1,1)
        m1 = torch.cat(m1,1)
        final_prob = torch.stack([m1,a1])       
        return final_prob

