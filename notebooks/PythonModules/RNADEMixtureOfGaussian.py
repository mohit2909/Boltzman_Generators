import torch
import torch.nn as nn
from pathlib import Path
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
from math import *
from random import gauss,seed
import torch.nn.functional as F

class RNADE2(nn.Module):
    def __init__(self,dimer_atoms,solvent_atoms):
        super(RNADE2, self).__init__()
        self.dimer_atoms = dimer_atoms
        self.solvent_atoms = solvent_atoms
        self.D = self.dimer_atoms + self.solvent_atoms
        self.H = 64
        self.K = 3
        self.params = nn.ParameterDict({
            "V" : nn.Parameter(torch.randn(self.D, self.H)),
            "b" : nn.Parameter(torch.zeros(self.D)),
            "V2" : nn.Parameter(torch.randn(self.D, self.H)),
            "b2" : nn.Parameter(torch.zeros(self.D)),
            "Vmean" : nn.Parameter(torch.randn(self.D,self.H, self.K)),
            "Valpha" : nn.Parameter(torch.randn(self.D,self.H, self.K)),
            "Vstd" : nn.Parameter(torch.randn(self.D,self.H, self.K)),
            "bmean" : nn.Parameter(torch.zeros(self.D,self.K)),
            "balpha" : nn.Parameter(torch.zeros(self.D,self.K)),
            "bstd" : nn.Parameter(torch.zeros(self.D,self.K)),
            "W" : nn.Parameter(torch.randn(self.H, self.D)),
            "c" : nn.Parameter(torch.zeros(1, self.H)),
        })
        nn.init.xavier_normal_(self.params["V"])
        nn.init.xavier_normal_(self.params["V2"])
        nn.init.xavier_normal_(self.params["W"])
        nn.init.xavier_normal_(self.params["Vmean"])
        nn.init.xavier_normal_(self.params["Valpha"])
        nn.init.xavier_normal_(self.params["Vstd"])
        
    def forward(self, x):
        ai = self.params["c"].expand(x.size(0), -1)   #B x H
        a= None
        m = None
        s = None
        for d in range(self.D):
            if(d<4):
                ai = x[:, d:d+1].mm(self.params["W"][:,d:d+1].t()) + ai
                continue
            h_i = torch.relu(ai) #B x H
            std = torch.sigmoid( ( h_i.mm(self.params["Vstd"][d,:,] ) + self.params["bstd"][d:d+1,:].expand(x.size(0), -1) ) )*2  + pow(10,-1) + 0.5#  BxH *  HxK = BxK  
            mean = ( h_i.mm(self.params["Vmean"][d,:,] ) + self.params["bmean"][d:d+1,:].expand(x.size(0), -1) ) #B xH  * HxK  = B x K + BxK
            alpha = torch.softmax( (h_i.mm(self.params["Valpha"][d,:,] ) +self.params["balpha"][d:d+1,:].expand(x.size(0), -1) ), dim = 1 )
            #print(alpha[0])
            if(a is not None):
                a = torch.cat((a, alpha.unsqueeze(dim = 0)),0)
                m = torch.cat((m, mean.unsqueeze(dim = 0)) , 0)
                s = torch.cat((s, std.unsqueeze(dim = 0)) , 0)
            else:
                a = alpha.unsqueeze(dim=0)
                m = mean.unsqueeze(dim=0)
                s = std.unsqueeze(dim=0)
            ai = x[:, d:d+1].mm(self.params["W"][:, d:d+1].t() ) + ai #Bx1 * 1xH =  BxH
        
        m = m.permute(1,0,2 )
        a = a.permute(1,0,2)
        s = s.permute(1,0,2)
        #print(a.size(),m.size(),s.size())
        #final_prob = torch.stack([m,s,a]) 
        #print(final_prob.size())
        return [m,s,a]
    def mixtureSample(self, pm):
        rn = torch.randn( pm.size()[0] )
        alpha = pm[0]
        mean = pm[1]
        ans = []
        std = pm[2]
        for i in range(alpha.size()[0]):
            while(1):
                r = np.randn()*6  -3
                prob = 0.0
                for j in range(alpha.size()[2] ):
                    prob = prob + ( alpha[i,j] * np.exp(-0.5* ((r-mean[i,j])/std[i,j])**2) / ( std[i,j]  * sqrt(2*3.14)) ).detach().to_numpy()
                if(prob>= np.randn()):
                    ans.append(prob)
                    break;
                print(prob)
        
            
    def sample(self,x):
        ai = self.params["c"].expand(x.size(0), -1)   #B x H
        a= None
        m = None
        s = None
        for d in range(self.D):
            if(d<4):
                ai = x[:, d:d+1].mm(self.params["W"][:,d:d+1].t()) + ai
                continue
            h_i = torch.relu(ai) #B x H
            std = torch.sigmoid( ( h_i.mm(self.params["Vstd"][d,:,] ) + self.params["bstd"][d:d+1,:].expand(x.size(0), -1) ) )*2  + pow(10,-1) + 0.5#  BxH *  HxK = BxK  
            mean = ( h_i.mm(self.params["Vmean"][d,:,] ) + self.params["bmean"][d:d+1,:].expand(x.size(0), -1) ) #B xH  * HxK  = B x K + BxK
            alpha = torch.softmax( (h_i.mm(self.params["Valpha"][d,:,] ) +self.params["balpha"][d:d+1,:].expand(x.size(0), -1) ), dim = 1 )
            samples = self.mixtureSample([alpha,mean,std])
            ai = x[:, d:d+1].mm(self.params["W"][:, d:d+1].t() ) + ai
