import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self,inchannel,outchannel,stride = 1,shortcut = None):

        super().__init__()
        self.left = nn.Sequential(
            nn.Conv3d(inchannel,outchannel,3,stride,1),
            nn.BatchNorm3d(outchannel),
            nn.ReLU(),
            nn.Conv3d(outchannel,outchannel,3,1,1), 
            nn.BatchNorm3d(outchannel)
         )
        self.right = shortcut

    def forward(self, input):
        out = self.left(input)
        residual = input if self.right is None else self.right(input)
        out+=residual
        return F.relu(out)

    
class ResModel(nn.Module):

    def __init__(self, num_class=6, in_ch = 3):
        super().__init__()
    
        self.pre = nn.Sequential(
            nn.Conv3d(in_ch,8,7,2,3),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.MaxPool3d(3,2,1)
        )

        self.layer1 = self._make_layer(8,16,3)
        self.layer2 = self._make_layer(16,32,4,stride=2) 
        self.layer3 = self._make_layer(32,64,6,stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(64,num_class)

    def _make_layer(self,inchannel,outchannel,block_num,stride = 1):

        shortcut = nn.Sequential(
            nn.Conv3d(inchannel,outchannel,1,stride),
            nn.BatchNorm3d(outchannel)
        )
        layers = []
        layers.append(ResidualBlock(inchannel,outchannel,stride,shortcut))
            
        for i in range(1,block_num):
            layers.append(ResidualBlock(outchannel,outchannel))

        return nn.Sequential(*layers)


    def forward(self, input):
        x = self.pre(input)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x) 
                                
        x = x.view(x.size(0),-1)
        
        return self.fc(x)
