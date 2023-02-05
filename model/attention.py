import torch
import torch.nn as nn

'''
https://medium.com/analytics-vidhya/talented-mr-1x1-comprehensive-look-at-1x1-convolution-in-deep-learning-f6b355825578

An attention gate in U-Net works as a form of attention mechanism to dynamically weigh different feature maps based on their importance for the final prediction.

Typically, an attention gate consists of two main components: a gating mechanism and a weighting mechanism. The gating mechanism is responsible for deciding which feature maps are important, while the weighting mechanism assigns different weights to each feature map.

In the U-Net architecture, the attention gate is typically implemented by concatenating the feature maps from the contracting path with the feature maps from the expanding path and using a 1x1 convolution to reduce the number of channels. Then, a sigmoid activation is applied to the output of the 1x1 convolution to get the attention weights, which are multiplied element-wise with the feature maps. The weighted feature maps are then fed into the next layer of the network.

The attention gate allows the network to focus on relevant features for the final prediction and helps to improve the accuracy and stability of the U-Net model.

'''
class AttentionBlock(nn.Module):
    def __init__(self, f_g, f_l, f_int):
        super().__init__()
        
        self.w_g = nn.Sequential(
                                nn.Conv2d(f_g, f_int,
                                         kernel_size=1, stride=1,
                                         padding=0, bias=True),
                                nn.BatchNorm2d(f_int)
        )
        
        self.w_x = nn.Sequential(
                                nn.Conv2d(f_l, f_int,
                                         kernel_size=1, stride=1,
                                         padding=0, bias=True),
                                nn.BatchNorm2d(f_int)
        )
        
        self.psi = nn.Sequential(
                                nn.Conv2d(f_int, 1,
                                         kernel_size=1, stride=1,
                                         padding=0,  bias=True),
                                nn.BatchNorm2d(1),
                                nn.Sigmoid(),
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.w_g(g)
        x1 = self.w_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
        
        return psi*x
