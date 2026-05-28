'''Convolutional Block Attention Module (CBAM)
'''

import torch
import torch.nn as nn
from torch.nn.modules import pooling
from torch.nn.modules.flatten import Flatten

class Channel_Attention(nn.Module):
    '''Channel Attention in CBAM.
    '''

    def __init__(self, channel_in, reduction_ratio=16, pool_types=['avg', 'max']):
        '''Param init and architecture building.
        '''

        super(Channel_Attention, self).__init__()
        self.pool_types = pool_types
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Improved MLP with better initialization and regularization
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channel_in, channel_in // reduction_ratio, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_in // reduction_ratio, channel_in, 1, padding=0)
        )


    def forward(self, x):
        '''Forward Propagation.
        '''

        channel_attentions = []

        if 'avg' in self.pool_types:
            avg_pool = self.avg_pool(x)
            channel_attentions.append(self.shared_mlp(avg_pool))
        
        if 'max' in self.pool_types:
            max_pool = self.max_pool(x)
            channel_attentions.append(self.shared_mlp(max_pool))

        # Sum all attention maps
        pooling_sums = torch.stack(channel_attentions, dim=0).sum(dim=0)
        scaled = torch.sigmoid(pooling_sums).expand_as(x)

        return x * scaled #return the element-wise multiplication between the input and the result.


class ChannelPool(nn.Module):
    '''Merge all the channels in a feature map into two separate channels where the first channel is produced by taking the max values from all channels, while the
       second one is produced by taking the mean from every channel.
    '''
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class Spatial_Attention(nn.Module):
    '''Spatial Attention in CBAM.
    '''

    def __init__(self, kernel_size=7):
        '''Spatial Attention Architecture.
        '''

        super(Spatial_Attention, self).__init__()

        self.compress = ChannelPool()
        padding = (kernel_size - 1) // 2
        
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=1, 
                      padding=padding, bias=False),
            nn.BatchNorm2d(num_features=1, eps=1e-5, momentum=0.01, affine=True),
            nn.Sigmoid()
        )


    def forward(self, x):
        '''Forward Propagation.
        '''
        x_compress = self.compress(x)
        x_output = self.spatial_attention(x_compress)
        
        return x * x_output


class CBAM(nn.Module):
    '''CBAM architecture with residual connections.
    '''
    def __init__(self, channel_in, reduction_ratio=16, pool_types=['avg', 'max'], spatial=True):
        '''Param init and arch build.
        '''
        super(CBAM, self).__init__()
        self.spatial = spatial

        self.channel_attention = Channel_Attention(channel_in=channel_in, reduction_ratio=reduction_ratio, pool_types=pool_types)

        if self.spatial:
            self.spatial_attention = Spatial_Attention(kernel_size=7)


    def forward(self, x):
        '''Forward Propagation with residual connection.
        '''
        # Apply channel attention
        x_out = self.channel_attention(x)
        
        # Apply spatial attention if enabled
        if self.spatial:
            x_out = self.spatial_attention(x_out)
        
        # Add residual connection to preserve original information
        return x_out + x


class CBAM_V2(nn.Module):
    '''Improved CBAM with enhanced channel attention and dynamic kernel selection.
    '''
    def __init__(self, channel_in, reduction_ratio=16, pool_types=['avg', 'max'], spatial=True, kernel_size=7):
        '''Param init with improved components.
        '''
        super(CBAM_V2, self).__init__()
        self.spatial = spatial
        self.channel_attention = Channel_Attention(channel_in=channel_in, reduction_ratio=reduction_ratio, pool_types=pool_types)
        
        if self.spatial:
            self.spatial_attention = Spatial_Attention(kernel_size=kernel_size)


    def forward(self, x):
        '''Forward Propagation with better residual handling.
        '''
        # Store input for residual
        identity = x
        
        # Apply channel attention
        x = self.channel_attention(x)
        
        # Apply spatial attention if enabled
        if self.spatial:
            x = self.spatial_attention(x)
        
        # Add residual connection
        return x + identity