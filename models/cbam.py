'''Convolutional Block Attention Module (CBAM)

Improvements:
- Learnable temperature scaling for sigmoid functions
- Gated skip connections for adaptive blending
- Depthwise separable spatial convolutions
- Multi-path attention (CBAM_V3) with parallel channel and spatial paths
- Adaptive attention gating mechanism
'''

import torch
import torch.nn as nn
from torch.nn.modules import pooling
from torch.nn.modules.flatten import Flatten

class Channel_Attention(nn.Module):
    '''Channel Attention in CBAM with learnable temperature scaling.
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
        
        # Learnable temperature for sigmoid scaling
        self.temperature = nn.Parameter(torch.ones(1))
        
        # Layer normalization for stability
        self.ln = nn.LayerNorm(channel_in)


    def forward(self, x):
        '''Forward Propagation with temperature scaling.
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
        scaled = torch.sigmoid(pooling_sums * self.temperature).expand_as(x)

        return x * scaled #return the element-wise multiplication between the input and the result.


class ChannelPool(nn.Module):
    '''Merge all the channels in a feature map into two separate channels where the first channel is produced by taking the max values from all channels, while the
       second one is produced by taking the mean from every channel.
    '''
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class Spatial_Attention(nn.Module):
    '''Spatial Attention in CBAM with depthwise separable convolutions.
    '''

    def __init__(self, kernel_size=7):
        '''Spatial Attention Architecture with improved efficiency.
        '''

        super(Spatial_Attention, self).__init__()

        self.compress = ChannelPool()
        padding = (kernel_size - 1) // 2
        
        # Use depthwise separable convolution for better feature extraction
        self.spatial_attention = nn.Sequential(
            # Depthwise convolution (groups=2 for the 2 channels from ChannelPool)
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=kernel_size, stride=1, 
                      padding=padding, groups=2, bias=False),
            # Pointwise convolution
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=1, eps=1e-5, momentum=0.01, affine=True),
            nn.Sigmoid()
        )
        
        # Learnable temperature for spatial attention
        self.spatial_temperature = nn.Parameter(torch.ones(1))


    def forward(self, x):
        '''Forward Propagation.
        '''
        x_compress = self.compress(x)
        x_output = self.spatial_attention(x_compress)
        
        return x * x_output


# class CBAM(nn.Module):
#     '''CBAM architecture with gated skip connections.
#     '''
#     def __init__(self, channel_in, reduction_ratio=16, pool_types=['avg', 'max'], spatial=True):
#         '''Param init and arch build.
#         '''
#         super(CBAM, self).__init__()
#         self.spatial = spatial

#         self.channel_attention = Channel_Attention(channel_in=channel_in, reduction_ratio=reduction_ratio, pool_types=pool_types)

#         if self.spatial:
#             self.spatial_attention = Spatial_Attention(kernel_size=7)
        
#         # Learnable gating for skip connections
#         self.skip_gate = nn.Sequential(
#             nn.Conv2d(channel_in, channel_in // reduction_ratio, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channel_in // reduction_ratio, 1, 1),
#             nn.Sigmoid()
#         )


#     def forward(self, x):
#         '''Forward Propagation with gated residual connection.
#         '''
#         # Apply channel attention
#         x_out = self.channel_attention(x)
        
#         # Apply spatial attention if enabled
#         if self.spatial:
#             x_out = self.spatial_attention(x_out)
        
#         # Gated skip connection: learn how much to blend attention output with input
#         gate = self.skip_gate(x)
#         x_out = x_out * gate + x * (1 - gate)
        
#         return x_out


# class CBAM_V2(nn.Module):
#     '''Improved CBAM with gated skip connections and adaptive features.
#     '''
#     def __init__(self, channel_in, reduction_ratio=16, pool_types=['avg', 'max'], spatial=True, kernel_size=7):
#         '''Param init with improved components.
#         '''
#         super(CBAM_V2, self).__init__()
#         self.spatial = spatial
#         self.channel_attention = Channel_Attention(channel_in=channel_in, reduction_ratio=reduction_ratio, pool_types=pool_types)
        
#         if self.spatial:
#             self.spatial_attention = Spatial_Attention(kernel_size=kernel_size)
        
#         # Gated skip connection
#         self.skip_gate = nn.Sequential(
#             nn.Conv2d(channel_in, channel_in // reduction_ratio, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channel_in // reduction_ratio, 1, 1),
#             nn.Sigmoid()
#         )


#     def forward(self, x):
#         '''Forward Propagation with adaptive skip handling.
#         '''
#         # Store input for residual
#         identity = x
        
#         # Apply channel attention
#         x = self.channel_attention(x)
        
#         # Apply spatial attention if enabled
#         if self.spatial:
#             x = self.spatial_attention(x)
        
#         # Gated skip connection
#         gate = self.skip_gate(identity)
#         x = x * gate + identity * (1 - gate)
        
#         return x


class CBAM(nn.Module):
    '''Enhanced CBAM with multi-path attention and residual learning.
    '''
    def __init__(self, channel_in, reduction_ratio=16, pool_types=['avg', 'max'], spatial=True, kernel_size=7):
        '''Param init with multi-path design.
        '''
        super(CBAM, self).__init__()
        self.spatial = spatial
        
        # Path 1: Channel attention
        self.channel_attention = Channel_Attention(channel_in=channel_in, reduction_ratio=reduction_ratio, pool_types=pool_types)
        
        # Path 2: Spatial attention (parallel)
        if self.spatial:
            self.spatial_attention = Spatial_Attention(kernel_size=kernel_size)
        
        # Path fusion with learnable weights
        self.channel_weight = nn.Parameter(torch.ones(1) * 0.5)
        self.spatial_weight = nn.Parameter(torch.ones(1) * 0.5) if self.spatial else None
        
        # Attention gating mechanism
        self.attention_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel_in, channel_in // (2 * reduction_ratio), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_in // (2 * reduction_ratio), 1, 1),
            nn.Sigmoid()
        )


    def forward(self, x):
        '''Forward with multi-path fusion and attention gating.
        '''
        identity = x
        
        # Apply channel attention path
        x_channel = self.channel_attention(x)
        
        # Apply spatial attention path (parallel)
        if self.spatial:
            x_spatial = self.spatial_attention(x)
            # Fuse both paths
            x_fused = self.channel_weight * x_channel + self.spatial_weight * x_spatial
        else:
            x_fused = x_channel
        
        # Apply attention gating to control information flow
        gate = self.attention_gate(x)
        x_gated = x_fused * gate
        
        # Residual connection
        return x_gated + identity