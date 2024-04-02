import torch.nn as nn
import torch

import monai.networks.blocks as mn
from typing import Tuple, Union

def basic_conv_block(
    in_channels,
    out_channels,
    strides = 1,
    kernel_size = 3,
    act = ('LEAKYRELU',{"negative_slope": 0.1}),
    norm = 'BATCH',
    spatial_dims = 3,
    adn_ordering = 'NDA', 
    dropout = 0., 
    dilation = 1,
    groups = 1,
    bias = True,
    is_transposed = False,
    padding = None, 
    ):
    # implements a convolutional block (just wraps the monai built-in function -> has a lot of functionality)

    return mn.Convolution(
        spatial_dims,
        in_channels,
        out_channels,
        strides = strides,
        kernel_size = kernel_size,
        act = act,
        norm = norm,
        adn_ordering = adn_ordering, 
        dropout = dropout, 
        dilation = dilation,
        groups = groups,
        bias = bias,
        is_transposed = is_transposed,
        padding = padding)

def residual_block(
    in_channels,
    out_channels,
    subunits = 2,
    strides = 1,
    kernel_size = 3,
    act = ('LEAKYRELU',{"negative_slope": 0.1}),
    norm = 'BATCH',
    spatial_dims = 3,
    adn_ordering = 'NDA', 
    dropout = 0., 
    dilation = 1,
    bias = True,
    padding = None, 
    last_conv_only=False
    ):
    # implements a convolutional block (just wraps the monai built-in function -> has a lot of functionality)

    return mn.ResidualUnit(
        spatial_dims,
        in_channels,
        out_channels,
        strides = strides,
        kernel_size = kernel_size,
        act = act,
        norm = norm,
        adn_ordering = adn_ordering, 
        dropout = dropout, 
        dilation = dilation,
        bias = bias,
        padding = padding,
        subunits = subunits,
        last_conv_only = last_conv_only,
    )

class DenseBlock(nn.Module):
    """
    Basic dense block with bottleneck layers
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        subunits = 2,
        strides = 1,
        kernel_size = 3,
        act = ('LEAKYRELU',{"negative_slope": 0.1}),
        norm = 'BATCH',
        spatial_dims = 3,
        adn_ordering = 'NDA', 
        dropout = 0., 
        dilation = 1,
        groups = 1,
        bias = True,
        is_transposed = False,
        padding = None,
        bn_factor = 2 # for bottleneck: mulitied by out_channels for layers in bottleneck conv
    ) -> None:

        super().__init__()

        self.ops = nn.ModuleList()
        self.subunits = subunits 

        # add transition layer if needed
        if in_channels != out_channels:
            self.ops.append(basic_conv_block(in_channels,out_channels,strides,1,act,norm,spatial_dims,adn_ordering,dropout))
        else:
            self.ops.append(nn.Identity())

        in_channels = out_channels

        # add ops
        for _ in range(subunits):
            
            self.ops.append(nn.Sequential(
                basic_conv_block(
                    in_channels,out_channels*bn_factor,strides,1,act,norm,spatial_dims,adn_ordering,dropout),
                basic_conv_block(
                    out_channels*bn_factor,out_channels,strides,kernel_size,
                    act,norm,spatial_dims,adn_ordering,dropout,
                    dilation,groups,bias,is_transposed,padding)
            ))
            
            in_channels += out_channels

        
    def forward(self,x_in):
        
        for i,op in enumerate(self.ops):
            x = op(x_in)
            
            # concatenate feature maps
            if i == 0: # first layer is just transition
                x_in = x
            else:
                x_in = torch.cat([x_in,x],1)
        
        return x

class InceptionBlock(nn.Module):
    """
    modified inception block: concatenates intraslice and interslice convs
    added squeeze and excitation block (for interchannel dependencies)
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        act = ('LEAKYRELU',{"negative_slope": 0.1}),
        norm = 'BATCH',
        dropout = 0., 
        spatial_dims = 3,
        **kwargs,
    ):
        super().__init__()

        self.xy_conv = basic_conv_block(in_channels,out_channels,kernel_size=(3,3,1),act=act,norm=norm,dropout=dropout,spatial_dims=spatial_dims,**kwargs)
        self.xz_conv = basic_conv_block(in_channels,out_channels,kernel_size=(3,1,3),act=act,norm=norm,dropout=dropout,spatial_dims=spatial_dims,**kwargs)
        self.yz_conv = basic_conv_block(in_channels,out_channels,kernel_size=(1,3,3),act=act,norm=norm,dropout=dropout,spatial_dims=spatial_dims,**kwargs)
        self.xyz_conv = basic_conv_block(in_channels,out_channels,kernel_size=3,act=act,norm=norm,dropout=dropout,spatial_dims=spatial_dims,**kwargs)

        # bottleneck
        self.bn = nn.Conv3d(out_channels*3,out_channels,1) 

        # squeeze excitation
        self.se = mn.ChannelSELayer(spatial_dims,out_channels,r=4, add_residual=False)

    def forward(self,x):
        out_xy = self.xy_conv(x)
        out_xz = self.xz_conv(x)
        out_yz = self.yz_conv(x)
        out_xyz = self.xyz_conv(x)

        
        x = torch.cat([out_xy,out_xz,out_yz],1) 
        
        # add the intraslice with interslice
        x = torch.nn.LeakyReLU(negative_slope=0.1)(self.bn(x) + out_xyz)

        return self.se(x)

def make_layer(
    in_features: int, 
    out_features: int,
    block_type: str = 'conv',
    norm: Union[Tuple, str] = "instance",
    subunits: int = 2, 
    dropout: float = 0.25,
    **kwargs):
    '''
    makes  layer for unet (num units repeats of block)
    in_features: input feature channels for layer
    out_features: output feature channels for layer
    block_type: type of block to use (ex. conv, res, dense)
    subunits: number of units in the layer
    '''

    # dictionary containing block type + block function
    block_dict = {
        'conv': basic_conv_block,
        'res': residual_block,
        'dense': DenseBlock,
        'inception': InceptionBlock
    }

    block_fn = block_dict[block_type]

    # if it is a premade block of multiple convs
    if block_type in ['res','dense']:
        return block_fn(in_features,out_features,subunits=subunits,norm=norm,dropout=dropout,**kwargs)

    # else if it is just a single conv block
    else:
        layers = [block_fn(in_features,out_features,norm=norm,dropout=dropout,**kwargs)]

        for _ in range(1,subunits):
            layers.append(block_fn(out_features,out_features,norm=norm,dropout=dropout,**kwargs))

    return nn.Sequential(*layers)

if __name__ == '__main__':

    block = make_layer(1,8,padding=1,dropout=0.1,subunits=3,block_type='dense')
    x = torch.randn(size=(1,1,16,16,16))
    print(block,block(x).shape)