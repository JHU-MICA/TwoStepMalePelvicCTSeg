# this file contains the final conv/attention model #

from typing import Tuple, Union, Sequence

import torch
import numpy as np
import torch.nn as nn
import monai.networks.blocks as mn

from .axial_transformer import AxialTransformer, SelfAttention

class ConvAxNet(nn.Module):

    def __init__(
        self,
        img_shape: Sequence,
        in_channels: int,
        feature_list: list,
        norm_name: Union[Tuple, str] = "instance",
        subunits: int = 1, 
        dropout: float = 0.25,
        act = ('LEAKYRELU',{"negative_slope": 0.1}),
        **kwargs,
    ) -> None:

        super(ConvAxNet, self).__init__()

        self.img_shape = np.array(img_shape)
        self.in_channels = in_channels
        self.feature_list = feature_list
        self.num_layers = len(feature_list)
        self.norm = norm_name
        self.subunits = subunits
        self.dropout = dropout
        self.act = act

        self.downsample = nn.MaxPool3d(2,2)

        self.ops = nn.ModuleList()
        
        self.ops.append(self._make_layer(img_shape,in_channels,feature_list[0],add_transformer=False))

        for i in range(1,len(feature_list)):
            in_features = feature_list[i-1]
            out_features = feature_list[i]
            input_shape = self.img_shape // (2 ** i)

            self.ops.append(self._make_layer(input_shape,in_features,out_features))
    
    def _make_layer(self,input_shape,in_features,out_features,num_heads=8,dk=8,dv=16,add_transformer=True):

        convs = nn.ModuleList(
            [mn.Convolution(3,in_features,out_features,1,3,"NDA",self.act,self.norm,self.dropout)]
        )

        for _ in range(1,self.subunits):
            convs.append(
                mn.Convolution(3,out_features,out_features,1,3,"NDA",self.act,self.norm,self.dropout)
            )

        convs = nn.Sequential(*convs)

        if add_transformer:
            trans = AxialTransformer(input_shape,in_features,out_features,out_features,num_heads,dk,dv,self.dropout)
            return nn.ModuleList([convs,trans])

        return convs

    def forward(self,x):

        # returns skip connections and bottom layers
        states = []

        x = self.ops[0](x)
        states.append(x)
        x = self.downsample(x)

        for i,(conv,trans) in enumerate(self.ops[1:]):
            
            x = trans(x,conv(x)) # run convolution then transformer

            if i < self.num_layers - 2:

                states.append(x) # add to features if not bottom layer
                x = self.downsample(x)

        return x,states

class PureAxNet(nn.Module):
    # pure axial attention transformer
    def __init__(
        self,
        img_shape: Sequence,
        in_channels: int,
        feature_list: list,
        dropout: float = 0.25,
        **kwargs,
    ) -> None:

        super(PureAxNet, self).__init__()

        self.img_shape = np.array(img_shape)
        self.in_channels = in_channels
        self.feature_list = feature_list
        self.num_layers = len(feature_list)
        self.dropout = dropout

        self.downsample = nn.MaxPool3d(2,2)

        self.ops = nn.ModuleList()
        
        # first layer: conv
        self.ops.append(
            nn.Sequential(
                mn.Convolution(3,in_channels,feature_list[0],1,3,"NDA","RELU","INSTANCE",self.dropout),
                mn.Convolution(3,feature_list[0],feature_list[0],1,3,"NDA","RELU","INSTANCE",self.dropout),
            )
        )

        for i in range(1,len(feature_list)):
            in_features = feature_list[i-1]
            out_features = feature_list[i]
            input_shape = self.img_shape // (2 ** i)

            self.ops.append(SelfAttention(input_shape,in_features,out_features,8,8,16,dropout))
    
    def forward(self,x):

        # returns skip connections and bottom layers
        states = []

        x = self.ops[0](x)
        states.append(x)
        x = self.downsample(x)

        for i, op in enumerate(self.ops[1:]):
            
            x = op(x) # run convolution then transformer

            if i < self.num_layers - 2:

                states.append(x) # add to features if not bottom layer
                x = self.downsample(x)

        return x,states

## NEW: AxInceptionNet: inception block as cross ##

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

        
        self.bn = nn.Conv3d(out_channels*4,out_channels,1) 

        # squeeze excitation
        self.se = mn.ChannelSELayer(spatial_dims,out_channels,r=4)

    def forward(self,x):
        out_xy = self.xy_conv(x)
        out_xz = self.xz_conv(x)
        out_yz = self.yz_conv(x)
        out_xyz = self.xyz_conv(x)

        x = torch.cat([out_xy,out_xz,out_yz,out_xyz],1)
        
        x = self.bn(x)

        return self.se(x)
        
class AxInceptionNet(nn.Module):

    def __init__(
        self,
        img_shape: Sequence,
        in_channels: int,
        feature_list: list,
        norm_name: Union[Tuple, str] = "instance",
        subunits: int = 1, 
        dropout: float = 0.25,
        act = ('LEAKYRELU',{"negative_slope": 0.1}),
        **kwargs,
    ) -> None:

        super(AxInceptionNet, self).__init__()

        self.img_shape = np.array(img_shape)
        self.in_channels = in_channels
        self.feature_list = feature_list
        self.num_layers = len(feature_list)
        self.norm = norm_name
        self.subunits = subunits
        self.dropout = dropout
        self.act = act

        self.downsample = nn.MaxPool3d(2,2)

        self.ops = nn.ModuleList()
        
        self.ops.append(self._make_layer(img_shape,in_channels,feature_list[0],add_transformer=False))

        for i in range(1,len(feature_list)):
            in_features = feature_list[i-1]
            out_features = feature_list[i]
            input_shape = self.img_shape // (2 ** i)

            self.ops.append(self._make_layer(input_shape,in_features,out_features))
    
    def _make_layer(self,input_shape,in_features,out_features,num_heads=8,dk=8,dv=16,add_transformer=True):

        convs = nn.ModuleList(
            [InceptionBlock(in_features,out_features,self.act,self.norm,self.dropout)]
        )

        for _ in range(1,self.subunits):
            convs.append(
                InceptionBlock(out_features,out_features,self.act,self.norm,self.dropout)
            )

        convs = nn.Sequential(*convs)

        if add_transformer:
            trans = AxialTransformer(input_shape,in_features,out_features,out_features,num_heads,dk,dv,self.dropout)
            return nn.ModuleList([convs,trans])

        return convs

    def forward(self,x):

        # returns skip connections and bottom layers
        states = []

        x = self.ops[0](x)
        states.append(x)
        x = self.downsample(x)

        for i,(conv,trans) in enumerate(self.ops[1:]):
            
            x = trans(x,conv(x)) # run convolution then transformer

            if i < self.num_layers - 2:

                states.append(x) # add to features if not bottom layer
                x = self.downsample(x)

        return x,states


# utility -> counts number of parameters in nn module
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':

    img_shape = (64,64,64)
    input = torch.randn((1,1,64,64,64))
    net = PureAxNet(img_shape,1,[16,32,64,128])

    print(net(input)[0].shape)
    print(count_params(net))