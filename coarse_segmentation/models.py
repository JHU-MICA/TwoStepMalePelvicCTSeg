from typing import Tuple, Union
 
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange

from backbones.blocks import *
from backbones.coatnet_3d import CoAtNet3D
from transformer.net import ConvAxNet, PureAxNet

class Encoder(nn.Module):
    """
    Basic Encoder for UNet
    """

    def __init__(
        self,
        in_channels: int,
        feature_list: list,
        norm_name: Union[Tuple, str] = "instance",
        block_type: str = 'conv',
        subunits: int = 2,
        dropout: float = 0.25,
        **kwargs,
    ) -> None:

        super().__init__()

        self.downsample = nn.MaxPool3d(2,2) # downsample using max pooling
        self.num_layers = len(feature_list)

        # add first layer to encoder ops
        self.ops = nn.ModuleList([make_layer(in_channels,feature_list[0],block_type,norm_name,subunits=subunits,dropout=dropout,**kwargs)])

        # add subsequent layers
        for i in range(1,len(feature_list)):
            in_features = feature_list[i-1]
            out_features = feature_list[i]
            self.ops.append(make_layer(in_features,out_features,block_type,norm_name,subunits=subunits,dropout=dropout,**kwargs))
        
    def forward(self,x):
        # returns skip connections and bottom layers
        states = []

        for i,op in enumerate(self.ops):
            x = op(x) # run the block

            if i < self.num_layers - 1:
                states.append(x) # add to features if not bottom layer
                x = self.downsample(x)

        return x,states

class GenericUnet(nn.Module):
    """
    Basic UNet with Different Backbone Options
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Tuple[int, int, int],
        num_features: int = 16,
        num_layers: int = 4,
        norm_name: Union[Tuple, str] = "instance",
        encoder_block: str = 'conv',
        encoder_units: int = 2,
        dropout: float = 0.25,
        decoder_block: str = 'conv',
        decoder_units: int = 2,
        feature_list: list = None,
        **kwargs,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            num_features: number of features in the first layer of the network
            num_layers: number of layers in the model
            feature_size: dimension of network feature size.
            norm_name: feature normalization type and arguments.
            encoder_block: type of block to use for encoder (ex. conv, res, dense)
            dropout: faction of the input units to drop.
            feature_list: if a list of features per layer is given -> overwrites the default pattern of num_features/num_layers
            ex. feature_list = [16,32,64,128] is equivalent to num_features = 16, num_layers = 4
        """

        super().__init__()

        if not (0 <= dropout <= 1):
            raise AssertionError("dropout should be between 0 and 1.")

        if encoder_block == 'coat':
            if feature_list is not None:
                if len(feature_list) != 5:
                    raise AssertionError("invalid feature list. coatnet needs 5 layers")
            elif num_layers != 5:
                raise AssertionError("invalid number of layers. coatnet needs 5 layers")

        if feature_list is None:
            feature_list = [num_features * (2**i) for i in range(num_layers)]
            
        self.num_layers = len(feature_list)

        # coatnet encoder
        if encoder_block == 'coat':

            # default parameters for coatnet_0
            coat_blocks = kwargs.pop('coat_blocks',[2, 2, 3, 5, 2])
            coat_layers = kwargs.pop('coat_layers',['C', 'C', 'T', 'T'])
            coat_features = kwargs.pop('coat_features',[64,96,192,384,726])

            self.encoder = CoAtNet3D(img_size,in_channels,coat_blocks,coat_features,coat_layers,feature_list)

        elif encoder_block == 'axial':
            self.encoder = ConvAxNet(img_size,in_channels,feature_list,norm_name,encoder_units,dropout)

        elif encoder_block == 'pure_axial':
            self.encoder = PureAxNet(img_size,in_channels,feature_list,dropout)

        else:
            self.encoder = Encoder(in_channels,feature_list,norm_name,encoder_block,encoder_units,dropout,**kwargs)

        # get upsampling transpose conv parameters
        self.upsample = nn.ModuleList()
        for i in range(len(feature_list)-1,0,-1):
            self.upsample.append(nn.ConvTranspose3d(feature_list[i],feature_list[i-1],2,2))

        # add ops to decoder layer
        self.decoder_ops = nn.ModuleList()

        for i in range(len(feature_list)-1,0,-1):
            in_features = feature_list[i-1]*2
            out_features = feature_list[i-1]
            self.decoder_ops.append(make_layer(in_features,out_features,decoder_block,norm_name,subunits=decoder_units,dropout=dropout,**kwargs))
        

        # out layer
        self.final_conv = nn.Conv3d(out_features,out_channels,1)

    def forward(self, x):
        
        x, encoder_features = self.encoder(x)

        for i,op in enumerate(self.decoder_ops):
            skip = encoder_features[self.num_layers-2-i]
            x = self.upsample[i](x)
            x = torch.cat([skip,x],axis=1)
            x = op(x)

        x = self.final_conv(x)
            
        return x

# utility -> counts number of parameters in nn module
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    import monai

    #x = torch.randn((1,1,96,96,64))

    model = GenericUnet(
        1,
        7,
        (112,112,48),
        norm_name='INSTANCE',
        num_layers=4,
        encoder_block='conv',
        encoder_units=3,
    )

    print(count_params(model))