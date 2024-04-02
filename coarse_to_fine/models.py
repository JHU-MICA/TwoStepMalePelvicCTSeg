from typing import Tuple, Union
 
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange

import monai.networks.blocks as mn
from monai.networks.nets import ViT
from transformer.net import ConvAxNet, PureAxNet

from backbones.blocks import *
from backbones.coatnet_3d import CoAtNet3D

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

class DeepLabEncoder(nn.Module):
    """
    Basic Encoder for DeepLab -> after layer 3, use dilated convs
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

        dilations = [1 if i < 3 else 2 ** (i-3) for i in range(self.num_layers)]

        # add first layer to encoder ops -> strided convolution
        self.ops = nn.ModuleList([
            make_layer(in_channels,feature_list[0],block_type,norm_name,subunits=subunits,dropout=dropout,**kwargs), 
        ])

        # add subsequent layers
        for i in range(1,len(feature_list)):
            in_features = feature_list[i-1]
            out_features = feature_list[i]
            dilation = dilations[i]

            self.ops.append(
                make_layer(in_features,out_features,block_type,norm_name,subunits=subunits,dropout=dropout,dilation=dilation,**kwargs)
            )
        
    def forward(self,x):
        # returns skip connections and bottom layers
        states = []

        for i,op in enumerate(self.ops):
            x = op(x) # run the block

            if i < self.num_layers - 1:
                states.append(x) # add to features if not bottom layer

                if i < 4:
                    x = self.downsample(x)

        return x,states

class DeepLabv3(nn.Module):
    """
    Basic DeepLabv3-like architecture with Different Backbone Options
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
        aspp_features: float = None,
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
            aspp_features: features for output of aspp
            feature_list: if a list of features per layer is given -> overwrites the default pattern of num_features/num_layers
            ex. feature_list = [16,32,64,128] is equivalent to num_features = 16, num_layers = 4
        """

        super().__init__()

        if not (0 <= dropout <= 1):
            raise AssertionError("dropout should be between 0 and 1.")

        if not (num_layers >= 5):
            raise AssertionError("model needs at least 5 layers")

        if encoder_block == 'coat':
            if feature_list is not None:
                if len(feature_list) != 5:
                    raise AssertionError("invalid feature list. coatnet needs 5 layers")
            elif num_layers != 5:
                raise AssertionError("invalid number of layers. coatnet needs 5 layers")

        if feature_list is None:
            feature_list = [num_features * (2**i) for i in range(num_layers)]
        
        # make the aspp features the same as the encoder features at the same depth
        if aspp_features is None:
            aspp_features = feature_list[2]
            
        self.num_layers = len(feature_list)
        self.encoder_state = 2 # depth to use for skip connection (usually 2, but 6 for vit)

        # coatnet encoder
        if encoder_block == 'coat':

            # default parameters for coatnet_0
            coat_blocks = kwargs.pop('coat_blocks',[2, 2, 3, 5, 2])
            coat_layers = kwargs.pop('coat_layers',['C', 'C', 'T', 'T'])
            coat_features = kwargs.pop('coat_features',[64,96,192,384,726])

            self.encoder = CoAtNet3D(img_size,in_channels,coat_blocks,coat_features,coat_layers,feature_list)

        # VIT encoder
        elif encoder_block == 'vit':
            
            # default parameters for VIT
            vit_patch_size = kwargs.pop('vit_patch_size',(img_size[0]//4,img_size[1]//4,img_size[2]//4)) # hidden size
            vit_hidden_size = kwargs.pop('vit_hidden_size',768) # hidden size
            vit_mlp_dim = kwargs.pop('vit_mlp_dim',3072) # mlp dims
            vit_num_layers = kwargs.pop('vit_num_layers',12) # number of layers
            vit_heads = kwargs.pop('vit_heads',8) # number of attention heads
            vit_dropout = kwargs.pop('vit_dropout',dropout) # dropout

            self.encoder = ViT(
                in_channels,
                img_size,
                vit_patch_size,
                vit_hidden_size,
                vit_mlp_dim,
                vit_num_layers,
                vit_heads,
                pos_embed = 'conv',
                num_classes = out_channels,
                dropout_rate = vit_dropout,
            )

            self.encoder_state = vit_num_layers // 2 # use the middle layer for the skip connection

        else:
            self.encoder = DeepLabEncoder(in_channels,feature_list,norm_name,encoder_block,encoder_units,dropout,**kwargs)

        upsampler = nn.Upsample(scale_factor=4,mode='trilinear',align_corners=True)
        
        # ASPP layer -> aspp and upsampling (as 2 transposed convolutions)
        aspp_dilations = kwargs.pop('aspp_dilations',[1, 2, 4, 6])
        self.aspp = nn.Sequential(
            mn.SimpleASPP(3,feature_list[-1],aspp_features,norm_type=norm_name,dilations=aspp_dilations),
            basic_conv_block(
                aspp_features*4,
                aspp_features,
                kernel_size=1,
                norm=norm_name,
                dropout=dropout),
            upsampler,
        )

        # encoder operations
        self.decoder_ops = nn.ModuleList([
            basic_conv_block(
                feature_list[2],
                aspp_features,
                kernel_size=1,
                norm=norm_name,
                dropout=dropout
                ),
            make_layer(aspp_features*2,32,norm=norm_name,dropout=dropout,subunits=2), # convolutional layer
            upsampler, #upsampling
            nn.Conv3d(32,out_channels,1), # final convolution
        ])

        # ViT needs some reshaping
        if encoder_block == 'vit':
            reshape = Rearrange(
                'b (i j k) f -> b f i j k',
                i = img_size[0] // vit_patch_size[0],
                j = img_size[1] // vit_patch_size[1],
                k = img_size[2] // vit_patch_size[2],
            )

            self.aspp = nn.Sequential(
                reshape,
                mn.SimpleASPP(3,vit_hidden_size,aspp_features,norm_type=norm_name,dilations=aspp_dilations),
                basic_conv_block(
                    aspp_features*4,
                    aspp_features,
                    kernel_size=1,
                    norm=norm_name,
                    dropout=dropout
                ),
                upsampler,
            )

            self.decoder_ops[0] = nn.Sequential(
                reshape,
                make_layer(
                    vit_hidden_size,
                    aspp_features,
                    kernel_size=3,
                    strides=2,
                    norm=norm_name,
                    dropout=dropout,
                    is_transposed=True,
                ),
                basic_conv_block(
                    feature_list[2],
                    aspp_features,
                    kernel_size=1,
                    norm=norm_name,
                    dropout=dropout
                )
            )            

    def forward(self, x):
        
        x, states = self.encoder(x)
        
        x = self.aspp(x) # ASPP (with upsampling)
        
        skip = self.decoder_ops[0](states[self.encoder_state]) # skip connection -> apply 1x1x1 conv

        # concatenate feature maps and perform convolutions
        x = torch.cat([x,skip],dim=1)
        x = self.decoder_ops[1](x)

        # upsample and perform final convolution
        x = self.decoder_ops[2](x)
        x = self.decoder_ops[3](x)

        return x

class AutoEncoder(nn.Module):
    """
    Basic Autoencoder with aspp bottleneck
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_features: int = 16,
        num_layers: int = 3,
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
            num_layers: number of layers in the model (NOT including the bottleneck) -> bottleneck as one downsample/dense block/upsample sequence
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


        if feature_list is None:
            feature_list = [num_features * (2**i) for i in range(num_layers)]
            
        self.num_layers = len(feature_list)

        self.encoder = []

        for i in range(len(feature_list)):

            in_features = in_channels if i == 0 else feature_list[i-1]
            out_features = feature_list[i]

            self.encoder.append(
                nn.Sequential(
                    make_layer(in_features,out_features,encoder_block,norm_name,encoder_units,dropout,**kwargs),
                    nn.MaxPool3d(2,2),
                )
            )

        self.encoder = nn.Sequential(*self.encoder)

        self.bottleneck = nn.Sequential(
            mn.SimpleASPP(3,feature_list[-1],feature_list[-1]*2,norm_type=norm_name,dilations=(1,2,4,6)),
            basic_conv_block(
                feature_list[-1]*8,
                feature_list[-1]*2,
                kernel_size=1,
                norm=norm_name,
                dropout=dropout),
        )

        self.decoder = []

        for i in range(len(feature_list)):
            
            in_features = feature_list[-1] * 2 if i == 0 else feature_list[-i]
            out_features = feature_list[-i-1]

            self.decoder.append(
                nn.Sequential(
                    nn.ConvTranspose3d(in_features,out_features,2,2),
                    make_layer(out_features,out_features,decoder_block,norm_name,decoder_units,dropout,**kwargs)
                )
            )

        self.decoder = nn.Sequential(*self.decoder)

        # out layer
        self.final_conv = nn.Conv3d(out_features,out_channels,1)

    def forward(self, x):
        
        x = self.encoder(x)

        x = self.bottleneck(x)

        x = self.decoder(x)

        x = self.final_conv(x)
            
        return torch.sigmoid(x)

# utility -> counts number of parameters in nn module
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# utility -> initialize weights in module
def he_normal(m):
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
        torch.nn.init.kaiming_uniform_(m.weight)

if __name__ == '__main__':
    import monai
    from config import COARSE_CONFIG


    model = GenericUnet(**COARSE_CONFIG)
    
    print(count_params(model))