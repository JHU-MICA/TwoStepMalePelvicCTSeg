# implements attention mechanism #

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

class qkv_transform(nn.Conv1d):
    """Conv1d for qkv_transform"""

class RelativeAxialAttention(nn.Module):

    def __init__(
        self,
        img_shape,
        in_channels,
        num_heads = 8,
        dk = 8,
        dv = 16,
        axial_dim = 0,
        cross_attention = False,
        q_channels = None,
        gated = True
        ):

        ''' 
        3D Axial Attention 
        in_channels: input channels
        img_shape: size of input
        num_heads: number of attention heads
        dk: q/k embedding dimension
        dv: v embedding dimension
        axial_dimension: dimension to compute attention over
        gated: whether or not to use gated attention
        '''

        super(RelativeAxialAttention, self).__init__()
        self.in_planes = in_channels
        self.out_planes = num_heads * dv
        self.num_heads = num_heads
        self.dk = dk
        self.dv = dv
        self.kernel_size = img_shape[axial_dim]
        self.axial_dimension = axial_dim

        self.cross_attention = cross_attention
        self.q_channels = q_channels if q_channels is not None else in_channels
        self.gated = gated

        # Multi-head self attention
        self.kv_transform = qkv_transform(in_channels, num_heads * (dv + dk), kernel_size=1, stride=1, padding=0, bias=False)
        self.q_transform = qkv_transform(self.q_channels, num_heads * dk, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_kv = nn.BatchNorm1d(num_heads * (dv + dk))
        self.bn_q = nn.BatchNorm1d(num_heads * dk)
        self.bn_similarity = nn.BatchNorm2d(num_heads * 3)
        self.bn_output = nn.BatchNorm1d(num_heads * dv * 2)

        # Position embedding
        self.relative = nn.Parameter(torch.randn(dv + 2 * dk, self.kernel_size * 2 - 1), requires_grad=True)
        query_index = torch.arange(self.kernel_size).unsqueeze(0)
        key_index = torch.arange(self.kernel_size).unsqueeze(1)
        relative_index = key_index - query_index + self.kernel_size - 1
        self.register_buffer('flatten_index', relative_index.view(-1))

        # Gates
        self.f_qr = nn.Parameter(torch.tensor(0.1)) 
        self.f_kr = nn.Parameter(torch.tensor(0.1))
        self.f_sve = nn.Parameter(torch.tensor(0.1))
        self.f_sv = nn.Parameter(torch.tensor(1.0))


    def forward(self, x, y = None):
        # x: input of shape N C H W D
        # y: optional input of shape N C H W D (for cross attention)

        # Convert to # N {other dims} C {needed dim}
        if self.axial_dimension == 0:
            x = x.permute(0, 3, 4, 1, 2)

        elif self.axial_dimension == 1:
            x = x.permute(0, 2, 4, 1, 3)

        else:
            x = x.permute(0, 2, 3, 1, 4)

        if self.cross_attention:
            if self.axial_dimension == 0:
                y = y.permute(0, 3, 4, 1, 2)

            elif self.axial_dimension == 1:
                y = y.permute(0, 2, 4, 1, 3)

            else:
                y = y.permute(0, 2, 3, 1, 4)

            N, W, D, C, H = y.shape
            y = y.contiguous().view(N * W * D, C, H)
            

        N, W, D, C, H = x.shape
        x = x.contiguous().view(N * W * D, C, H)


        # Transformations: q,k,v N*W*D, HEADS, D_k/D_v, H
        kv = self.bn_kv(self.kv_transform(x))
        k, v = torch.split(kv.reshape(N * W * D, self.num_heads, self.dk + self.dv, H), [self.dk, self.dv], dim=2)

        if self.cross_attention:
            q = self.bn_q(self.q_transform(y)).reshape(N * W * D, self.num_heads, self.dk, H)
        
        else:
            q = self.bn_q(self.q_transform(x)).reshape(N * W * D, self.num_heads, self.dk, H)

        # Calculate position embedding
        all_embeddings = torch.index_select(self.relative, 1, self.flatten_index).view(self.dv + 2 * self.dk, self.kernel_size, self.kernel_size)
        q_embedding, k_embedding, v_embedding = torch.split(all_embeddings, [self.dk, self.dk, self.dv], dim=0)

        qr = torch.einsum('bgci,cij->bgij', q, q_embedding)
        kr = torch.einsum('bgci,cij->bgij', k, k_embedding).transpose(2, 3)
        qk = torch.einsum('bgci, bgcj->bgij', q, k)

        if self.gated:
            qr = torch.mul(qr, self.f_qr)
            kr = torch.mul(kr, self.f_kr)

        stacked_similarity = torch.cat([qk, qr, kr], dim=1)
        stacked_similarity = self.bn_similarity(stacked_similarity).view(N * W * D, 3, self.num_heads, H, H).sum(dim=1)

        similarity = F.softmax(stacked_similarity, dim=3)
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
        sve = torch.einsum('bgij,cij->bgci', similarity, v_embedding)

        if self.gated:
            sv = torch.mul(sv, self.f_sv)
            sve = torch.mul(sve, self.f_sve)

        stacked_output = torch.cat([sv, sve], dim=-1).view(N * W * D, self.out_planes * 2, H)
        output = self.bn_output(stacked_output).view(N, W, D, self.out_planes, 2, H).sum(dim=-2)

        if self.axial_dimension == 0:
            output = output.permute(0, 3, 4, 1, 2)

        elif self.axial_dimension == 1:
            output = output.permute(0, 3, 1, 4, 2)

        else:
            output = output.permute(0, 3, 1, 2, 4)

        return output

class AxialTransformer(nn.Module):

    def __init__(self,img_shape,in_channels,cross_channels,bn_channels=64,num_heads=8,dk=8,dv=16,dropout_rate=0.2,gated=True,**kwargs):

        '''
        transformer containing axial cross attention with relative axial position encoding
        input -> 1x1x1 conv -> x,y,z sequential axial attention -> 1x1x1 conv residual layer
        img_shape: shape of input image
        in_channels: input channels into layer
        cross_channels: channels of cross attention input
        bn_channels: bottleneck channels-> into transformer
        num_heads: number of heads for MHA
        dk: dimension of k and q per head
        dv: dimension of v per head
        dropout_rate: dropout to add to convolutions
        gated: whether or not to use gated transformer and gated residual (multiply the whole block by a constant factor) (defaults to True now)
        '''

        super(AxialTransformer, self).__init__()

        self.img_shape = img_shape
        self.in_channnels = in_channels
        self.out_channels = kwargs.pop('out_channels',cross_channels) # specify number of output channels if wanted
        self.gated = gated

        self.x_attention = RelativeAxialAttention(img_shape,bn_channels,num_heads,dk,dv,0,True,cross_channels,gated=gated)
        self.y_attention = RelativeAxialAttention(img_shape,dv*num_heads,num_heads,dk,dv,1,True,cross_channels,gated=gated)
        self.z_attention = RelativeAxialAttention(img_shape,dv*num_heads,num_heads,dk,dv,2,True,cross_channels,gated=gated)


        self.in_conv = nn.Sequential(
            nn.Conv3d(in_channels,bn_channels,1,1),
            nn.InstanceNorm3d(bn_channels),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )

        self.out_conv = nn.Sequential(
            nn.Conv3d(dv*num_heads,self.out_channels,1,1),
            nn.InstanceNorm3d(self.out_channels),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(p=dropout_rate),
        )

        self.trans_gate = nn.Parameter(torch.tensor(0.5))


    def forward(self,x,y):
        
        x_out = self.in_conv(x)

        x_out = self.x_attention(x_out,y)
        x_out = self.y_attention(x_out,y)
        x_out = self.z_attention(x_out,y)
        
        x_out = self.out_conv(x_out)

        if self.gated:
            return y + torch.mul(x_out, self.trans_gate)
        
        return x_out + y

class SelfAttention(nn.Module):

    def __init__(self,img_shape,in_channels,out_channels,num_heads=8,dk=8,dv=16,dropout_rate=0.2,**kwargs):

        '''
        transformer containing axial self attention with relative axial position encoding
        input -> 1x1x1 conv -> x,y,z sequential axial attention -> 1x1x1 conv residual layer
        img_shape: shape of input image
        in_channels: input channels into layer
        num_heads: number of heads for MHA
        dk: dimension of k and q per head
        dv: dimension of v per head
        dropout_rate: dropout to add to convolutions
        '''

        super(SelfAttention, self).__init__()

        self.img_shape = img_shape
        self.in_channnels = in_channels
        self.bn_channels = kwargs.pop('bn_channels',16) # bottleneck channels

        self.x_attention = RelativeAxialAttention(img_shape,self.bn_channels,num_heads,dk,dv,0)
        self.y_attention = RelativeAxialAttention(img_shape,dv*num_heads,num_heads,dk,dv,1)
        self.z_attention = RelativeAxialAttention(img_shape,dv*num_heads,num_heads,dk,dv,2)


        self.in_conv = nn.Sequential(
            nn.Conv3d(in_channels,self.bn_channels,1,1),
            nn.InstanceNorm3d(self.bn_channels),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )

        self.out_conv = nn.Sequential(
            nn.Conv3d(dv*num_heads,out_channels,1,1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )

        self.skip_conv = nn.Sequential(
            nn.Conv3d(in_channels,out_channels,1,1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )


    def forward(self,x_in):
        
        
        x_out = self.in_conv(x_in)

        x_out = self.x_attention(x_out)
        x_out = self.y_attention(x_out)
        x_out = self.z_attention(x_out)
        
        x_out = self.out_conv(x_out)
        x_in = self.skip_conv(x_in)
        
        return x_in + x_out

if __name__ == '__main__':

    net = AxialTransformer((24,24,24),32,32)
    x = torch.rand((2,32,24,24,24))
    y = torch.rand((2,32,24,24,24))

    print(net(x,y).shape)
