import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init
from torch.nn.modules.utils import _triple
import math
from loguru import logger

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            # print("classname: {}, init1".format(classname))
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            # print("classname: {}, init2".format(classname))
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    # print('initialize network with {}'.format(init_type))
    net.apply(init_func)  # apply the initialization function <init_func>

def init_net(net,init_type='kaiming', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net

class SpatioTemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(SpatioTemporalConv, self).__init__()

		
        # if ints are entered, convert them to iterables, 1 -> [1, 1, 1]
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)

        # decomposing the parameters into spatial and temporal components by
        # masking out the values with the defaults on the axis that
        # won't be convolved over. This is necessary to avoid unintentional
        # behavior such as padding being added twice
        spatial_kernel_size =  [1, kernel_size[1], kernel_size[2]]
        spatial_stride =  [1, stride[1], stride[2]]
        spatial_padding =  [0, padding[1], padding[2]]

        temporal_kernel_size = [kernel_size[0], 1, 1]
        temporal_stride =  [stride[0], 1, 1]
        temporal_padding =  [padding[0], 0, 0]

        # compute the number of intermediary channels (M) using formula 
        # from the paper section 3.5
        intermed_channels = int(math.floor((kernel_size[0] * kernel_size[1] * kernel_size[2] * in_channels * out_channels)/(kernel_size[1]* kernel_size[2] * in_channels + kernel_size[0] * out_channels)))
        
        # self-definition
        #intermed_channels = int((in_channels+intermed_channels)/2)

        # the spatial conv is effectively a 2D conv due to the 
        # spatial_kernel_size, followed by batch_norm and ReLU
        self.spatial_conv = nn.Conv3d(in_channels, intermed_channels, spatial_kernel_size,
                                    stride=spatial_stride, padding=spatial_padding, bias=bias)
        self.bn = nn.BatchNorm3d(intermed_channels)
        self.relu = nn.ReLU()   ##   nn.Tanh()   or   nn.ReLU(inplace=True)


        # the temporal conv is effectively a 1D conv, but has batch norm 
        # and ReLU added inside the model constructor, not here. This is an 
        # intentional design choice, to allow this module to externally act 
        # identical to a standard Conv3D, so it can be reused easily in any 
        # other codebase
        self.temporal_conv = nn.Conv3d(intermed_channels, out_channels, temporal_kernel_size, 
                                    stride=temporal_stride, padding=temporal_padding, bias=bias)

    def forward(self, x):
        x = self.relu(self.bn(self.spatial_conv(x)))  
        x = self.temporal_conv(x)                      
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels=128, out_channels=1):
        super(Decoder, self).__init__()
        frames = 64
        self.DConv1 = nn.Sequential(
            nn.ConvTranspose3d(in_channels, in_channels, kernel_size=(4,1,1), stride=(2,1,1), padding=(1,0,0), bias=False),
            nn.InstanceNorm3d(in_channels, affine=True, track_running_stats=True),
            nn.LeakyReLU(inplace=True)
        )
        self.DConv2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels, in_channels, kernel_size=(4,1,1), stride=(2,1,1), padding=(1,0,0), bias=False),
            nn.InstanceNorm3d(in_channels, affine=True, track_running_stats=True),
            nn.LeakyReLU(inplace=True)
        )
        self.Conv = nn.Conv3d(in_channels, out_channels, [1,1,1],stride=1, padding=0)
        self.poolspa = nn.AdaptiveAvgPool3d((frames,1,1))


    def forward(self, x):
        """
            inputs :
                x : latent-space feature [B, C:128, T:16, W:8, H:8]
            returns :
                x : rppg signal [B, T:64]
        """

        x = self.DConv1(x)                      # [128, 16, 8, 8] -> [64, 32, 8, 8]
        x = self.DConv2(x)                      # [64, 32, 8, 8] -> [32, 64, 8, 8]

        x = self.poolspa(x)                     # [32, 64, 8, 8] -> [32, 64, 1, 1]

        x = self.Conv(x).squeeze(1).squeeze(-1).squeeze(-1)    # [32, 64, 1, 1] -> [1, 64, 1, 1] -> [ , 64]

        return x

class Encoder128(nn.Module):
    def __init__(self):
        super(Encoder128, self).__init__()

        self.Conv1 = nn.Sequential(
            nn.Conv3d(3, 16, [1,5,5], stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        self.STConv1 = nn.Sequential(
            SpatioTemporalConv(16, 32, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.STConv2 = nn.Sequential(
            SpatioTemporalConv(32, 32, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.STConv3 = nn.Sequential(
            SpatioTemporalConv(32, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.STConv4 = nn.Sequential(
            SpatioTemporalConv(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.AvgpoolSpa = nn.AvgPool3d((1, 2, 2), stride=(1, 2, 2))

        # self.t1 = nn.Conv3d(3, 16, [1,5,5], stride=1, padding=[0,2,2])
        # self.t2 = nn.BatchNorm3d(16)
        # self.t3 = nn.ReLU(inplace=True)

    def forward(self, x):
        """
            inputs :
                x : input feature [B, C:3, T:64, W:128, H:128]
            returns :
                x : latent-space feature [B, C:64, T:64, W:32, H:32]
        """
        # print("input shape: {}".format(x.shape))
        # print("x0.0: {}".format(x))
        # x = self.t1(x)
        # print("x0.1: {}".format(x))
        # x = self.t2(x)
        # print("x0.2: {}".format(x))
        # x = self.t3(x)
        # print("x0.3: {}".format(x))

        x = self.Conv1(x)                   # [3, 64, 128, 128] -> [16, 64, 128, 128]

        x = self.AvgpoolSpa(x)              # [16, 64, 128, 128] -> [16, 64, 64, 64]
        # print("x1: {}".format(x))

        x = self.STConv1(x)                 # [16, 64, 64, 64] -> [32, 64, 64, 64]
        x = self.STConv2(x)                 # [32, 64, 64, 64] -> [32, 64, 64, 64]
        # print("x2: {}".format(x))

        x = self.AvgpoolSpa(x)              # [32, 64, 64, 64] -> [32, 64, 32, 32]
        # print("x3: {}".format(x))

        x = self.STConv3(x)                 # [32, 64, 32, 32] -> [64, 64, 32, 32]
        x = self.STConv4(x)                 # [64, 64, 32, 32] -> [64, 64, 32, 32]
        # print("x4: {}".format(x))

        return x

class Block(nn.Module):
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(Block, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.leakyrelu = nn.LeakyReLU(inplace=True)

        self.conv2 = nn.Conv3d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leakyrelu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.leakyrelu(out)

        return out

class Encoder32(nn.Module):
    def __init__(self):
        super(Encoder32, self).__init__()

        self.Conv1 = nn.Sequential(                                 # [3, 64, 32, 32] -> [64, 64, 32, 32]
            nn.Conv3d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(inplace=True)
        )


        downsample = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm3d(128)
        )
        self.CB1 = Block(64, 128, 1, downsample)
        self.down1 = nn.Conv3d(128, 128, kernel_size=2, stride=2)

        self.CB2 = Block(128, 128)
        self.down2 = nn.Conv3d(128, 128, kernel_size=2, stride=2)

        self.CB3 = Block(128, 128)
        # self.down3 = nn.Conv3d(128, 128, kernel_size=2, stride=2)
        

    def forward(self, x):
        """
            inputs :
                x : input feature [B, C:3, T:64, W:32, H:32]
            returns :
                x : latent-space feature [B, C:128, T:16, W:8, H:8]
        """
        # print("In Encoder32 forward(), x: {}".format(x.shape))
        x = self.Conv1(x)                   # [3, 64, 32, 32] -> [64, 64, 32, 32]
        # print("x1: {}".format(x))
        # print("x: {}".format(x.shape))
        out = self.CB1(x)                   # [64, 64, 32, 32] -> [128, 64, 32, 32]
        # print("CB1: {}".format(out.shape))
        out = self.down1(out)               # [128, 64, 32, 32] -> [128, 32, 16, 16]
        # print("CB1 down1: {}".format(out.shape))

        out = self.CB2(out)                   # [128, 32, 16, 16] -> [128, 32, 16, 16]
        # print("CB2: {}".format(out.shape))
        out = self.down2(out)               # [128, 32, 16, 16] -> [128, 16, 8, 8]
        # print("CB2 down2: {}".format(out.shape))

        out = self.CB3(out)                   # [128, 16, 8, 8] -> [128, 16, 8, 8]
        # out = self.down3(out)

        return out

class Classifier(nn.Module):
    def __init__(self, in_channels=128):
        super(Classifier, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, 1024, [3,3,3], stride=1, padding=1),
            nn.BatchNorm3d(1024),
            nn.ReLU(inplace=True),
        )

        self.avgpooling = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.classifier = nn.Sequential(nn.Linear(1024, 512),
                                        nn.BatchNorm1d(512),
                                        nn.Dropout(p=0.3),
                                        nn.ReLU(),
                                        nn.Linear(512, 128),
                                        nn.BatchNorm1d(128),
                                        nn.Dropout(p=0.3),
                                        nn.ReLU(),
                                        nn.Linear(128, 2)
                                        )
                    

    def forward(self, x):
        """
            inputs :
                x : latent-space feature [B, C:128, T:16, W:8, H:8]
            returns :
                pred
        """
        # print("input: {}".format(x.shape))
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!! 1 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        x = self.conv(x)
        # print("after conv: {}".format(x.shape))
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!! 2 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        x = self.avgpooling(x)
        # print("after avgpooling: {}".format(x.shape))
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!! 3 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        x = x.view(x.size(0), -1)
        # print("after view: {}".format(x.shape))
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!! 4 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # feat = x
        pred = self.classifier(x)
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!! 5 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
         
        return pred 

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(4, 8, [1,5,5], stride=[1,2,2], padding=[0,2,2]),                 # [4,D,W,H] -> [8,D,W/2,H/2]
            nn.InstanceNorm3d(8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(8, 16, [3,3,3], stride=[2,4,4], padding=1),                      # [8,D,W/2,H/2] -> [16,D/2,W/8,H/8] [16,32,4,4]
            nn.InstanceNorm3d(16, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool3d((8, 2, 2))

        self.model = nn.Sequential(
            nn.Linear(512, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        """
            inputs :
                x : input feature [B, C:4, T:64, W:32, H:32]
            returns :
                validity
        """
        x = self.conv(x)                                    # [B, C:4, T:64, W:32, H:32] -> [B, 16, 32, 4, 4]
        x = self.pool(x)                                    # [B, 16, 32, 4, 4] -> [B, 16, 8, 2, 2]
        x = x.reshape(x.shape[0], -1)                       # [B, 16, 8, 2, 2] -> [B, 512]
        validity = self.model(x)                            # [B, 512] -> [B, 1]
        return validity