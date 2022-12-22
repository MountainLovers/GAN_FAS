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
    def __init__(self, in_channels=64, out_channels=1):
        super(Decoder, self).__init__()
        frames = 64
        self.ConvBlock = nn.Conv3d(in_channels, out_channels, [1,1,1],stride=1, padding=0)
        self.poolspa = nn.AdaptiveAvgPool3d((frames,1,1))


    def forward(self, x):
        """
            inputs :
                x : latent-space feature [B, C:64, T:64, W:32, H:32]
            returns :
                x : rppg signal [B, T:64]
        """
        x = self.poolspa(x)                     # [64, 64, 32, 32] -> [64, 64, 1, 1]
        x = self.ConvBlock(x).squeeze(1).squeeze(-1).squeeze(-1)    # [64, 64, 1, 1] -> [1, 64, 1, 1] -> [ , 64]
     
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

class Encoder32(nn.Module):
    def __init__(self):
        super(Encoder32, self).__init__()

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

    def forward(self, x):
        """
            inputs :
                x : input feature [B, C:3, T:64, W:32, H:32]
            returns :
                x : latent-space feature [B, C:64, T:64, W:32, H:32]
        """
        # print("In Encoder32 forward(), x: {}".format(x.shape))
        x = self.Conv1(x)                   # [3, 64, 32, 32] -> [16, 64, 32, 32]
        # print("x1: {}".format(x))

        x = self.STConv1(x)                 # [16, 64, 32, 32] -> [32, 64, 32, 32]
        x = self.STConv2(x)                 # [32, 64, 32, 32] -> [32, 64, 32, 32]
        # print("x2: {}".format(x))

        x = self.STConv3(x)                 # [32, 64, 32, 32] -> [64, 64, 32, 32]
        x = self.STConv4(x)                 # [64, 64, 32, 32] -> [64, 64, 32, 32]
        # print("x3: {}".format(x))

        return x

class Classifier(nn.Module):
    def __init__(self, in_channels=128):
        super(Classifier, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(64, 64, [3,3,3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(64, 128, [3,3,3], stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(128, 256, [3,3,3], stride=1, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(256, 512, [3,3,3], stride=1, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            )

        self.avgpooling = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.classifier = nn.Sequential(nn.Linear(512, 128),
                                    nn.BatchNorm1d(128),
                                    nn.Dropout(p=0.3),
                                    nn.ReLU(),
                                    nn.Linear(128, 2))
                    

    def forward(self, x):
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

        self.Conv = nn.Sequential(
            nn.Conv3d(4, 16, [1,5,5], stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(16, 32, [1,5,5], stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(32, 64, [3,3,3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.Conv3d(64, 64, [3,3,3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.GAP = nn.AdaptiveAvgPool3d((2, 2, 2))

        self.FC = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # print("In Discriminator(), x.dtype: {}".format(x.dtype))
        b = x.shape[0]
        x = self.Conv(x)                # [B, 4, D, W, H] -> [B, 64, D, W, H]
        x = self.GAP(x)                 # [B, 64, D, W, H] -> [B, 64, 2, 2, 2]
        x = x.reshape(b, -1)            # [B, 64, 2, 2, 2] -> [B, 512]
        output = self.FC(x)
        return output
