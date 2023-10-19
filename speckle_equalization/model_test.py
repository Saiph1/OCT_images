# encoding: utf-8
# from: https://github.com/WarBean/tps_stn_pytorch/blob/master/mnist_model.py
# https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html
# https://github.com/ayumiymk/aster.pytorch/blob/master/lib/models/stn_head.py

import math
import torch
import itertools
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from grid_sample import grid_sample
from torch.autograd import Variable
from tps_grid_gen import TPSGridGen

r1 = 0.9997
r2 = 0.999
grid_height = 10
grid_width = 201 # make sure this-1 is a factor of the img width.

def conv3x3_block(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    conv_layer = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1)

    block = nn.Sequential(
    conv_layer,
    nn.BatchNorm2d(out_planes),
    nn.ReLU(inplace=True),
    )
    return block

class CNN(nn.Module):
    def __init__(self, num_output):
        super(CNN, self).__init__()

        self.control_points = num_output
        self.stn_convnet = nn.Sequential(
                          conv3x3_block(1, 32), # 32*64
                          nn.MaxPool2d(kernel_size=2, stride=2),
                          conv3x3_block(32, 64), # 16*32
                          nn.MaxPool2d(kernel_size=2, stride=2),
                          conv3x3_block(64, 128), # 8*16
                          nn.MaxPool2d(kernel_size=2, stride=2),
                          conv3x3_block(128, 256), # 4*8
                          nn.MaxPool2d(kernel_size=2, stride=2),
                          conv3x3_block(256, 512), # 2*4,
                          nn.MaxPool2d(kernel_size=2, stride=2),
                          conv3x3_block(512, 512), # 1*2
                          nn.MaxPool2d(kernel_size=2, stride=2),
                          conv3x3_block(512, 512), 
                          nn.MaxPool2d(kernel_size=2, stride=2),
                          conv3x3_block(512, 1024),
                          nn.Dropout(0.15))
        
        self.stn_fc1 = nn.Sequential(
                    # nn.Linear(2*256, 512),
                    # nn.Linear(32*256*200, 512),
                    nn.Linear(1024* 100, 1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.25))

        self.stn_fc2 = nn.Linear(1024, num_output)
        

    def forward(self, x):

        x = self.stn_convnet(x)
        # print("x shape = ", x.shape)
        batch_size, _, h, w = x.size()
        x = x.view(batch_size, -1)
        # x = x.view(-1, 32*256*200)
        img_feat = self.stn_fc1(x)
        img_feat = F.dropout(img_feat, training=self.training)
        x = self.stn_fc2(img_feat)
        return x

# This is not used, BoundedGridLocNet is used. 
class UnBoundedGridLocNet(nn.Module):

    def __init__(self, grid_height, grid_width, target_control_points):
        super(UnBoundedGridLocNet, self).__init__()
        self.cnn = CNN(grid_height * grid_width * 2)
        bias = target_control_points.view(-1)
        self.cnn.stn_fc2.bias.data.copy_(bias)
        self.cnn.stn_fc2.weight.data.zero_()

    def forward(self, x):
        batch_size = x.size(0)
        points = self.cnn(x)
        return points.view(batch_size, -1, 2)

class BoundedGridLocNet(nn.Module):

    def __init__(self, grid_height, grid_width, target_control_points):
        super(BoundedGridLocNet, self).__init__()
        self.cnn = CNN(grid_width*1 - 1)
        # print(target_control_points.shape)
        target_control_points_diff = torch.diff(target_control_points, dim = 0)

        # compute inverse of sigmoid.
        # bias = np.log(target_control_points_diff.numpy() / 2*(1 - target_control_points_diff.numpy()))
        # bias = torch.from_numpy(bias)
        bias = torch.from_numpy(target_control_points_diff)
        # print("bias = ", bias)
        bias = bias.view(-1)
        # Initialization of the target point locations, such that training starts from the identity transformation. 
        self.cnn.stn_fc2.bias.data.copy_(bias)
        self.cnn.stn_fc2.weight.data.zero_()

    def forward(self, x):

        batch_size = x.size(0)
        # points = F.tanh(self.cnn(x)
        x = self.cnn(x)
        x = 2 * F.sigmoid(x) # ensure positive
        # points = 2 * r1 * F.softmax(torch.log(x)) # ensure sums up to 2*r1
        points = 2 * r1 * F.softmax(torch.log(x)) 
        points = torch.unsqueeze(points, dim=2)
        return points.view(batch_size, -1, 1)

class STNClsNet(nn.Module):

    def __init__(self, args):
        super(STNClsNet, self).__init__()
        self.args = args

        r1 = args.span_range_height
        r2 = args.span_range_width
        assert r1 < 1 and r2 < 1 # if >= 1, arctanh will cause error in BoundedGridLocNet
        target_control_points = torch.Tensor(list(itertools.product(
            np.arange(-r1, r1 + 0.00001, 2.0  * r1 / (args.grid_height - 1)),
            np.arange(-r2, r2 + 0.00001, 2.0  * r2 / (args.grid_width - 1)),
        )))
        Y, X = target_control_points.split(1, dim = 1)

        target_control_points_1c = torch.cat([X], dim = 1)
        target_control_points_1c = target_control_points_1c[:args.grid_width,:]
        # This is the initialization of the identity target points position uniformly distributed across an image, in x axis. 

        GridLocNet = {
            'unbounded_stn': UnBoundedGridLocNet,
            'bounded_stn': BoundedGridLocNet,
        }[args.model]

        self.loc_net = GridLocNet(args.grid_height, args.grid_width, target_control_points_1c)
        self.tps = TPSGridGen(args.image_height, args.image_width, target_control_points_1c)

    def forward(self, x):
        # batch_size = x.size(0)
        source_control_points = self.loc_net(x)

        grid = self.tps(source_control_points, x.shape[3], x.shape[2])

        transformed_x = grid_sample(x, grid)
        # grid sample is the gradient traceable built-in function in torch.F

        return transformed_x

def get_model(args):
    if args.model == 'no_stn':
        print('create model without STN')
        model = ClsNet()
    else:
        print('create model with STN')
        model = STNClsNet(args)
    return model