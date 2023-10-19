# encoding: utf-8

import torch
import itertools
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, Variable

r1 = 0.9997
r2 = 0.999

class TPSGridGen(nn.Module):
    def __init__(self, target_height, target_width, target_control_points):
        super(TPSGridGen, self).__init__()

        target_control_points_diff = torch.diff(target_control_points, dim = 0)

        theta = torch.tensor([1,0,0,0,1,0], dtype=torch.float)
        theta = theta.view(-1,2,3)
        # creates the identity grid, same as the identity transformation in spatial transformation network. 
        grid = F.affine_grid(theta, [1,1,target_height, target_width])
        print(grid)
        print(target_control_points_diff)
        self.register_buffer('identity_grid', grid)
        # target_control_points_diff stores the information of the original position of the control points. 
        self.register_buffer('original_diff', target_control_points_diff)

    def forward(self, source_control_points, img_width, img_height):
        # source_control_points are the predicted location distributions of the control points displacement.
        # (computed with torch.diff)
        # This is computed in the range of [0, 1], which is the output of the Sigmoid function. 
        # Suppose there are 101 control points, then the source control point will be in shape of 101-1,
        # which stores the interval of the control points distance to the next consecutive points on the right. (counting from the left.)
        ratio = torch.div(source_control_points, self.original_diff) 
        # self.original_diff is the reference points position, if the source_control_points is close to the value 
        # of the self.original_diff, then it produces the same image.
        ratio = ratio / (img_width/2)
        # reduce the ratio to the scale of -1 to 1, with 3200 pixels values, 2/3200 = 1 pixel unit.
        # Ratio is the pixels per units that the interval between two points will be contained in the new frame. 
        space = img_width/source_control_points.shape[1]

        ratio = torch.repeat_interleave(ratio, int(space), dim =1)
        # repeated to fit the whole image width.
        ratio = ratio[:,:img_width-1]

        # Converts the interval to the actual position on the frame. 
        r1_tensor = torch.tensor([[-r1]]).to(ratio.device)
        ratio = torch.cat((r1_tensor.expand(4,1,1), ratio), dim=1)
        ratio = torch.cumsum(ratio, dim=1)

        # The identity grid if the image is resampled identically to the original image. 
        grid = self.identity_grid.clone().expand(source_control_points.shape[0], -1, -1, -1)
        
        # concat the computed ratio in the grid form.
        ratio = ratio.unsqueeze(1).expand_as(grid[:,:,:,1:])

        grid = torch.cat((ratio, grid[:,:,:,1:]), dim = 3)

        # Together with grid_sample function, this maps the old x,y coordinates of the original image to the new transformed image. 
        # The grid describes: what content should be placed on the new frame, from the old frame, the coordinates on the grid 
        # (1,2) "value" represents that pixel value at (1,2) should be used at this particular location.  
        return grid