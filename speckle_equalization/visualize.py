# encoding: utf-8
# modified from https://github.com/WarBean/tps_stn_pytorch/blob/master/single_visualize.py

import os
import glob
import torch
import random
import argparse
import model_test
from dataset import data_loader
# import data_loader_original
import numpy as np
from model_test import STNClsNet
from grid_sample import grid_sample
from torch.autograd import Variable
from PIL import Image, ImageDraw, ImageFont
from torchvision import datasets, transforms
import itertools
from model_test import grid_height, grid_width

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type = int, default = 4)
parser.add_argument('--angle', type = int, default = 0)
parser.add_argument('--no-cuda', action = 'store_true', default = False)
parser.add_argument('--model', required = True)
parser.add_argument('--span_range', type = int, default = 0.9)
parser.add_argument('--grid_size', type = int, default = 4)
args = parser.parse_args()

args.span_range_height = args.span_range_width = args.span_range
args.grid_height = grid_height
# args.grid_height = 1
args.grid_width = grid_width
args.image_height = 512 # 1024
args.image_width = 3200 # 6400
# args.image_height = 28
# args.image_width = 28
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(1024)

assert args.model in ['bounded_stn', 'unbounded_stn']
model = STNClsNet(args).to(device)
if args.cuda:
    model.cuda()
image_dir = 'image/%s_angle%d_grid%d/' % (args.model, args.angle, args.grid_width)
if not os.path.isdir(image_dir):
    os.makedirs(image_dir)

test_loader = data_loader
# test_loader = data_loader_original.get_test_loader(args)
target2data_list = { i: [] for i in range(10) }
total = 0
N = 4
for data_batch, target_batch in test_loader:

    for data, target in zip(data_batch, target_batch):
        # print(data , data.shape)
        # print(target)
        data_list = target2data_list[0]
        if len(data_list) < N:
            data_list.append(data)
            total += 1
    if total == N:
        break
data_list = [target2data_list[i][j] for i in range(1) for j in range(N)]
source_data = torch.stack(data_list)
if args.cuda:
    source_data = source_data.cuda()
batch_size = 4
frames_list = [[] for _ in range(batch_size)]

paths = sorted(glob.glob('./tmp/*.pth'))
print(paths)
font = ImageFont.truetype('Comic Sans MS.ttf', 20)

r1 = 0.999
r2 = 0.999
test_height = grid_height
test_width = grid_width
target_control_points = torch.Tensor(list(itertools.product(
    np.arange(-r1, r1 + 0.00001, 2.0  * r1 / (2 - 1)),
    # np.arange(0.1, 0.1 + 0.00001, 2.0  * r1 / (test_height - 1)),
    np.arange(-r2, r2 + 0.00001, 2.0  * r2 / (test_width - 1)),
)))
y_height = 96
for pi, path in enumerate(paths): # path index
    print('path %d/%d: %s' % (pi, len(paths), path))
    model.load_state_dict(torch.load(path))
    source_control_points = model.loc_net(Variable(source_data))
    source_coordinate = model.tps(source_control_points, args.image_width, args.image_height)
    grid = source_coordinate.view(batch_size, args.image_height, args.image_width, 2)
    target_data = grid_sample(source_data, grid).data
    source_array = (source_data[:, 0] * 255).cpu().numpy().astype('uint8')
    target_array = (target_data[:, 0] * 255).cpu().numpy().astype('uint8')
    for si in range(batch_size): # sample index
        # resize for better visualization
        source_image = Image.fromarray(source_array[si]).convert('RGB').resize((400, 64))
        target_image = Image.fromarray(target_array[si]).convert('RGB').resize((400, 64))
        # create grey canvas for external control points
        canvas = Image.new(mode = 'RGB', size = (1500, 192), color = (128, 128, 128))
        canvas.paste(source_image, (64, 64))
        canvas.paste(target_image, (64 + 500, 64))
        source_points = source_control_points.data[si]
        # print(source_points, source_points.shape)
        # print(target_points, target_points.shape)
        source_points = torch.cat((torch.tensor([[-r1]]).to("cuda"), source_points), dim = 0)
        source_points = torch.cumsum(source_points, dim=0)
        # print(source_points, source_points.shape)
        source_points += 1
        source_points /= 2
        source_points *= 400
        source_points += 64 + 500
        # source_points[:,1] += 1
        # source_points[:,1] /= 2
        # source_points[:,1] *= 64
        # source_points[:,1] += 64
        
        # source_points = (source_points + 1) / 2 * 128 + 64
        draw = ImageDraw.Draw(canvas)
        for x in source_points:
            draw.rectangle([x - 1, y_height - 1, x + 1, y_height + 1], fill = (255, 0+x*20, 0))
        # source_points = source_points.view(1, args.grid_width, 2)
        source_points = source_points.view(1, args.grid_width, 1)
        # j left right, k up down
        for k in range(args.grid_width):
            x1 = source_points[0][k]
            if k > 0: # connect to left
                x2 = source_points[0][k - 1]
                draw.line((x1, y_height, x2, y_height), fill = (255, 0, 0))
            # if k > 0: # connect to up
            #     x2, y2 = source_points[k]
            #     draw.line((x1, y1, x2, y2), fill = (255, 0, 0))

        Y, X = target_control_points.split(1, dim = 1)
        target_points = torch.cat([X, Y], dim = 1)
        target_points[:,0] += 1
        target_points[:,0] /= 2
        target_points[:,0] *= 400
        target_points[:,0] += 64
        target_points[:,1] *= 0
        target_points[:,1] += 96 
        # target_points[:,1] += 1
        # target_points[:,1] /= 2
        # target_points[:,1] *= 64
        # target_points[:,1] += 64

        for x, y in target_points[:test_width,:]:
            draw.rectangle([x - 1, y - 1, x + 1, y + 1], fill = (255, 0+x*20, 0))
            draw.rectangle([x - 1  + 500, y - 1 - 16, x + 1  + 500, y + 1 - 16], fill = (255, 0+x*20, 0))
        target_points = target_points.view(2, test_width, 2)

        # j left right, k up down
        # for j in range(1):
        for j in range(1):
            for k in range(test_width):
                x1, y1 = target_points[j, k]
                if j > 0: # connect to left
                    x2, y2 = target_points[j - 1, k]
                    draw.line((x1, y1, x2, y2), fill = (255, 0, 0))
                if k > 0: # connect to up
                    x2, y2 = target_points[j, k - 1]
                    draw.line((x1, y1, x2, y2), fill = (255, 0, 0))

        for j in range(1):
            for k in range(test_width):
                x1, y1 = target_points[j, k]
                if j > 0: # connect to left
                    x2, y2 = target_points[j - 1, k]
                    draw.line((x1 + 500, 96-16, x2 + 500, 96-16), fill = (0, 0, 255))
                if k > 0: # connect to up
                    x2, y2 = target_points[j, k - 1]
                    draw.line((x1 + 500, 96-16, x2 + 500, 96-16), fill = (0, 0, 255))

        draw.text((10, 0), 'sample %03d, iter %03d, %s' % (si, pi, path[6:]), fill = (255, 0, 0), font = font)
        canvas.save(image_dir + 'sample%03d_checkpoint_%03d.png' % (si, pi))