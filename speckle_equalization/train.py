# modified from https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
# from model import Net
import model_test
from model_test import STNClsNet
from dataset import data_loader
from torchinfo import summary
import argparse
import json
from torch.autograd import gradcheck

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type = int, default = 4)
parser.add_argument('--test-batch-size', type = int, default = 1000)
parser.add_argument('--epochs', type = int, default = 20)
parser.add_argument('--lr', type = float, default = 0.01)
parser.add_argument('--momentum', type=float, default = 0.5)
parser.add_argument('--no-cuda', action = 'store_true', default = False)
parser.add_argument('--seed', type = int, default = 1)
parser.add_argument('--log-interval', type = int, default = 10)
parser.add_argument('--save-interval', type = int, default = 100)
parser.add_argument('--model', required = True)
parser.add_argument('--angle', type = int, default=0)
parser.add_argument('--span_range', type = int, default = model_test.r1)
parser.add_argument('--grid_size', type = int, default = 4)
parser.add_argument('--checkpoint', type = int, default = 0)
args = parser.parse_args()
args.span_range_height = args.span_range_width = args.span_range
# args.grid_height = args.grid_width = args.grid_size
args.grid_height = model_test.grid_height
args.grid_width = model_test.grid_width
args.image_height = 512 # 1024
args.image_width = 3200 # 6400

iheight = args.image_height
iwidth = args.image_width

plt.ion()   # interactive mode
# model = Net().to(device)

# Data loading
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = STNClsNet(args).to(device)
# print(model)
# Get a summary of the model
model_summary = summary(model, input_size=(4, 1, args.image_height, args.image_width))
# # Print the summary
print(model_summary)

if args.checkpoint > 0:
    start = args.checkpoint
    model.load_state_dict(torch.load("./tmp/ep_{:05d}".format(args.checkpoint)+".pth"))
else :
    start = 1

def ssim_loss(prediction, ground_truth):
    # Compute the mean of the input images
    mean_pred = torch.mean(prediction, dim=(2, 3), keepdim=True)
    mean_gt = torch.mean(ground_truth, dim=(2, 3), keepdim=True)

    # Compute the standard deviation of the input images
    std_pred = torch.std(prediction, dim=(2, 3), keepdim=True)
    std_gt = torch.std(ground_truth, dim=(2, 3), keepdim=True)

    # Compute the covariance between the input images
    cov = torch.mean((prediction - mean_pred) * (ground_truth - mean_gt), dim=(2, 3), keepdim=True)

    # Compute the SSIM
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim = (2 * mean_pred * mean_gt + c1) * (2 * cov + c2) / ((mean_pred ** 2 + mean_gt ** 2 + c1) * (std_pred ** 2 + std_gt ** 2 + c2))

    # Compute the SSIM loss
    ssim_loss = 1 - ssim

    return torch.mean(ssim_loss)

# model training:
optimizer = optim.SGD(model.parameters(), lr=0.1)
# scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.997)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size= 100, gamma=0.6)

def train(epoch):
    model.train()
    loss_per_epoch = 0.
    total_number_batches = 0 
    for batch_idx, (data, target) in enumerate(data_loader):
        # print(f"Epoch: {epoch}, Batch: {batch_idx}." )
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # loss_func = nn.MSELoss()
        loss_func = nn.L1Loss()
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 5 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), loss.item()))
        
        loss_per_epoch += loss.item()
        total_number_batches += 1
    
    return loss_per_epoch/total_number_batches

def visualize_stn(ep):
    with torch.no_grad():
        # Get a batch of training data
        data_img = next(iter(data_loader))
        data = data_img[0].to(device)
        ground_truth = data_img[1].to(device)
        ground_truth = ground_truth.cpu()
        ground_truth = ground_truth*0.5+0.5
        input_tensor = data.cpu()
        transformed_input_tensor = model(data).cpu()

        input_tensor = input_tensor*0.5 +0.5 # reverse of z score normalization
        transformed_input_tensor = transformed_input_tensor*0.5+0.5

        import cv2
        cv2.imwrite("./tmp/ground_ep{}.png".format(ep), ground_truth[0].permute(1, 2, 0).numpy() * 255)
        cv2.imwrite("./tmp/in_ep{}.png".format(ep), input_tensor[0].permute(1, 2, 0).numpy() * 255)
        cv2.imwrite("./tmp/out_ep{}.png".format(ep), transformed_input_tensor[0].permute(1, 2, 0).numpy() * 255)

All_losses = [] # per 50 epoch

for epoch in range(start, 1000 + 1):
    # test()
    if epoch == 1: 
        visualize_stn(epoch)
        # torch.save(model.state_dict(), "./tmp/ep_"+str(epoch)+".pth"
    epoch_loss = train(epoch)
    if epoch == 2: 
        visualize_stn(epoch)
        torch.save(model.state_dict(), "./tmp/ep_{:05d}".format(epoch)+".pth")
    
    if epoch % 200 == 0: 
        visualize_stn(epoch)
    
    if epoch % 200 == 0:
        torch.save(model.state_dict(), "./tmp/ep_{:05d}".format(epoch)+".pth")

    if epoch % 20 == 0: 
        All_losses.append(epoch_loss)
        with open('./tmp/Losses_per_20_epochs.json', 'w') as f:
            json.dump(All_losses, f)

    scheduler.step()