import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from skimage.metrics import structural_similarity as compare_ssim
import pandas as pd
from skimage.filters import rank
# from skimage.draw import disk
from skimage.feature import local_binary_pattern
from skimage.transform import resize

from skimage import transform, io, color
from skimage.feature import match_descriptors, ORB, plot_matches
from skimage.measure import ransac
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

img1 = "./final_124k_02.png"
img2 = "./compare.png"
img = cv2.imread(img1)
reference = cv2.imread(img2)
start_size = img.shape[1]

# Step 1: Load the VGG network model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg_model = models.vgg16(pretrained=True)
vgg_model.eval()
loss_fn = nn.MSELoss()

# Step 2: Preprocess the images
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# img3 = "../../test_dis/cat.png"
# img4 = "../../test_dis/final/compare.png"

# ==========================================================
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = gray[400:600, :]
img_detect= img[400:600, :]
# gray_original = gray
print("gray shape", gray.shape)
gray = cv2.equalizeHist(gray)
# Define the size of the regions
region_size_h = 100
region_size_w = 20

# Compute the number of regions
num_cols = int(gray.shape[1] / region_size_w)
num_rows = int(gray.shape[0] / region_size_h)
ref_region = reference[300:400, 980:1000]
# cv2.imshow("reference", ref_region)
# cv2.waitKey()
# cv2.destroyAllWindows()
print(num_cols, num_rows)
# ref_region = cv2.cvtColor(ref_region, cv2.COLOR_BGR2GRAY)
# Compute the entropy for each region
entropies = np.zeros((num_rows, num_cols))
features = np.zeros((num_rows, num_cols))

for i in range(num_rows):
    for j in range(num_cols):
        # Extract the region 
        region = gray[i*region_size_h:(i+1)*region_size_h, j*region_size_w:(j+1)*region_size_w]
        print(region.shape, ref_region.shape)
        # Compute the histogram of pixel intensities
        hist = cv2.calcHist([region], [0], None, [256], [0, 256])
        hist = hist.ravel() / hist.sum()
        # Compute the entropy
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        image1 = preprocess(cv2.cvtColor(region, cv2.COLOR_GRAY2RGB))
        image2 = preprocess(ref_region)
        features1 = vgg_model(image1.unsqueeze(0))
        features2 = vgg_model(image2.unsqueeze(0))
        feature_loss = loss_fn(features1, features2)
        print("entropy vs. feature space diff. ", entropy, feature_loss.item())
        print("i, j", i,j)
        # Store the entropy value
        entropies[i, j] = entropy
        features[i, j] = feature_loss.item()

# Normalize the entropies
entropies = (entropies - entropies.min()) / (entropies.max() - entropies.min())
# features = (features - features.min()) / (features.max() - features.min())
# Compute the histogram of the data within the specified range
# ==============================================================================
# for testing purpose
bin_range = (0, 10)
# hist, bins = np.histogram(features, bins=20, range=bin_range)
# Plot the histogram
print(features)
print(features.shape)
plt.hist(features, bins=2, range=bin_range)

# # Add labels and title to the plot
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Data in Range [{}, {}]'.format(*bin_range))
# plt.show()
plt.savefig("histogram.png")
# ==============================================================================

print("shape", entropies.shape)
print("shape", features.shape)
# print(entropies)
# find the average along axis 0
avg = np.mean(entropies, axis=0)
# avg_features = np.mean(features, axis=0)
# create a boolean mask for elements that are below average
# indices = np.where((features > (avg_features + 1*np.std(features))) & (features < (avg_features + 2*np.std(features))))
indices = np.where((features > 4)&(features <100000))
print(indices)
x_coords = indices[1] * region_size_w
y_coords = indices[0] * region_size_h                                                         
x_set = set(x_coords)
print("filtered x size", len(x_set))

# print the indices
print("Regions with entropy below average: x ", sorted(list(x_set)))
print("Total length = ", len(x_set))
# ===================================================================
# Visualize the result chosen for the range below range. 
# features[(features > 2)&(features <4)] = 0 
mask_chosen = (features > 4) & (features < 100000)
features[mask_chosen] = 1
features[~mask_chosen] = 0

# entropies = [entropies>0.5]
# Create a heatmap
heatmap = cv2.applyColorMap(np.uint8(entropies * 255), cv2.COLORMAP_JET)
heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
# Resize the heatmap to match the original image
heatmap = cv2.resize(heatmap, (img_detect.shape[1], img_detect.shape[0])) 
# Overlay the heatmap on the original image
print(heatmap.shape, img_detect.shape)
result2 = cv2.addWeighted(img_detect, 0.5, heatmap, 0.5, 0)

heatmap2 = cv2.applyColorMap(np.uint8(features * 255), cv2.COLORMAP_JET)
heatmap2 = cv2.cvtColor(heatmap2, cv2.COLOR_BGR2RGB)
# Resize the heatmap to match the original image
heatmap2 = cv2.resize(heatmap2, (img_detect.shape[1], img_detect.shape[0])) 
# Overlay the heatmap on the original image
print(heatmap2.shape, img_detect.shape)
result3 = cv2.addWeighted(img_detect, 0.5, heatmap2, 0.5, 0)

# Display the result
# cv2.imshow("heatmap", result)
# cv2.waitKey()
# cv2.destroyAllWindows()
# plt.imshow(result2)
# plt.axis('off')
# plt.show()

plt.imshow(result3)
plt.axis('off')
plt.savefig("heatmap_for_features.png")
plt.show()

x_bounds = sorted(list(x_set))

print(x_bounds)
while(len(x_set)!=0):
    cv_img2 = [] if x_bounds[0]==0 else img[0:1024, 0:int(x_bounds[0])]
    for xs, item in enumerate(x_bounds):
        if item + 20 >= img.shape[1]:
            break
        subregion = (int(x_bounds[xs]),0,int(x_bounds[xs]+20), 1024)
        # Crop the subregion from the original image
        cropped_image = img[:, subregion[0]:subregion[2]]
        for ph in range(3):
            l1 = int(2*cropped_image.shape[1]/10)
            l2 = int((8*cropped_image.shape[1])/10)
            resize_image = cropped_image[:, l1:l2]
            left = cropped_image[:, :l1]
            right = cropped_image[:, l2:]
            # scale_factor = 0.9 - ph*0.1
            scale_factor = 0.98 -ph*0.02
            # Calculate the new width and height after downsampling
            new_width = int(resize_image.shape[1] * scale_factor)
            new_height = int(resize_image.shape[0])
            # Perform Bilinear interpolation to downsample the subregion
            # print("resizing x_data", dis)
            downsampled_subregion = cv2.resize(resize_image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
            # dst = left + mid + right
            cropped_image = np.concatenate((left, downsampled_subregion), axis = 1) if left != [] else downsampled_subregion
            if right != []:
                cropped_image = np.concatenate((cropped_image, right), axis = 1) 
            print("l1 ,l2", l1, l2)
            # print("cropped shape", cropped.shape)
            # print("left and right shape: ", left.shape, right.shape)
            if resize_image.shape[1] < 5: 
                break
            downsampled_subregion = cropped_image
            print("downsampled_subregion size: ", downsampled_subregion.shape)
        
        # ==========================================================
        # scale_factor = 0.8
        # # Calculate the new width and height after downsampling
        # new_width = int(cropped_image.shape[1] * scale_factor)
        # new_height = int(cropped_image.shape[0])
        # downsampled_subregion = cv2.resize(cropped_image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        
        # Create a transition region with a horizontal gradient
        if cv_img2 != []:
            cv_img2 = np.concatenate((cv_img2, downsampled_subregion), axis=1) 
        else: 
            cv_img2 = downsampled_subregion

        if xs == len(x_bounds)-1:
            cv_img2 = np.concatenate((cv_img2, img[:, (x_bounds[xs]+20):]), axis=1)
        else:
            cv_img2 = np.concatenate((cv_img2, img[:, (x_bounds[xs]+20):x_bounds[xs+1]]), axis=1)

    print("resized_image shape, ", cv_img2.shape, img.shape)
    # Define the size of the regions
    cv2.imwrite("./tmp.png", cv_img2)
    img = cv_img2
    img = cv2.resize(img, (start_size, 1024), interpolation=cv2.INTER_LANCZOS4)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = gray[400:600, :]
    region_size_h = 100
    region_size_w = 20

    # Compute the number of regions
    num_cols = int(gray.shape[1] / region_size_w)
    num_rows = int(gray.shape[0] / region_size_h)

    # Compute the entropy for each region
    entropies = np.zeros((num_rows, num_cols))
    features = np.zeros((num_rows, num_cols))
    i = j = 0
    for i in range(num_rows):
        for j in range(num_cols):
            # Extract the region 
            region = gray[i*region_size_h:(i+1)*region_size_h, j*region_size_w:(j+1)*region_size_w]
            # Compute the histogram of pixel intensities
            hist = cv2.calcHist([region], [0], None, [256], [0, 256])
            hist = hist.ravel() / hist.sum()
            # Compute the entropy
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            image1 = preprocess(cv2.cvtColor(region, cv2.COLOR_GRAY2RGB))
            image2 = preprocess(ref_region)
            features1 = vgg_model(image1.unsqueeze(0))
            features2 = vgg_model(image2.unsqueeze(0))
            feature_loss = loss_fn(features1, features2)
            print("entropy vs. feature space diff. ", entropy, feature_loss.item())
            print("i, j", i,j)
            features[i, j] = feature_loss.item()
            # Store the entropy value
            entropies[i, j] = entropy

    # Normalize the entropies
    entropies = (entropies - entropies.min()) / (entropies.max() - entropies.min())
    # features = (features - features.min()) / (features.max() - features.min())
    print("shape", entropies.shape)
    # find the average along axis 0
    avg = np.mean(entropies, axis=0)
    # avg_features = np.mean(features, axis=0)
    # create a boolean mask for elements that are below average
    # indices = np.where((features > (avg_features + 1*np.std(features))) & (features < (avg_features + 3.5*np.std(features))))
    indices = np.where((features > 4)&(features <10000))
    print(indices)
    x_coords = indices[1] * region_size_w
    y_coords = indices[0] * region_size_h
    x_set = set(x_coords)
    x_bounds = sorted(list(x_set))
    # hist, bins = np.histogram(features, bins=100, range=bin_range)
    # Plot the histogram
    plt.hist(features, bins=3, range=bin_range)

    # # Add labels and title to the plot
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Data in Range [{}, {}]'.format(*bin_range))
    # plt.show()
    plt.savefig("histogram1.png")
    # print the indices
    print("Regions with entropy below average: x ", sorted(list(x_set)))
    print("Total length: ", len(x_set))
    
cv2.imshow("resized", img)
cv2.imwrite("./feature_space_resize.png", img)
cv2.waitKey()
cv2.destroyAllWindows()
# ==============================================================
# gray = cv2.cvtColor(cv_img2, cv2.COLOR_BGR2GRAY)
# cv_img2 = cv_img2[400:600, :]
# gray = gray[400:600, :]
# # Define the size of the regions
# egion_size_h = 100
# region_size_w = 20

# # Compute the number of regions
# num_cols = int(gray.shape[1] / region_size_w)
# num_rows = int(gray.shape[0] / region_size_h)
# # Compute the entropy for each region
# entropies = np.zeros((num_rows, num_cols))
# for i in range(num_rows):
#     for j in range(num_cols):
#         # Extract the region 
#         region = gray[i*region_size:(i+1)*region_size, j*region_size:(j+1)*region_size]
#         # Compute the histogram of pixel intensities
#         hist = cv2.calcHist([region], [0], None, [256], [0, 256])
#         hist = hist.ravel() / hist.sum()
        
#         # Compute the entropy
#         entropy = -np.sum(hist * np.log2(hist + 1e-10))
#         # Store the entropy value
#         entropies[i, j] = entropy

# # Normalize the entropies
# entropies = (entropies - entropies.min()) / (entropies.max() - entropies.min())
# avg = np.mean(entropies, axis=0)
# entropies[entropies < (avg - 2.5*np.std(entropies))] = 0 
# # Create a heatmap
# heatmap = cv2.applyColorMap(np.uint8(entropies * 255), cv2.COLORMAP_JET)
# heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
# # Resize the heatmap to match the original image
# heatmap = cv2.resize(heatmap, (cv_img2.shape[1], cv_img2.shape[0]))
# # Overlay the heatmap on the original image
# result = cv2.addWeighted(cv_img2, 0.5, heatmap, 0.5, 0)
# # Display the result
# plt.imshow(result)
# plt.axis('off')
# plt.show()
region_size_w