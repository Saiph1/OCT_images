import numpy as np
from pyswarm import pso
from PIL import Image
import glob
from mmcv import Config
from mmdet.apis import async_inference_detector, inference_detector, show_result_pyplot
from ssod.apis.inference import init_detector, save_result
from ssod.utils import patch_config
import math
import cv2

# This script is used to calculate the low frequency error presented in an ICT image. 

path = "./data"
count = 0

# Load the image
cv_img = cv2.imread(path + "/00187.oct.png") # Two images concatenate to construct one completed oct image. 
cv_img2 = cv2.imread(path + "/00188.oct.png") # Two images concatenate to construct one completed oct image. 

# Manual inspection for the struts positions. 
struts = [2206, 2247, 713+cv_img.shape[1], 777+cv_img.shape[1], 3767+cv_img.shape[1], 3820+cv_img.shape[1]]

info = []
# effective_areas = [struts[0]+(cv_img.shape[1]+cv_img2.shape[0]-struts[5]),struts[2]-struts[1],struts[4]-struts[3]]
effective_areas = [struts[0]+(cv_img.shape[1]+cv_img2.shape[1]-struts[5]),struts[2]-struts[1],struts[4]-struts[3]]
print(effective_areas)
info.append("Low_frquency_error = "+ str(math.sqrt(np.sum(((effective_areas - np.mean(effective_areas)) / np.mean(effective_areas))**2)/len(effective_areas))))

# stores the low frequency information for this corresponding image into a txt file. 
with open(path + "/resize/low_freq_00007.oct.txt", "w") as file:
    file.write(str(info))
