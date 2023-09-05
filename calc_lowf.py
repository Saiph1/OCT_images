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

path = "../../test_dis/final_project/Bend-insensitive/2023-01-06/rat_trachea_S40_R20/rat_03_2023-01-06_00000"
count = 0

# Load the image
cv_img = cv2.imread(path + "/00187.oct.png")
cv_img2 = cv2.imread(path + "/00188.oct.png")

struts = [2206, 2247, 713+cv_img.shape[1], 777+cv_img.shape[1], 3767+cv_img.shape[1], 3820+cv_img.shape[1]]

info = []
# effective_areas = [struts[0]+(cv_img.shape[1]+cv_img2.shape[0]-struts[5]),struts[2]-struts[1],struts[4]-struts[3]]
effective_areas = [struts[0]+(cv_img.shape[1]+cv_img2.shape[1]-struts[5]),struts[2]-struts[1],struts[4]-struts[3]]
print(effective_areas)
info.append("Low_frquency_error = "+ str(math.sqrt(np.sum(((effective_areas - np.mean(effective_areas)) / np.mean(effective_areas))**2)/len(effective_areas))))

with open(path + "/resize/low_freq_00007.oct.txt", "w") as file:
    file.write(str(info))
