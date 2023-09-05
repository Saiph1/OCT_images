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

# This script calculates the low frequency error with the help of pretrained model, 
# which is used for detecting the position of the struts. 

path = "../../test_dis/final_project/Bend-insensitive/2023-01-06/rat_trachea_S40_R20/rat_03_2023-01-06_00000"
imgs = sorted(glob.glob(path+"/*.png"))
print("Length of the images: ", len(imgs))
# print("order: ", imgs)
score_thr = 0.3
count = 0
cfg = Config.fromfile("data/0722.py")
    # Not affect anything, just avoid index error
cfg.work_dir = "./work_dirs"
cfg = patch_config(cfg)
# build the model from a config file and a checkpoint file
model = init_detector(cfg, "data/0722iter_124000.pth", device="cpu")
model.eval()
# info = []
count = 0
for img in imgs:
    # Load the image
    cv_img = cv2.imread(img)
    original_size = cv_img.shape
    predict = inference_detector(model, cv_img)
    struts = []
    for strut in predict[1]:                                                                                
        struts.append(int(strut[0]))
        struts.append(int(strut[2]))
    
    if count %2 :
        info = []
        for elements_of_the_struts in range(len(struts)):
            struts[elements_of_the_struts] += buff_original
        all_struts = np.sort(buff_struts + (struts))
        print(all_struts)
        if len(all_struts) != 6 :
            print(img[-13:])
            count +=1
            continue
        else: 
            effective_areas = [all_struts[0]+(original_size[1]+buff_original-all_struts[5]),all_struts[2]-all_struts[1],all_struts[4]-all_struts[3]]
        print(effective_areas)
        info.append("Name = {:05d}".format(int(count/2)))
        info.append("Low_frquency_error = "+ str(math.sqrt(np.sum(((effective_areas - np.mean(effective_areas)) / np.mean(effective_areas))**2)/len(effective_areas))))
        print(int(count/2))
        with open(path + "/resize/low_freq_{:05d}.oct.txt".format(int(count/2)), "w") as file:
            file.write(str(info))
    else:
        buff_final = cv_img.shape[1]
        buff_original = original_size[1]
        buff_struts = struts

    count += 1