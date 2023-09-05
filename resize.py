# Copyright (c) OpenMMLab. All rights reserved.
# Modified from thirdparty/mmdetection/demo/image_demo.py
import glob
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import math
from scipy.optimize import curve_fit, leastsq, minimize
import scipy.integrate as spi
from skimage.metrics import structural_similarity as ssim
from mmcv import Config
from mmdet.apis import async_inference_detector, inference_detector, show_result_pyplot
from ssod.apis.inference import init_detector, save_result
from ssod.utils import patch_config
from collections import Counter

# This script reposition and remap the OCT image content after the concatenation. 
# A completed, repaired OCT image should have 3 struts each occupying 4/360 portion of the whole image. 
# Normal areas separated by the struts (assuming 4 in total) should add up to (360-4*3)/360. 

def main():
    # build the model from a config file and a checkpoint file
    # imgs = sorted(glob.glob(args.img+"/*"))
    # imgs = sorted(glob.glob("../../../../../oct_data/main_data/in_vivo_rat_colorectum_R20fps_S40fps/complete/*"))
    # print("order: ", imgs)
    path = "../../test_dis/aug/mouse_05_2022-12-03_00000/resize/"
    imgs = sorted(glob.glob(path+"*.png"))
    print("All images: ", imgs, len(imgs))
    score_thr = 0.5
    
    # cfg = Config.fromfile("data/soft_teacher_faster_rcnn_r50_caffe_fpn_coco_180k.py")
    cfg = Config.fromfile("data/0722.py")
    # Not affect anything, just avoid index error
    cfg.work_dir = "./work_dirs"
    cfg = patch_config(cfg)
    # build the model from a config file and a checkpoint file
    model = init_detector(cfg, "data/0722iter_86000.pth", device="cpu")
    model.eval()
    count = 0
    for img in imgs:
        count +=1
        if count < 30:
            continue
        # test a single image
        cv_img = cv2.imread(img)
        # load image in cv2 format for further process of region colors.
        # show the results
        # print(f"Saving results for No.{count+1} image...")
        print("result of: ", img[-13:])
        # print("count = ", count+1)
        # print("p = ", p)
        result = inference_detector(model, cv_img)

        x_data2 = []
        for item in result[1]:
            if item[4] > score_thr:
                x_data2.append(int(item[0]))
                x_data2.append(int(item[2]))

        # if len(x_data) == 0:
        #     break 

        x_data2 = np.sort(np.array(x_data2))

        # print(x_data)
        # counts = Counter(x_data)
        # x_data = [elem for elem in x_data if counts[elem] == 1]

        # cv2.imshow("Pre-process",cv_img)
        # cv2.imwrite("./demo/result.png", cv_img)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        print(x_data2)

        # PROCESS: Resizing the strut and the normal areas
        # Here I assume that the strut has the length equal to the shortest x_data range (the length of the strut.)
        # suppose the final image has the width of 6244, or cv_img.shape[1], each strut should have the length of size*4/360, constant value.
        # strut_length = (cv_img.shape[1]*4)/360
        strut_length = (cv_img.shape[1]*4)/360
        # print("strut", strut_length)
        # a full length of the normal region should be cv_img.shape[1]*116/360
        # normal_length = (cv_img.shape[1]*116)/360
        normal_length = (cv_img.shape[1]*116)/360
        # print("normal length,", normal_length)
        # cv_img2 = [] if x_data2[0] == 0 else resized_img[:, :x_data2[0]]
        x_data2 = x_data2.astype(int)

        # concat the same struts that are being detected more than once.
        # while len(x_data2)>6 :
        #     for k in range(len(x_data2)):
        #         if k%2 == 0:
        #             continue 
        #         if k == len(x_data2) -1: 
        #             break
        #         if x_data2[k+1]-x_data2[k] <800:
        #             print("test", x_data2)
        #             x_data2 = np.delete(x_data2, [k, k+1])
        #             print(x_data2)
        #             break
        if len(x_data2)<6 or len(x_data2)>8: 
            continue
        # x_data2 = [1465, 1504, 3176, 3234, 5067, 5169]
        print(x_data2)
        cv_img2 = []
        if (len(x_data2) ==6):
            if x_data2[-1] == 6244 or x_data2[0] < 5:
                continue
            for j, item in enumerate(x_data2):
                if j == len(x_data2)-1:
                    # new_region = resized_img[:, x_data[j]:]
                    break
                if j%2 == 0:  
                    new_region = cv_img[:, x_data2[j]:x_data2[j+1]]
                    new_region = cv2.resize(new_region, (int(strut_length), cv_img.shape[0]), interpolation=cv2.INTER_LANCZOS4)
                    cv_img2 = np.concatenate((cv_img2, new_region), axis=1) if cv_img2 != [] else new_region
                else: 
                    new_region = cv_img[:, x_data2[j]:x_data2[j+1]]
                    new_region = cv2.resize(new_region, (int(normal_length), cv_img.shape[0]), interpolation=cv2.INTER_LANCZOS4)
                    cv_img2 = np.concatenate((cv_img2, new_region), axis= 1)
            
            total_length = x_data2[0]+cv_img.shape[1]-1-x_data2[-1]
            print("total length", total_length)
        
            l1 = ((x_data2[0])/total_length)*(normal_length)
            l2 = ((cv_img.shape[1]-1-x_data2[-1])/total_length)*(normal_length)
            print("ls", l1 ,l2)
            region1 = cv_img[:, 0:x_data2[0]]
            region2 = cv_img[:, x_data2[-1]:]
            region1 = cv2.resize(region1, (int(l1), cv_img.shape[0]), interpolation=cv2.INTER_LANCZOS4)
            region2 = cv2.resize(region2, (int(l2), cv_img.shape[0]), interpolation=cv2.INTER_LANCZOS4)
            cv_img2 = np.concatenate((cv_img2, region2), axis= 1)
            cv_img2 = np.concatenate((region1, cv_img2), axis= 1)

            cv_img2 = cv2.resize(cv_img2, (cv_img.shape[1], cv_img.shape[0]), interpolation= cv2.INTER_LANCZOS4)
            cv2.imwrite(path + img[-13:], cv_img2[:,:,0])
        elif (len(x_data2) == 8):
            for j, item in enumerate(x_data2):
                if j == len(x_data2)-1:
                    # new_region = resized_img[:, x_data[j]:]
                    break
                if j%2 == 0:  
                    new_region = cv_img[:, x_data2[j]:x_data2[j+1]]
                    new_region = cv2.resize(new_region, (int(strut_length), cv_img.shape[0]), interpolation=cv2.INTER_LANCZOS4)
                    cv_img2 = np.concatenate((cv_img2, new_region), axis=1) if cv_img2 != [] else new_region
                else: 
                    new_region = cv_img[:, x_data2[j]:x_data2[j+1]]
                    new_region = cv2.resize(new_region, (int(normal_length), cv_img.shape[0]), interpolation=cv2.INTER_LANCZOS4)
                    cv_img2 = np.concatenate((cv_img2, new_region), axis= 1)

            cv_img2 = cv2.resize(cv_img2, (cv_img.shape[1], cv_img.shape[0]), interpolation= cv2.INTER_LANCZOS4)
            cv2.imwrite(path + img[-13:], cv_img2[:,:,0])   
        # cv2.imwrite(path + "/rescale_"+img[-13:], cv_img2[:,:,0])
        # cv2.imwrite("../../test_dis/original.png", cv_img)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        # count+=1

if __name__ == "__main__":
    main()
