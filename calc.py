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

imgs = sorted(glob.glob("../../test_dis/final_project/Bend-insensitive/2023-01-06/rat_trachea_S40_R20/rat_03_2023-01-06_00000/*.png"))
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
    x_data = []
    struts = []
    for strut in predict[1]:                                                                                
        struts.append(int(strut[0]))
        struts.append(int(strut[2]))
    for item in predict[0]:
        if item[4] > score_thr:
            x_data.append(int(item[0]))
            x_data.append(int(item[2]))

    x_data = np.sort(np.array(x_data))
    print(x_data)
    while len(x_data) != 0: 
        cv_img2 = [] if x_data[0]==0 else cv_img[0:1024, 0:int(x_data[0])]
        # def logistic_function(x, k, x0):
        #     return np.log((x+1)/(x0-x+1))*k + x0/2
        for i, dis in enumerate(x_data):
            if i%2:
                continue
            if x_data[i+1]-dis < 3: 
                continue

            subregion = (int(x_data[i]),0,int(x_data[i+1]), 1024)
            # Crop the subregion from the original image
            cropped_image = cv_img[:, subregion[0]:subregion[2]]
            # print(cropped_image.shape)
            # print(cv_img.shape)
            # Method 3: resize with concat and iterations:
            
            # cropped = src.copy()
            for ph in range(3):
                l1 = int(2*cropped_image.shape[1]/10)
                l2 = int((8*cropped_image.shape[1])/10)
                resize_image = cropped_image[:, l1:l2]
                left = cropped_image[:, :l1]
                right = cropped_image[:, l2:]
                scale_factor = 0.9 - ph*0.1
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
                
                # print("l1 ,l2", l1, l2)
                
                # print("cropped shape", cropped.shape)
                # print("left and right shape: ", left.shape, right.shape)
                if resize_image.shape[1] < 5: 
                    break

            downsampled_subregion = cropped_image
            print("downsampled_subregion size: ", downsampled_subregion.shape)
            
            # Create a transition region with a horizontal gradient
            if cv_img2 != []:
                # transition = np.zeros((new_height, transition_width, 3), dtype=np.uint8)
                # for j in range(transition_width):
                #     alpha = j / transition_width  # alpha increases from 0 to 1
                #     transition[:, j, :] = alpha*cv_img2[:, -transition_width+j, :] + (1-alpha)*downsampled_subregion[:, j, :]
                # cv_img2 = np.concatenate((cv_img2[:,:-transition_width], transition), axis=1) 
                # cv_img2 = np.concatenate((cv_img2, downsampled_subregion[:, transition_width:]), axis=1) 
                cv_img2 = np.concatenate((cv_img2, downsampled_subregion), axis=1) 
            else: 
                cv_img2 = downsampled_subregion

            if i == len(x_data)-2:
                if x_data[i+1] < cv_img.shape[1]-1: 
                    cv_img2 = np.concatenate((cv_img2, cv_img[0:1024, int(x_data[i+1]):]), axis=1)
                break

            cv_img2 = np.concatenate((cv_img2, cv_img[:, (x_data[i+1]):x_data[i+2]]), axis=1)

        cv_img = cv_img2
        predict = inference_detector(model, cv_img)
        x_data = []
        for item in predict[0]:
            if item[4] > score_thr:
                x_data.append(int(item[0]))
                x_data.append(int(item[2]))
        x_data = np.sort(np.array(x_data))
    if count %2 :
        info = []
        for elements_of_the_struts in range(len(struts)):
            struts[elements_of_the_struts] += buff_original
        all_struts = np.sort(buff_struts + (struts))
        print(all_struts)
        if len(all_struts) < 6:
            effective_areas = []
        else: 
            effective_areas = [all_struts[0]+(original_size[1]+buff_original-all_struts[5]),all_struts[2]-all_struts[1],all_struts[4]-all_struts[3]]
        print(effective_areas)
        info.append("Name = {:05d}".format(int(count/2)))
        info.append("Original size = "+ str(original_size[1]+buff_original))
        info.append("Final size = "+ str(cv_img.shape[1]+buff_final))
        if len(all_struts)<6:
            info.append("Low_frequency_error = NAN as number of struts < 3. ")
        else:
            info.append("Low_frquency_error = "+ str(math.sqrt(np.sum(((effective_areas - np.mean(effective_areas)) / np.mean(effective_areas))**2)/len(effective_areas))))
        info.append("High_frequency_error = "+ str((1-((buff_final+cv_img.shape[1])/(original_size[1]+buff_original)))))
        print("Final shape = "+ str(cv_img.shape[1]))
        print(int(count/2))
        with open("../../test_dis/final_project/Bend-insensitive/2023-01-06/rat_trachea_S40_R20/rat_03_2023-01-06_00000/resize/{:05d}.oct.txt".format(int(count/2)), "w") as file:
            file.write(str(info))
    else:
        buff_final = cv_img.shape[1]
        buff_original = original_size[1]
        buff_struts = struts

    count += 1