import cv2
import numpy as np
import glob 
from mmcv import Config
from mmdet.apis import async_inference_detector, inference_detector, show_result_pyplot
from ssod.apis.inference import init_detector, save_result
from ssod.utils import patch_config

config_path = "./config.py"
model_path = "./iter_124000.pth"
path = "./data/"
imgs = sorted(glob.glob(path +"*.png"))
print("length = ", len(imgs))
imgs = imgs[330:]

score_thr = 0.3
cfg = Config.fromfile(config_path)
# Not affect anything, just avoid index error
cfg.work_dir = "./work_dirs"
cfg = patch_config(cfg)
# build the model from a config file and a checkpoint file
model = init_detector(cfg, model_path, device="cpu")
model.eval()
    
for i, img in enumerate(imgs):
    if i % 2 != 0: 
        continue
    # introduce checkpoint resume :)
    if i< 600:
        continue
    cv_img1 = cv2.imread(imgs[i])
    cv_img2 = cv2.imread(imgs[i+1])
    # cv_img3 = cv2.imread(imgs[i+2])
    # cv_img4 = cv2.imread(imgs[i+3])
    new = np.concatenate((cv_img1, cv_img2), axis=1)
    # new = np.concatenate((new, cv_img3), axis=1)
    # new = np.concatenate((new, cv_img4), axis=1)
    print("shape = ", new.shape)
    new = cv2.resize(new, (cv_img1.shape[1], cv_img1.shape[0]), interpolation=cv2.INTER_LANCZOS4)

    result = inference_detector(model, new)
    x_data2 = []
    for item in result[1]:
        if item[4] > score_thr:
            x_data2.append(int(item[0]))
            x_data2.append(int(item[2]))

    x_data2 = np.sort(np.array(x_data2))
    print("Struts = ", x_data2)
    strut_length = (new.shape[1]*4)/360
    # print("strut", strut_length)
    # a full length of the normal region should be cv_img.shape[1]*116/360
    # normal_length = (cv_img.shape[1]*116)/360
    normal_length = (new.shape[1]*116)/360
    # print("normal length,", normal_length)
    # cv_img2 = [] if x_data2[0] == 0 else resized_img[:, :x_data2[0]]
    x_data2 = x_data2.astype(int)
    while len(x_data2)>6 :
        for k in range(len(x_data2)):
            if k%2 == 0:
                continue 
            if k == len(x_data2) -1: 
                break
            if x_data2[k+1]-x_data2[k] <800:
                print("test", x_data2)
                x_data2 = np.delete(x_data2, [k, k+1])
                print(x_data2)
                break
    
    cv_img2 = []
    for j, item in enumerate(x_data2):
        if j == len(x_data2)-1:
            # new_region = resized_img[:, x_data[j]:]
            break
        if j%2 == 0:  
            new_region = new[:, x_data2[j]:x_data2[j+1]]
            new_region = cv2.resize(new_region, (int(strut_length), new.shape[0]), interpolation=cv2.INTER_LANCZOS4)
            cv_img2 = np.concatenate((cv_img2, new_region), axis=1) if cv_img2 != [] else new_region
        else: 
            new_region = new[:, x_data2[j]:x_data2[j+1]]
            new_region = cv2.resize(new_region, (int(normal_length), new.shape[0]), interpolation=cv2.INTER_LANCZOS4)
            cv_img2 = np.concatenate((cv_img2, new_region), axis= 1)
    
    total_length = x_data2[0]+new.shape[1]-1-x_data2[-1]
    print("total length", total_length)
    l1 = ((x_data2[0])/total_length)*(normal_length)
    l2 = ((new.shape[1]-1-x_data2[-1])/total_length)*(normal_length)
    if l1 < 0 or l2 < 0:
        continue
    print("ls", l1 ,l2)
    region1 = new[:, 0:x_data2[0]]
    region2 = new[:, x_data2[-1]:]
    region1 = cv2.resize(region1, (int(l1), new.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    region2 = cv2.resize(region2, (int(l2), new.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    cv_img2 = np.concatenate((cv_img2, region2), axis= 1)
    cv_img2 = np.concatenate((region1, cv_img2), axis= 1)

    cv_img2 = cv2.resize(cv_img2, (new.shape[1], new.shape[0]), interpolation= cv2.INTER_LANCZOS4)
    # cv2.imwrite(path + "/rescale_"+img[-13:], cv_img2[:,:,0])
    # print(new.shape)
    print("Number: {:05d}".format(int(i/2)))
    cv2.imwrite("./rescale/"+"{:05d}.oct.png".format(int(i/2)), cv_img2[:,:,0])
    print("Done storing image ({:05d}.oct.png)".format(int(i/2)))