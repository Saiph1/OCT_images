import glob
import cv2
import numpy as np
from mmcv import Config
from mmdet.apis import inference_detector
from ssod.apis.inference import init_detector
from ssod.utils import patch_config

# path = "./data/mouse_03_2022-11-24_00000"
path = "./data/synthesized/00032.oct.png"
# output_path = "./output/mouse_1124/"
# config_path = "./config2.py"
config_path = "./config.py"
model_path = "./iter_124000.pth"
# model_path = "./iter_180000_oct31.pth"
score_thr = 0.3 # filtering the undesired prediction. (sigmoid output)

def main():
    # build the model from a config file and a checkpoint file
    # imgs = sorted(glob.glob(path+"/*.oct.*"))
    imgs = sorted(glob.glob(path))
    print("All images length: ", len(imgs))
    # loading the model config:
    cfg = Config.fromfile(config_path)
    cfg = patch_config(cfg)
    # build the model from a config file and a checkpoint file
    model = init_detector(cfg, model_path, device="cpu")
    model.eval()
    for img in imgs:
        cv_img = cv2.imread(img) # load image using cv2 library. 
        # forward inference of the faster RCNN NURD region detection network.
        result = inference_detector(model, cv_img)
        x_data = []
        for item in result[0]:
            if item[4] > score_thr:
                x_data.append(int(item[0]))
                x_data.append(int(item[2]))
        
        # create overlay
        overlay = cv_img.copy()
        cv_img2 = cv_img.copy()
        for item in result[0]:
            # print(item)
            if item[4] > score_thr:
                cv2.rectangle(overlay, (int(item[0]),0), (int(item[2]), 1024), (0,0,255), 2, lineType=cv2.LINE_AA)
                cv2.rectangle(overlay, (int(item[0]),0), (int(item[2]), 1024), (0,0,255), -1)            

        for item in result[1]:
            if item[4] > score_thr:
                cv2.rectangle(overlay, (int(item[0]),0), (int(item[2]), 1024), (255,0,0), 2, lineType=cv2.LINE_AA)
                cv2.rectangle(overlay, (int(item[0]),0), (int(item[2]), 1024), (255,0,0), -1)

        # transparent level alpha
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, cv_img2, 1-alpha, 0, cv_img2)
        cv2.imwrite("./network_predict.png", cv_img2)
        print("image stored.")

if __name__ == "__main__":
    main()