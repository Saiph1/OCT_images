import glob
import cv2
import numpy as np
from mmcv import Config
from mmdet.apis import inference_detector
from ssod.apis.inference import init_detector
from ssod.utils import patch_config

# Modified from mmdetection/demo/image_demo.py
# This script rescales the NURD region presented in an OCT image, such that oversampled information could be down-scaled.
# modify the path with your path to the OCT data
# A script for iterative restoration of the NURD oct image.

path = "./data"
output_path = "./output/"
config_path = "./config.py"
model_path = "./iter_124000.pth"

def main():
    # build the model from a config file and a checkpoint file
    # imgs = sorted(glob.glob(args.img+"/*"))
    # imgs = sorted(glob.glob("../../../../../oct_data/main_data/in_vivo_rat_colorectum_R20fps_S40fps/complete/*"))
    # print("order: ", imgs)
    imgs = sorted(glob.glob(path+"/*.png"))
    print("All images: ", imgs, len(imgs))
    score_thr = 0.6
    # cfg = Config.fromfile("data/soft_teacher_faster_rcnn_r50_caffe_fpn_coco_180k.py")
    cfg = Config.fromfile(config_path)
    # Not affect anything, just avoid index error
    # cfg.work_dir = "./work_dirs"
    cfg = patch_config(cfg)
    # build the model from a config file and a checkpoint file
    model = init_detector(cfg, model_path, device="cpu")
    model.eval()
    # model_ann = []
    for img in imgs:
        count = 0
        cv_img = cv2.imread(img)
        for p in range(30):
            # test a single image
            
            # load image in cv2 format for further process of region colors.
            # show the results
            # print(f"Saving results for No.{count+1} image...")
            print("result of: ", img[len(path)+1:])
            # print("count = ", count+1)
            print("p = ", p)
            # with open('../../../../../oct_data/main_data/in_vivo_rat_colorectum_R20fps_S40fps/region/'+img[-8:-4]+'.json', 'r') as file:
            #     my_list = json.load(file)
            # result = np.array(my_list)
            result = inference_detector(model, cv_img)

            x_data = []
            for item in result[0]:
                if item[4] > score_thr:
                    x_data.append(int(item[0]))
                    x_data.append(int(item[2]))

            if len(x_data) == 0:
                break 

            x_data = np.sort(np.array(x_data))
            print(x_data)
            # images for the stitch concat method:
            cv_img2 = [] if x_data[0]==0 else cv_img[0:1024, 0:int(x_data[0])]
            # def logistic_function(x, k, x0):
            #     return np.log((x+1)/(x0-x+1))*k + x0/2
            def logistic_function(x, k, x0):
                return (2*x0) / (1 + np.exp(-k * (x - x0)))
            
            for i, dis in enumerate(x_data):
                if i%2:
                    continue
                if x_data[i+1]-dis < 3: 
                    continue

                subregion = (int(x_data[i]),0,int(x_data[i+1]), 1024)
                # Crop the subregion from the original image
                cropped_image = cv_img[:, subregion[0]:subregion[2]]
                print(cropped_image.shape)
                print(cv_img.shape)
                # Define the scale factor for downsampling
                # scale_factor = scale[int(i/2)]
                # 0.05 ,0.1, 0.2, 0.5
                # 0.9 + 0.6 = test 06
                # ====================================================
                # Method 2: using the remap fucntion for non-uniform scaling. 
                # map_x = np.zeros((cropped_image.shape[0], cropped_image.shape[1]), dtype=np.float32)
                # map_y = np.zeros((cropped_image.shape[0], cropped_image.shape[1]), dtype=np.float32)
                # width = map_x.shape[1]
                # print("width",width)
                # for k in range(map_x.shape[0]):
                #     map_x[k,:] = [logistic_function(x, 100*np.pi/cropped_image.shape[1], cropped_image.shape[1]/2) for x in range(map_x.shape[1])]
                # for j in range(map_y.shape[1]):
                #     map_y[:,j] = [y for y in range(map_y.shape[0])]
                # downsampled_subregion = cv2.remap(cropped_image, map_x, map_y, cv2.INTER_LINEAR)
                # print("after", downsampled_subregion.shape)
                # cv2.imshow("rescaled", downsampled_subregion)
                # cv2.waitKey()
                # cv2.destroyAllWindows()
                # =====================================================
                # Method 1: using the resize function directly.
                # if cropped_image.shape[1] < 15: 
                #     continue
                # scale_factor = 0.55
                # # Calculate the new width and height after downsampling
                # new_width = int(cropped_image.shape[1] * scale_factor)
                # new_height = int(cropped_image.shape[0])
                # # Perform Bilinear interpolation to downsample the subregion
                # print("resizing x_data", dis)
                # downsampled_subregion = cv2.resize(cropped_image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
                # Paste the downscaled subregion back into the original image
                # ======================================================
                # Method 3: resize with concat and iterations:
                
                for ph in range(3):
                    l1 = int(2*cropped_image.shape[1]/10)
                    l2 = int((8*cropped_image.shape[1])/10)
                    resize_image = cropped_image[:, l1:l2]
                    left = cropped_image[:, :l1]
                    right = cropped_image[:, l2:]
                    scale_factor = 0.7- ph*0.1
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
                    if resize_image.shape[1] < 15: 
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

            
            # if cv_img2.shape[1]< 6000:
            #     # resized_img = cv2.resize(cv_img2, (cv_img.shape[1], cv_img.shape[0]), interpolation=cv2.INTER_AREA)
            #     resized_img = cv2.resize(cv_img2, (6244, 1024), interpolation=cv2.INTER_LANCZOS4)

            # else:     
            resized_img = cv2.resize(cv_img2, (cv_img.shape[1], cv_img.shape[0]), interpolation=cv2.INTER_LANCZOS4)
            
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
            cv_img = resized_img

        cv2.imwrite(output_path+img[len(path)+1:], resized_img[:,:,0])
        print("image stored.")
        # cv2.imwrite("./original.png", cv_img)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        count+=1


if __name__ == "__main__":
    main()
