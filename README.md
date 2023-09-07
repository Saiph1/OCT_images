# OCT_images
OCT image application

Getting started:

1. Install required library: 
```
conda create --name oct_image python=3.8 
conda activate oct_image
```
For CUDA>11.3 (pytorch==1.9.0 torchvision==0.10.0): 
```
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.3 -c conda-forge
```
(Please refer to https://pytorch.org/get-started/previous-versions/ for more detail)
```
make install
```
2. Create the data directory for training
```
cd SoftTeacher && mkdir data && cd data
```
3. Download the training/val dataset via:

Link: https://pan.baidu.com/s/1bSFuoaaKyJssQ2kwvTk3-Q?pwd=vyax 
Code: vyax

4. Extract data and make sure the route looks like:
```
# oct/
#  train/
#     *.png
#       .
#       .
#       .
#  annotations/
#     semi/
#       train_val.json
#       train_label.json
#       unlabel.json
```

Training:

- Train the model using the mmdetection library with Soft teacher (semi-supervised learning): 
    1. Modify the config. 
    config file1 (baseline): SoftTeacher/configs/soft_teacher/base.py
    config file2: SoftTeacher/configs/soft_teacher/soft_teacher_faster_rcnn_r50_caffe_fpn_coco_180k.py
    ```# note that config file 2 imports baseline module.
    ```
    3. train with 
    ``` 
    cd SoftTeacher/ 
    bash tools/dist_train_partially.sh semi 1 5 1
    ```
- Speckle equalization still needs to be improved.

Image post-processing: 
1. Create a data directory for further processing. 
```
mkdir data && mkdir output
```

2. Visualize the training log result with: 
```
python analyze_logs.py plot_curve log/baseline.json --keys sup_acc unsup_acc
```
