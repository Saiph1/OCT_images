# OCT_images
OCT image application

1. clone this repo and the Soft-Teacher repo:
```
git clone https://github.com/microsoft/SoftTeacher.git
```
2. pip install required library for Soft-Teacher:
```
// To do
```
3. Download the dataset via:


4. Create a data directory for further processing. 
```
mkdir data
```

Notes:

- Train the model using the mmdetection library with Soft teacher (semi-supervised learning): 
    1. download the dataset
    2. modify the config of your choice. 
    config file1 (baseline): SoftTeacher/configs/soft_teacher/base.py
    config file2: SoftTeacher/configs/soft_teacher/soft_teacher_faster_rcnn_r50_caffe_fpn_coco_180k.py
    *** note that config file 2 imports baseline module.
    3. train with 
    ``` 
    cd SoftTeacher 
    bash tools/dist_train_partially.sh semi 1 5 1
    ```
- Speckle equalization still needs to be improved.