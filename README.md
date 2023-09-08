# Getting started:

1. Environments/Library setup: 
```
conda create --name oct_image python=3.7
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
3. Download the training/val dataset via [Here](https://pan.baidu.com/s/1bSFuoaaKyJssQ2kwvTk3-Q?pwd=vyax)

Code: vyax

4. Extract data and make sure the route looks like:
```
# SoftTeacher/
#  data/
#   oct/
#    train/
#       *.png
#         .
#         .
#         .
#    annotations/
#       semi/
#         train_val.json
#         train_label.json
#         unlabel.json
```

# Training:

- Train the model using the mmdetection library with Soft teacher (semi-supervised learning): 
    1. Modify the config of your choices.
       
    config file1 (baseline): SoftTeacher/configs/soft_teacher/base.py
    - img_scale (that matches the GPU memory)
    - cancel random flip
    - scale in x direction
    - keep ratio = false
    - ...
  
    config file2: SoftTeacher/configs/soft_teacher/soft_teacher_faster_rcnn_r50_caffe_fpn_coco_180k.py
    ```
    CLASSES = ['d1','d2']
    data = dict(
        samples_per_gpu=5,
        workers_per_gpu=5,
        train=dict(
            sup=dict(
                type="CocoDataset",
                ann_file="data/oct/annotations/semi/train_label.json",
                img_prefix="data/oct/train/",
                classes=CLASSES,
            ),
            unsup=dict(
                type="CocoDataset",
                ann_file="data/oct/annotations/semi/unlabel.json",
                img_prefix="data/oct/train/",
                classes=CLASSES,
            ),
        ),
        val=dict(
            ann_file="data/oct/annotations/semi/train_val.json",
            img_prefix="data/oct/train/",
            classes=CLASSES,
        ),
    
        sampler=dict(
            train=dict(
                sample_ratio=[1, 4],
            )
        ),
    )
    #    .
    #    .
    #    .
    ```
    2. train with

    ``` 
    cd SoftTeacher/ 
    bash tools/dist_train_partially.sh semi 1 5 1
    ```
    The results and checkpoints will be stored at SoftTeacher/work_dir/{name of the training session}
  
    3. Visualize the training log result with (example at /oct_images):
       
    ```
    python analyze_logs.py plot_curve log/baseline.json --keys sup_acc unsup_acc
    ```

    4. To test the model performance in AR/AP metrics: 
    ```
    bash tools/dist_test.sh work_dirs/{project name}/5/1/{config name}.py work_dirs/{project name}/5/1/latest.pth 1 --eval proposal --cfg-options model.test_cfg.rcnn.score_thr=0.5
    ```
    
# Image post-processing: 
- Create a data directory for further processing. 
```
mkdir data && mkdir output
```
- Iterative restoration: 
    1. Modify path to trained model and config file in fix.py : 
    ```
    config_path = "./SoftTeacher/work_dir/{name of the training session}/{config file name}.py"
    model_path = "./SoftTeacher/work_dir/{name of the training session}/iter_{latest model}.pth"
    ```
    2. run command: 
    ```
    python fix.py
    ``` 
    3. The output will be the repair version of original images presented in data folder

- Concatenate and redistribute oct content with struts positions:
    1. Modify path to trained model and config file:
    ```
    config_path = "./config.py"
    model_path = "./iter_124000.pth"
    ```
    The above files could be downloaded from the baidu cloud link [Here.](https://pan.baidu.com/s/1bSFuoaaKyJssQ2kwvTk3-Q?pwd=vyax)

    2. create new folder to store the processed images:
    ```
    mkdir rescale
    ```
    3. run with:
    ```
    python cat2.py
    ```

# To Do list: 
- Speckle equalization
