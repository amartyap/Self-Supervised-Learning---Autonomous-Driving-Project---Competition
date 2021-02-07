Deep Learning Project submission 1:  April 28th, 2020


Team Name: LAG
Team Members: Sree Gowri Addepalli (sga297@nyu.edu)
                         : Amartya Prasad (ap5891@nyu.edu)                         
                         : Sree Lakshmi Addepalli(sla410@nyu.edu)



Tasks: 

1. Road Map prediction:  used it as a classification supervised task using Resnet18 to get a baseline. (This we got currently as 0.8893 on validation set.)
2. Object Detection: Regression Problem using faster RCNN.  (This even though the coding is done, we are working on improving accuracy due to camera intrinsics, and bounding box relative coordinates) ( This is currently is zero)



Project Description and how to run:

The file consists:

1. run_test.py (The main file for running)
2. Baseline1.pth (Consisting of state_dicts of both models)
3. model_loader.py
4. helper.py
5. data_helper.py (please use our file)
6. 2 model file (FasterRCNN, resnet18)
7. 2 main files for training these models.
8 5 additional python helper files (coco_eval,  coco_utils, utils, transforms, engine).
9. explore_the_data.ipynb.
10. Data folder (Unzip student_data.zip)
11. Readme.md




a) Please ensure to use numpy version less than 1.17.4.
b) All the environment libraries we have used to run our files are in requirement.txt.
c) We have tested these models on k80 GPU. For execution it takes some time (5mins) (It is not stuck, the object detection part is heavy).


For any issues regarding running, please contact sga297@nyu.edu.



Future Tasks:

1. Semi and Self supervision through SIMCLR on unlabelled dataset.
2. Camera Intrinsics
3. Ensembling various Self supervision task to build robust models and combine last few layers of supervised models like DeepLabv3 and FrrnB for roadMap and CascadeRCNN for object detection . (If time permits)


Failed Attempts:

1. Object detection IoU and camera intrinsics.
2. Running FrrnB and DeepLabv3 on our models (Cuda memory issues)



If you have any suggestions on any improvements we should work on please mail the above.



    
