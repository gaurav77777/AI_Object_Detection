# AI_Object_Detection


# Prerequisite 


Download required file:

Download YOLOv3 weights file
!wget https://pjreddie.com/media/files/yolov3.weights

Download YOLOv3 config file
!wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg -O yolov3.cfg

Download coco.names (class labels for the objects YOLO can detect)
!wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names


File :

yolov3.weights: The YOLO model weights.
yolov3.cfg: The YOLO model configuration.
coco.names: The class labels file.
Test Image: The image you want to test the object detection on.


# Deployment

Use google colab to deploy object detection application

Step:

1. create workspace in google colab
2. import all required file in workspace
3. copy code of object_detection.py in a cell
4. run the cell to get output
