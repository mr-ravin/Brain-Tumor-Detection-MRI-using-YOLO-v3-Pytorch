# Detection of Brain Tumor in MRI Images using YOLO v3 Model implemented in Pytorch

### Directory Structure:
```
|-- dataset/
|      |-- images/
|      |     |-- train/   
|      |     |-- test/
|      |
|      |-- labels/
|            |-- train/
|            |-- test/
|
|-- source_code
|      |-- yolo_configuration.json
|      |-- model.py
|      |-- dataset.py
|      |-- loss_fn.py
|      |-- utils.py
|      |-- run.py    # entry point
|
|-- weights/
|-- image_plots/ # saved images after plotting of pred bounding boxes
```

### Commands
- To Train the model
  ```
  python3 run.py --mode train --epoch 12
  ```
- To visualise predicted bboxes on Test set 
  ```
  python3 run.py --mode test
  ```
  
### Visualise
`prediction.jpg` showcases `image with ground truth and predicted bounded boxes` with resolution 416x416.

![image](https://github.com/mr-ravin/Brain-Tumor-Detection-MRI-using-YOLO-v3-Pytorch/blob/main/prediction.jpg?raw=true)


#### Resources
- Dataset: [Brain Tumor Dataset](https://www.kaggle.com/datasets/davidbroberts/brain-tumor-object-detection-datasets)
- Reference used for Yolov3 implementation: [aladdinpersson](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/object_detection/YOLOv3)
