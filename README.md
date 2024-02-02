<H1> DSAIE TU DELFT PLASTIC PROJECT WITH NORIA SUSTAINABLE INNOVATORS </H1>


Created on Thu Feb  1 18:14:31 2024

@author: Henry Gunawan, Dimitris Mantas, Sam Parijs




## Documentation
In this project, there are 10 classes of garbage being trained with YOLOv8 model based on the 10 most found OSPAR categories. There are:
0: caps_and_lids
1: cigarette_butts
2: drinking_cans
3: food_packaging
4: food_wrapping
5: pet_bottles
6: plastic_cups
7: plastic_films
8: styrofoam
9: undef_hard_plastic

To reduce the quality of data annotating since OSPAR categories since there is no clear lines between each classes, we created a "Data Labelling Guidelines".
This Guidelines can be accessed in ("Documentation/Data Labelling Guidelines.docx") 

General Images and Videos taken during the project can be accessed in General Images Folder.

For all of the data, images, video, code, and documentation related to this project can be accessed in:
[`TU DELFT ACCESS`](https://tud365.sharepoint.com/:f:/s/PLASTIC/Eul0m8qcwnRPjb7IiBlr9KoBkV4chwi4yFUYqnuNLxtnDg?e=hcoxF4)







## Trained Model
In this project, there are 3 yolov8 models trained:
1. yolov8l ("Fix Model/yolov8l/") 
mAP50-95: 0.724
mAP50: 0.918

2. yolov8m ("Fix Model/yolov8m/")
mAP50-95: 0.728
mAP50: 0.922

3. yolov8s ("Fix Model/yolov8s/")
mAP50-95: 0.732
mAP50: 0.926

Inside the folder of these models consist of the best weights, detail of training such as confusion matric, data batch, arguments, and results.




## Training and Running Code
Code produced includes:
1. main_code in Python ("Python Code/main_code.py")
This code is the main code used during object detection and counting. Consist of 'path to directory', choice of using camera or video as input, define counting line, and stop the computer vision when pressing ESC.
This code will produces 2 output:
- Processed video
- CSV consist of "Class", "Class Name", "Count", and "Confidence".
Count is the object counted, while the confidence is the average confidence per class.

2. oject_counter in Python ("Python Code/object_counter.py")
This code include all of the class and funtion needed on the main code. This include setting counting 	lines, drawing bounding box, counting objects, print result on frames, and export to csv.

3. YOLOv8 Training in Jupyter Notebook ("Python Code/YOLOv8_TRAIN.ipynb")
This code mainly used to train yolov8 model.

4. Deploy Model Roboflow in Jupyter Notebook ("Python Code/Deploy Model Roboflow.ipynb")
This is the code we usually used to deploy the model to Roboflow.


## Dataset with Annotations
At the time this file made, there are 787 dataset annotated in roboflow.
Roboflow for this project can be accessed in:
[`Roboflow`](https://app.roboflow.com/dsaie-project-plastic/dsaie-project-plastic/)


By using Roboflow, it is possible to automated the annotation. We have deployed a yolov8m model to Roboflow which trained with 1869 images and 164 images for annotation, the amount of images comes from augmentations.
The automated annotation has a model with:
mAP: 92.2%
Precision: 88.4%
Recall: 86.8%

This model can be accessed in:
[`Roboflow Automated Annotation`](https://app.roboflow.com/dsaie-project-plastic/dsaie-project-plastic/9)




In Dataset Folder ("Dataset"), consists of:
1. Prelim_Dataset
This is a dataset taken from open source. The result in training is not good, so this dataset is not used.

2. Drijfvuil detectie.v1i.yolov8 by Jip
This dataset is made by Jip during his bachelor Thesis. It consists of 6 classes and already merged with the dataset used now in the Roboflow.

3. Raw Dataset
This is the raw data taken by us which is not labelled yet.

4. Video
These videos were taken by us during the project and can be used to test the model.






## Camera Recommendation
Below is the consideration for camera recommendation:
High Resolution  Ensure quality and ability to capture small object i.e. Cigarette Butts
Polarized Lens Filter  Reduce glare from artificial lights and reflective surface
Wide Dynamic Rage  Able to handle various lighting scenarios
Direct Connectivity to Laptop  either USB, HDMI, or WIFI which is crucial for real-time monitoring
Power Source  AC adapter or USB to operate without battery concerns
Mounting Capability  Camera fixed, need to be able to mounted




Based on this Criteria, we found some camera:
1.Canon SX740: This is a great choice for 4K videos at 30p. It's really light and has Optical Image Stabilization to keep your shots steady. Also, it's got Wi-Fi and Bluetooth, so connecting it to other devices is a breeze. Price around €409.99 in canon.nl

2. Sony HX99: This compact camera is awesome for 4K videos. It comes with an 18.2 MP sensor and can zoom up to 30x, or even 60x with a special feature. Around €479.00 in sony.nl

3. GoPro HERO8: Perfect for action shots, this camera can handle 4K videos at 60 fps. It's super compact, has several video modes (including time warp and timelapse), advanced stabilization, and voice control. Price at €361.99 in bol.com

4. Logitech Brio 4K Webcam: Though primarily a webcam, it records in 4K at 30 fps and has excellent HDR support. It features autofocus, noise-canceling dual mics, and is great for streaming or video calls. Price at €155 in coolblue.nl
