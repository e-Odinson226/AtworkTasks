PPT (Precision Placement Tables)

...
23 March 2023:
TODO:   1. Try to config only color filters.            [DONE]
        2. Add post-processing.                         [DONE]
        3. Align depth and color images.                [DONE]
        4. Calculate weight on ROIs.                    [DONE]
...



1. Create a dataset of images of cavities classified into 10 classes.

2. Preprocess the dataset by performing operations like resizing, normalization, and data augmentation.

3. Split the dataset into training and testing sets.

4. Use TensorFlow's deep learning models like Convolutional Neural Networks (CNNs) or Transfer Learning models like ResNet, Inception, or VGG to train on the dataset.

5. Evaluate the models' performance by predicting labels for the test set and computing accuracy, precision, recall, F1 scores, and confusion matrix.

6. Fine-tune the model by adjusting hyperparameters like learning rate, batch size, number of epochs, optimizer, and regularization.

7. Use OpenCV to load and preprocess new images, and then pass them to the trained model for classification.

---

![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Logo Title Text 1")

---

## **Results analysis:**
> ### **bounding-box detector** *v.1* : /bbox_classification/boundingBox_detector_ros.py

| Model | F20_20_horizontal | M20_30_horizontal | M20_30_vertical | M20_100_horizontal | R20_horizontal | R20_vertical | S40_40_horizontal | S40_40_virtical |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |  
| **16-04-2023_17-59-15.h5** | NA | NA | NA | NA | NA | NA | NA | NA |
| **classification_VGG16.h5** | NA | NA | NA | NA | NA | NA | NA | NA |
| **EfficientNetV2B0_v1.h5** | NA | NA | NA | NA | NA | NA | NA | NA |
| **EfficientNetV2B0_v2.h5** | NA | NA | NA | NA | NA | NA | NA | NA |
| **MobileNetV2_v2.h5** | NA | NA | NA | NA | NA | NA | NA | NA |
| **Pretrained_EfficientNetB0.h5** | NA | NA | NA | NA | NA | NA | NA | NA |
| **VGG16_v1.h5** | NA | NA | NA | NA | NA | NA | NA | NA |
| **VGG16_v2.h5** | NA | NA | NA | NA | NA | NA | NA | NA |

 
 
 
 
 
 
 
 


