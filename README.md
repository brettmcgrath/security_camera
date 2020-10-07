## Table of Contents  
- [Project Contents](#Project_Contents)
- [Data](#Data)  
- [Data Problem](#Data_Problem)  
- [Data Cleaning and Exploratory Data Analysis](#Data_Cleaning_and_Exploratory_Data_Analysis)  
- [Preprocessing and Modeling](#Preprocessing_and_Modeling)  
- [Reflections on the modeling process](#Reflections_on_the_modeling_process)  

## Project Contents:

The attached file contains, for now, one Jupyter notebook.
1. **EDA_and_img_aug.ipynb** - This file contains the exploratory data analysis and image augmentation of my security camera project. The first part of this notebook details image size, number of images, train/test split amounts, and information regarding data that was removed from the original image dataset. Images were augmented to increase size of training data.

## Data

Data used in this project can be seen on a University of Grenada handgun image dataset listed on [Kaggle](https://www.kaggle.com/andrewmvd/handgun-detection). Annotations from this dataset were converted from COCO ([x_min, y_min, width, height]) to YOLO (normalized[x_center, y_center, width, height]) format for future use of modeling with the [YOLO v5 model](https://github.com/ultralytics/yolov5) architecture.

## Data_Problem

Through use of a convolutional neural network through YOLO v5, I intend to create a model that can be deployable on both still photos and live video to detect the presence of handguns, minimizing false positives on other objects. My aim is for this to be a part of a larger future project of threat detection in video surveillance. My project at present aims to detect the presence of handguns, but future ideas for this project include, but are not limited to: \
• Detection of long guns \
• Distinguishing between drawn and holstered weapons \
• Identifying police uniforms to rule out police posession of weapons \
• Detecting fire  \ 
• Detecting flood waters \
• Detection of broken glass and forced entry \
 \
Upon deployment this project will be immediately useful to the detection of unwanted firearms in places where they are forbidden such as schools and hospitals. Use of this or similar deployed models could be used to draw attention to or instantly alert authorities to dangerous or potentially life threatening situations and hopefully minimize their impact. 
 
## Data_Cleaning_and_Exploratory_Data_Analysis

A large amount of time was spent physically sifting through every photo of the 2986 image dataset. 444 images were removed for various reasons. 34 were removed because they were holstered and will hopefully form part of a future dataset on holstered weapons in hopes to distinguish between weapons in a holster and those drawn. 26 were removed because they depicted non-firearm objects or were (as was the case of the majority of these images) long guns. 384 images were removed for many different reasons, including but not limited to: \
• Image depicts a cartoon weapon \
• Image is from a video game \
• Watermark too prominent \
• Poor image quality \
• Image is barely visible \
• Technical drawing \
• Wood or toy gun \
 \
Images were train/test split for modeling purposes, and training images were augmented at random to increase the 2,022 training image pool to add 8,371 augmented images for a total of 10,393. Images were augmented through a series of tranformations using the [albumentations](https://albumentations.ai/) Python package. The following transformations were made to photos during the augmentation process along with descriptions and the corresponding probability that this augmentation occured on any random image generation: \
• Flip: Flip the input either horizontally, vertically or both horizontally and vertically - 20% \
• RandomBrightnessContrast: Randomly change brightness and contrast of the input image - 20% \
• GaussNoise: Apply gaussian noise to the input image - 20% \
• HorizontalFlip: Flip the input horizontally - 30% \
• RandomSnow: Bleach out some pixel values simulating snow - 20% \
• Downscale: Decreases image quality by downscaling and upscaling back - 20% \
• HueSaturationValue: Randomly change hue, saturation and value of the input image - 30% \
• ISONoise: Apply camera sensor noise - 30% \
• InvertImg: Invert the input image by subtracting pixel values from 255 - 20% \
• MotionBlur: Apply motion blur to the input image using a random-sized kernel - 20% \
• RGBShift: Randomly shift values for each channel of the input RGB image - 20% \
• Rotate: Rotate the input by an angle selected randomly from the uniform distribution - 100% \
• ToGray: Convert the input RGB image to grayscale. If the mean pixel value for the resulting image is greater than 127, invert the resulting • grayscale image - 10% \
 \
To see how these images transformations look, please see the EDA_and_img_aug jupyter notebook attached, or visit the [albumentations demo site](https://albumentations-demo.herokuapp.com/).


## Preprocessing_and_Modeling

This is where I currently find myself. I have begun the process of migrating data and the first test model to the Microsoft Azure Machine Learning studio environment. 

## Reflections_on_the_modeling_process

TBD

## Conclusions_and_Recommendations

TBD
