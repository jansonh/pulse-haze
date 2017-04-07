# Inferring the Level of Visibility from Hazy Images

This repository contains our team's deep learning implementation to solve the problem of Inference of visibility levels from images of haze in Sumatra island. This is our work from the Research Dive: Image Mining for Disaster Management event organized by United Nations Pulse Lab Jakarta (http://unglobalpulse.org/image-mining-disaster-management).

The technical report of our work is available at https://issuu.com/pulselabjakarta/docs/tech_report_2_v8.

### Our Deep Learning Approach
In the report, we explained various approaches to solve the problem. In the learning-based approach, we proposed a framework inspired by Fayao Liu's work (https://bitbucket.org/fayao/dcnf-fcsp) utilizing a deep convolutional neural fields to detect the depth map of an image and a dark channel prior for the transmission matrix. Both depth map and transmission matrix are then used to estimate the haze level.

This repository stores the code of our other deep learning-based model. Our second learning-based approach used a simple convolutional neural network (CNN). The CNN is basically consists of 3 layers of Conv + ReLU + MaxPool and 3 fully-connected layers. The output of the CNN is the haze level (heavy or light haze).

### Dataset
The dataset of haze images is not publicly available at this time. The images are collected by Pulse Lab Jakarta from social media by querying on related keywords and hashtags. To train and validate the CNN model, we used 357 manually-labeled images which are categorized into two classes. As the dataset size is so small, we use various image augmentation technique in our code.

### Requirements
1. Python 2.7
2. Numpy
3. TensorFlow
4. Keras
