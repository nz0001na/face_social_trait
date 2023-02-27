# Facial Traits Rating Prediction Using Deep Model

This repository provides the code of Facial Traits Rating Prediction and Visualization used in paper : Comprehensive social trait judgments from faces in autism spectrum disorder.

### Link: 
[[PDF]](https://europepmc.org/article/ppr/ppr537217)

This repository contains the implementation of:
* VGG16 based regression model 
* LRP Analysis


# Introduction
* People often make trait judgments about unfamiliar others based on their faces in the absence of context or other information, such as forming an impression that someone looks friendly, trustworthy, or strong
* Individuals with Autism Spectrum Disorder (ASD) often have difficulties reading social information from faces 
* In this code, we try to study how do the people with ASD make facial trait judgement

# Methods
* give a face image, participant is asked to give traiting rating value for each trait, like warm, strong, etc.
![arch](fig/data.png)

* based on the images and rating labels, we train a pre-trained vgg16 based regression model using transfer learning.

![arch](fig/model.png)
![arch](fig/transfer.png)

* Layer-wise Relevance Propagation (LRP) technique to highlight which input features the deep neural network uses to support its output.

![arch](fig/lrp.png)

[[paper]](https://arxiv.org/abs/1808.04260)
[[github]](https://github.com/albermax/innvestigate)
