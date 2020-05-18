# Restoration-of-damaged-images-using-deep-learning

## Overview
Academic project oriented by Prof. Guillaume Bourmaud at Bordeaux INP - ENSEIRB MATMECA [Bordeaux Graduate School of Engineering]:    

Pytorch implementation of a deteriorated image restoration algorithm based on “[Deep Image Prior](https://dmitryulyanov.github.io/deep_image_prior)” research paper.

![image](./Figures/Inpainting.PNG)

When an image has deteriorated, that is to say that some of the visual content is lost, it is sometimes important to perform a restoration step "inpainting". This step usually involves recreating visual information at damaged locations so that a human being can not say that it is a deteriorated image that has been restored.
The objective of this project is to implement in Pytorch a deteriorated image restoration algorithm using an approach based on neural networks.

## Method

### Introduction

Deep networks are applied to image restoration by learning generator networks that map a random initial image x to an image x0. This approach can be used to solve inverse problems such as denoising and super-resolution. In our case, we use it for inpainting.

To carry out this project, we put into play 4 images:
- The original image: it is the unaltered image, with no missing information. We will never give this image to the neural network. We used it to compare visually with the output of the neural network. We also used it in our program to create the deteriorated image. 
<img src="./Figures/Original_img.png" width=300>
- The damaged image: This is the image for which a part of information is missing. 
<img src="./Figures/damaged_img.png" width=300>
- The input image: This is the input image of the model initialied randomly.
<img src="./Figures/initial_img.png" width=300>
- The output image: This image is produced by the model when given the initial random input image.
<img src="./Figures/output_img.png" width=300>

In this project, we have implemented a U-Net type “hourglass” architecture as in the research paper and also a Seg-Net architecture to compare the results of both architectures.

### U-Net architecture
<img src="./Figures/U_net.png">

### Seg-Net architecture
<img src="./Figures/Seg_net.png">

### Results

#### Image of a landscape
The damaged image: 

<img src="./Figures/damaged_img_landscape.jpg" width=300>

The output image of U-Net: 

<img src="./Figures/output_img_landscape_U_net.jpg" width=300>

The output image of Seg-Net: 

<img src="./Figures/output_img_landscape_Seg_net.jpg" width=300>

The original image: 

<img src="./Figures/original_img_landscape.jpg" width=300>


#### Image of a face
The damaged image: <img src="./Figures/damaged_img_face.png">
The output image of U-Net: <img src="./Figures/output_img_face_U_net.png">
The output image of Seg-Net: <img src="./Figures/output_img_face_Seg_net.png">
The original image: <img src="./Figures/original_img_face.png">


## Contributors, Contact and License

Abdelkarim ELASSAM,  2019  
abdelkarim.elassam@enseirb-matmeca.fr  
ENSEIRB-MATMECA (Bordeaux INP), Electronic Engineering - Signal and Image Processing

Augustin HUET,  2019  
augustin.huet@enseirb-matmeca.fr  
ENSEIRB-MATMECA (Bordeaux INP), Electronic Engineering - Embedded Systems
 

This code is free to use, share and modify for any non-commercial purposes.  
Any commercial use is strictly prohibited without the authors' consent.

