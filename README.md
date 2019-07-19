# Restoration-of-damaged-images-using-deep-learning

Pytorch implementation of a deteriorated image restoration algorithm based on “[Deep Image Prior](https://dmitryulyanov.github.io/deep_image_prior)” research paper.

## Introduction

When an image has deteriorated, that is to say that some of the visual content is lost, it is sometimes important to perform a restoration step "inpainting". This step usually involves recreating visual information at damaged locations (see Figure 1) so that a human being can not say that it is a deteriorated image that has been restored.
The objective of this project is to implement in python a deteriorated image restoration algorithm using an approach based on neural networks.

Generally in Deep Learning, you have to adjust 2 things:
- the dataset on which the network will train.
- the structure of the neuron network.

One can naively think that without a dataset, we can't train a neural network. This project of image restoration proves the opposite.
Inspired by the works of researchers Dmitry Ulyanov, Andrea Vedaldi and Victor Lempitsky in "Deep Image Prior", I came up with the concluson that a neuron network without a dataset and with a good structure can produce better predictions than a poorly structured neural network with a large training dataset.

That's why i will not use a training dataset here. The results will only be due to the structure of the network.

