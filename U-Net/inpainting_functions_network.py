import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torchvision import transforms
from PIL import Image


import inpainting_functions_conversion as ifc
import inpainting_functions_treatment as ift


def up_model(num_input, num_up=128, filter_up_size=3):
	up_model = nn.Sequential(nn.BatchNorm2d(num_input),
							 nn.Conv2d(num_input, num_up, filter_up_size, padding=1),
							 nn.BatchNorm2d(num_up),
							 nn.LeakyReLU(),
                             #nn.Upsample(scale_factor=2, mode='nearest'),
							 nn.Conv2d(num_up, num_up, filter_up_size, padding=1),
							 nn.BatchNorm2d(num_up),
							 nn.LeakyReLU())
	return up_model

def down_model(num_input, num_down, filter_down_size):
	down_model = nn.Sequential( nn.Conv2d(num_input, num_down, filter_down_size, padding=1, stride=2),
							   nn.BatchNorm2d(num_down),
							   nn.LeakyReLU(),
							   nn.Conv2d(num_down, num_down, filter_down_size, padding=1),
							   nn.BatchNorm2d(num_down),
							   nn.LeakyReLU())
	return down_model


def skip(num_input, num_skip=4, filter_skip_size=1):
	skip = nn.Sequential(nn.Conv2d(num_input, num_skip, filter_skip_size, padding=0),
    					 nn.BatchNorm2d(num_skip),
						 nn.LeakyReLU())
	return skip


def output_(num_input=128, num_skip=3, filter_skip_size=1):
	output = nn.Sequential(nn.Conv2d(num_input, num_skip, filter_skip_size, padding=0))
	return output
