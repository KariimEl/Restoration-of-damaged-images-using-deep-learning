import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torchvision import transforms
from PIL import Image

nom_dossier = 'dossier\_'

import inpainting_functions_conversion as ifc
import inpainting_functions_treatment as ift
import inpainting_functions_RW as ifrw
import inpainting_network as i_n

train_losses = torch.load(nom_dossier +'loss.pt')

#plot training loss function
plt.plot(train_losses, label='Training loss')
plt.ylabel('Training loss')
plt.xlabel("Iteration number")
plt.axis([0,8000,0,0.005])
plt.savefig(nom_dossier + 'train_loss.jpg')





#mpimg.imsave('image_finale.jpg', outputPIL)

#print(z.shape)
#print(output.shape)

