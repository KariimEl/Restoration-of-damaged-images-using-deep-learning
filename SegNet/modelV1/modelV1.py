# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 15:01:20 2019

@author: abdel
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 14:18:47 2019

@author: abdel
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch.optim as optim
from PIL import Image
import PIL

import HelpFunctions as myHelp



def up_model(num_input, num_up, filter_up_size=5):
    
    up_model = nn.Sequential(nn.BatchNorm2d(num_input),
                             nn.Conv2d(num_input, num_up, filter_up_size, padding=2),
                             nn.BatchNorm2d(num_up),
                             nn.LeakyReLU(),
                             nn.Conv2d(num_up, num_up, filter_up_size, padding=2),
                             nn.BatchNorm2d(num_up),
                             nn.LeakyReLU())

    return up_model

def down_model(num_input, num_down, filter_down_size):
    
    down_model = nn.Sequential( nn.Conv2d(num_input, num_down, filter_down_size,stride=2,padding=1),
                                nn.BatchNorm2d(num_down),
                                nn.LeakyReLU(),
                                nn.Conv2d(num_down, num_down, filter_down_size,padding=1),
                                nn.BatchNorm2d(num_down),
                                nn.LeakyReLU())
    return down_model

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.down1 = down_model(32, 16, 3)
        self.down2 = down_model(16, 32, 3)
        self.down3 = down_model(32, 64, 3)
        self.down4 = down_model(64, 128, 3)
        self.down5 = down_model(128, 128, 3)
        self.down6 = down_model(128, 128, 3)
                                     
        self.up6 = up_model(128, 128, 5)
        self.up5 = up_model(128, 128, 5)
        self.up4 = up_model(128, 128, 5) 
        self.up3 = up_model(128, 64, 5)
        self.up2 = up_model(64, 32, 5)
        self.up1 = up_model(32, 16, 5)   
        self.out = nn.Conv2d(16,3,1)
        self.sig = nn.Sigmoid()
        
    def forward(self, x):
        
        x = self.down1(x)
        x = self.down2(x)   
        x = self.down3(x)
        x = self.down4(x)     
        x = self.down5(x)
        x = self.down6(x)
        x = self.up6(x)
        x = F.interpolate(x,scale_factor = 2, mode='nearest')
        x = self.up5(x)
        x = F.interpolate(x,scale_factor = 2, mode='nearest')        
        x = self.up4(x)
        x = F.interpolate(x,scale_factor = 2, mode='nearest')
        x = self.up3(x)
        x = F.interpolate(x,scale_factor = 2, mode='nearest')
        x = self.up2(x)
        x = F.interpolate(x,scale_factor = 2, mode='nearest')
        x = self.up1(x)
        x = F.interpolate(x,scale_factor = 2, mode='nearest')
        x = self.out(x)
        x = self.sig(x)
        return x
 


PLOT = True
imsize = -1
dim_div_by = 64
img_path  = 'test.jpg'
mask_path = 'test_mask.jpg'


    
img_pil, img_np = myHelp.get_image(img_path, imsize)
img_mask_pil, img_mask_np = myHelp.get_image(mask_path, imsize)

img_mask_pil = myHelp.crop_image(img_mask_pil, dim_div_by)
img_pil      = myHelp.crop_image(img_pil,      dim_div_by)

img_np      = myHelp.pil_to_np(img_pil)
img_mask_np = myHelp.pil_to_np(img_mask_pil)

img_inp = img_np*img_mask_np 
img_inp = myHelp.np_to_pil(img_inp)
img_inp.save("image_masquee.jpg")


plt.figure(1)
plt.imshow(img_inp)

  
model = Net()
model = model.float()
 
[M,H,W] = img_np.shape

img = np.random.uniform(0,0.1, (32, H, W))
np.save("input",img)
img = torch.from_numpy(img)
img = img.unsqueeze(0)
img = img.float()


# Loss
optimizer = optim.Adam(model.parameters(), lr=0.1)        

mse = torch.nn.MSELoss()
n_epochs = 6000
img_var = myHelp.np_to_torch(img_np)
mask_var = myHelp.np_to_torch(img_mask_np)

model = model.cuda()
img = img.cuda()
img_var = img_var.cuda()
mask_var = mask_var.cuda()
train_loss = 0.0 
train_losses = []
for epoch in range(1, n_epochs+1):

    ###################
    # train the model #
    ###################    
    model.train() 
    # clear the gradients of all optimized variables
    optimizer.zero_grad()
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(img)
    # calculate the batch loss
    loss = mse(output * mask_var, img_var * mask_var)
    # backward pass: compute gradient of the loss with respect to model parameters
    loss.backward()
    # perform a single optimization step (parameter update)
    optimizer.step()      
    if (epoch%600 == 0):
        out_np = myHelp.torch_to_np(output)
        out_pil = myHelp.np_to_pil(out_np) 
        out_pil.save("test"+str(epoch)+".jpg")

    # update training loss
    train_loss = loss.item()*img.size(0)
    train_losses.append(train_loss)
    if (epoch%50 == 0):
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
    torch.cuda.empty_cache()

    
torch.save(model, 'model_inpaintingV1.pt')

model = torch.load('model_inpaintingV1.pt')
plt.figure(2)
plt.plot(train_losses, label='Training loss')
plt.ylabel('Training loss')
plt.xlabel("Iteration number")
plt.savefig('train_loss.jpg')
data = np.load("input.npy")
data = torch.from_numpy(data)
data = data.unsqueeze(0)
data = data.float()
data = data.cuda()
out =  model(data)
out_np = myHelp.torch_to_np(out)
out_pil = myHelp.np_to_pil(out_np)

plt.figure(3)
plt.imshow(out_pil)
out_pil.save("model_resultat.jpg")








