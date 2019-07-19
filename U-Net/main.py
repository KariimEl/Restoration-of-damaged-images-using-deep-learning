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
import inpainting_functions_RW as ifrw
import inpainting_network as i_n

#parametrage du processeur a utiliser
device = torch.device('cuda')
device_cpu = torch.device('cpu')
torch.cuda.empty_cache()

#parametrage image d'origine
img_origin_path  = 'test.jpg'
img_origin_reshape_path = 'reshape_img.jpg'
mask_path = 'masque.jpg'
mask_reshape_path = 'masque_reshape.jpg'

#parametrage image de sortie du reseau
nom_dossier = 'dossier\_'
nom_fichier = 'resultat_en_cours_'
extention = '.jpg'

#chargement de l'imae d'origine
img_origin = mpimg.imread(img_origin_path);
[W_o,H_o,M_o] = img_origin.shape

#creation dune image plus grande
ifrw.save_reshape_img(img_origin_path, img_origin_reshape_path)#create and save the reshape image
img_origin_reshape = mpimg.imread(img_origin_reshape_path)#load the reshape image

ifrw.save_reshape_img(mask_path, mask_reshape_path)
img_origin_reshape = mpimg.imread(mask_reshape_path)

[W,H,M] = img_origin_reshape.shape



#adaptation de l'image trouee et masque
PLOT = True
imsize = -1
dim_div_by = 64

img_pil, img_np = ifrw.get_image(img_origin_reshape_path, imsize)
img_mask_pil, img_mask_np = ifrw.get_image(mask_reshape_path, imsize)

img_mask_pil = ift.crop_image(img_mask_pil, dim_div_by)
img_pil      = ift.crop_image(img_pil,      dim_div_by)

img_np      = ifc.pil_to_np(img_pil)
img_mask_np = ifc.pil_to_np(img_mask_pil)

img_mask_var = ifc.np_to_torch(img_mask_np)
ifrw.plot_image_grid([img_np, img_mask_np, img_mask_np*img_np], 3,11);


#creation du reseau de neuronne
model = i_n.Net()
model = model.float()


#creation de l'image aleatoire
img_gen = np.random.uniform(0,0.1, (32, W, H))
np.save("image_generee",img_gen)
img_gen = torch.from_numpy(img_gen)
img_gen = img_gen.unsqueeze(0)
img_gen = img_gen.type(torch.FloatTensor)



# Loss
optimizer = optim.Adam(model.parameters(), lr=0.1)

mse = torch.nn.MSELoss()
n_epochs = 5000
img_var = ifc.np_to_torch(img_np)
mask_var = ifc.np_to_torch(img_mask_np)

model = model.to(device)
img_gen = img_gen.to(device)
img_var = img_var.to(device)
mask_var = mask_var.to(device)

train_losses = []
for epoch in range(0, n_epochs+1):

    ###################
    # train the model #
    ###################
    model.train()
    # clear the gradients of all optimized variables
    optimizer.zero_grad()
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(img_gen)
    # calculate the batch loss
    loss = mse(output * mask_var, img_var * mask_var)
    # backward pass: compute gradient of the loss with respect to model parameters
    loss.backward()
    # perform a single optimization step (parameter update)
    optimizer.step()
    train_losses.append(loss.item())

    #affichage et enregistrement image
    if ((epoch%500) == 0):
        outputNP = ifc.torch_to_np(output)
        outputPIL = ifc.np_to_pil(outputNP[:, 0:W_o, 0:H_o])

        plt.figure(2)
        plt.imshow(outputPIL)
        plt.pause(0.01)
        #sauvegarde image
        outputPIL.save(nom_dossier + nom_fichier + str(epoch) +extention)
        
        #affichage de la courbe de loss
        plt.plot(train_losses, label='Training loss')
        plt.ylabel('Training loss')
        plt.xlabel("Iteration number")
        plt.savefig(nom_dossier + 'train_loss.jpg')
        torch.save(train_losses, nom_dossier + 'loss.pt')

    #affiche le loss
    if ((epoch%20) == 0):
        print('iteration : ', epoch,'/',n_epochs, '--  loss : ', loss.item())

    torch.cuda.empty_cache()


torch.save(model, 'model_inpaintingV1.pt')
model = model.to(device_cpu)



data = np.load("image_generee.npy")
data = torch.from_numpy(data)
data = data.unsqueeze(0)
data = data.float()
out =  model(data)
out_np = ifc.torch_to_np(out)
ifrw.plot_image_grid([out_np], factor=5)

#plot training loss function
plt.plot(train_losses, label='Training loss')
plt.ylabel('Training loss')
plt.xlabel("Iteration number")
plt.savefig(nom_dossier + 'train_loss.jpg')

torch.cuda.empty_cache()

outputNP = ifc.torch_to_np(output)
outputPIL = ifc.np_to_pil(outputNP)
plt.figure(3)
plt.imshow(outputPIL)
plt.pause(0.1)

n = 1
nom_fichier = 'resultat'
numero_fichier = str(n)
extention = '.jpg'
outputPIL.save(nom_fichier + numero_fichier +extention)