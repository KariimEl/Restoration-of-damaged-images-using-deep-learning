import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torchvision import transforms

import inpainting_functions_treatment as ift
import inpainting_functions_network as ifn

TAILLE_NETWORK = 5

##definition des filtres
#down
nd = [128, 128, 128, 128, 128]#nb filtre = taille sortie
nd_in= [32, 128, 128, 128, 128]#taille entree
kd = [3, 3, 3, 3, 3]#taille du filtre

#up
nu = [128, 128, 128, 128, 128]#nb filtre = taille sortie
nu_in = [132,132, 132, 132, 4]#taille entree
ku = [3, 3, 3, 3, 3]#taille du filtre

#skip
ns = [4, 4, 4, 4, 4]#nb filtre = taille sortie
ns_in = [128, 128, 128, 128, 128]#taille entree
ks = [1, 1, 1, 1, 1]#taille du filtre

## definition du reseau de neuronne
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        ##declaration des tableaux qui vont contenir les filtres
        self.down = []
        self.skip = []
        self.up = []
        self.output_ = ifn.output_()

        ##definition des filtres
        for i in range (0, TAILLE_NETWORK):
            self.down.append(ifn.down_model(nd_in[i], nd[i], kd[i]))
            self.skip.append(ifn.skip(ns_in[i], ns[i], ks[i]))
            self.up.append(ifn.up_model(nu_in[i], nu[i], ku[i]))

            self.add_module('down_{}'.format(i), self.down[i])
            self.add_module('up_{}'.format(i), self.up[i])
            self.add_module('skip_{}'.format(i), self.skip[i])

    def forward(self, x):
        #print('start ', x.size())

        x = self.down[0](x)
        #print('d0 ',x.size())
        x0 = self.skip[0](x)
        #print('s0 ',x1.size())
        x = self.down[1](x)
        #print('d1 ',x.size())
        x1 = self.skip[1](x)
        #print('s1 ',x.size())
        x = self.down[2](x)
        #print('d2 ',x.size())
        x2 = self.skip[2](x)
        #print('s2 ',x.size())
        x = self.down[3](x)
        #print('d3 ',x.size())
        x3 = self.skip[3](x)
        #print('s3 ',x.size())


        x = self.down[4](x)
        #print('d4 ',x.size())
        x = self.skip[4](x)
        #print('s4 ',x.size())


        x = self.up[4](x)
        #print('u4 ',x.size())
        x = F.interpolate(x,scale_factor = 2, mode='nearest')
        #print('interpolate3 ',x.size())
        x =  torch.cat((x,x3),1)
        #print('Concatetation ', x.size())

        x = self.up[3](x)
        #print('u2 ',x.size())
        x = F.interpolate(x,scale_factor = 2, mode='nearest')
        #print('interpolate1 ',x.size())
        x =  torch.cat((x,x2),1)
        #print('Concatetation ', x.size())

        x = self.up[2](x)
        #print('u2 ',x.size())
        x = F.interpolate(x,scale_factor = 2, mode='nearest')
        #print('interpolate1 ',x.size())
        x =  torch.cat((x,x1),1)
        #print('Concatetation ', x.size())

        x = self.up[1](x)
        #print('u1 ',x.size())
        x = F.interpolate(x,scale_factor = 2, mode='nearest')
        #print('interpolate1 ',x.size())
        x =  torch.cat((x,x0),1)
        #print('Concatetation ', x.size())

        x = self.up[0](x)
        #print('u2 ', x.size())
        x = F.interpolate(x,scale_factor = 2, mode='nearest')
        #print('interpolate0 ',x.size())

        x = self.output_(x)
        #print('output ', x.size())

        return x
