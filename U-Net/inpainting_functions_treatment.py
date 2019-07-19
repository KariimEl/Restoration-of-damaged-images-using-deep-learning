import torch

#add zeros on right and on left of the image
def add_border_zero_RL(small_M, big_M):
    state = 'right';
    T_big = big_M.shape
    T_small = small_M.shape
    Zeros_M = torch.zeros(T_small[0],T_small[1],T_small[2],1)


    while (T_small[3]<T_big[3]):

        if (state == 'right'):
            small_M = torch.cat((small_M, Zeros_M),3)
            state = 'left'
        else :
            small_M = torch.cat((Zeros_M, small_M),3)
            state = 'right'
        T_small = small_M.shape
    return small_M

#add zeros on top and on bottom of the image
def add_border_zero_TB(small_M, big_M):
    state = 'bottom';
    T_big = big_M.shape
    T_small = small_M.shape
    Zeros_M = torch.zeros(T_small[0],T_small[1],1,T_small[3])

    while (T_small[2]<T_big[2]):

        if (state == 'bottom'):
            small_M = torch.cat((small_M, Zeros_M),2)
            state = 'top'
        else :
            small_M = torch.cat((Zeros_M, small_M),2)
            state = 'bottom'
        T_small = small_M.shape
    return small_M

#add zeros until the shape of the 2 matrix are equal
def add_border_zero_around(small_M, big_M):
    small_M = add_border_zero_TB(small_M, big_M)
    small_M = add_border_zero_RL(small_M, big_M)
    return small_M

##############################################♠

#add zeros on right of the image
def add_border_zero_Rig(small_M, big_M):
    T_big = big_M.shape
    T_small = small_M.shape
    Zeros_M = torch.zeros(T_small[0],T_small[1],1, T_small[3]).type(torch.ByteTensor)

    while (T_small[2]<T_big[2]):
        small_M = torch.cat((small_M, Zeros_M),2)
        T_small = small_M.shape
    return small_M


#add zeros on bottom of the image
def add_border_zero_Bot(small_M, big_M):
    T_big = big_M.shape
    T_small = small_M.shape
    Zeros_M = torch.zeros(T_small[0],1,T_small[2], T_small[3]).type(torch.ByteTensor)
    while (T_small[1]<T_big[1]):
        small_M = torch.cat((small_M, Zeros_M),1)
        T_small = small_M.shape
    return small_M

#add zeros until the shape of the 2 matrix are equal using bottom and right concatenation
def add_border_zero_RB(small_M, H, W):
    big_M = torch.rand(1, H, W, 3)
    small_M = add_border_zero_Bot(small_M, big_M)
    small_M = add_border_zero_Rig(small_M, big_M)
    return small_M


#renvoie 2 taille de la puissance de 2 superieure
def taille_sup_binaire(small_M):
    Taille = small_M.shape
    H = Taille[1]
    W = Taille[2]

    n_h = 0
    while (2**n_h)<H:
        n_h = n_h+1
    H_bin = 2**n_h

    n_w = 0
    while (2**n_w)<W:
        n_w = n_w+1
    W_bin = 2**n_w

    return H_bin, W_bin


#renvoie 2 taille divisible par 2**N (N = taille reseau de Neuronne)
def taille_sup_diviseur(small_M, N=5):
    Taille = small_M.shape
    H = Taille[1]
    W = Taille[2]

    while ( H%(2**N) != 0 ):
        H = H+1

    while ( W%(2**N) != 0 ):
        W = W+1

    return H, W

#add zeros at bottom and right until the size can be devide N times by 2
def add_border_zero(small_M, N=6, methode='diviseur_sup'):
    if (methode == 'diviseur_sup'):
        H, W = taille_sup_diviseur(small_M, N)
    elif (methode == 'diviseur_bin'):
        H, W = taille_sup_binaire(small_M)
    else :
        print('add_border_zero mauvaise methode')
        return 0
    return add_border_zero_RB(small_M, H, W)




def crop_image(img, d=32):
    '''Make dimensions divisible by `d`'''

    new_size = (img.size[0] - img.size[0] % d,
                img.size[1] - img.size[1] % d)

    bbox = [
            int((img.size[0] - new_size[0])/2),
            int((img.size[1] - new_size[1])/2),
            int((img.size[0] + new_size[0])/2),
            int((img.size[1] + new_size[1])/2),
    ]

    img_cropped = img.crop(bbox)
    return img_cropped
