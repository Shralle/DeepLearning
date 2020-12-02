import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

def SoftDiceloss(prediction, target, mask):
    batch_size = mask.shape[0]
    mask = torch.abs(mask-1)
    mask = mask.expand(9,batch_size,256,256)
    mask = mask.permute(1,0,2,3)
    
    upper = (prediction * mask) * target
    upper = 2 * torch.sum(upper)

    down = torch.sum((prediction * mask) **2) + torch.sum(target**2)

    loss = (1 - (upper / down)) / batch_size

    return loss