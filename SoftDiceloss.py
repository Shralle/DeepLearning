import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

def SoftDiceloss(prediction, target, mask, batch_size):
    mask = mask.expand(9,batch_size,256,256)
    mask = mask.permute(1,0,2,3)
    print("mask,pred,target")
    print(mask.shape)
    print(prediction.shape)
    print(target.shape)

    upper = (prediction * mask) * target * 2
    upper = torch.sum(upper)
    down = torch.sum(prediction*mask)^2 + torch.sum(target)^2

    loss = 1 - (upper / down)

    return loss