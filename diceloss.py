def SoftDiceloss(prediction, target, mask, batch):
    mask = mask.expand(9,batch,256,256))
    upper = (prediction * mask * target * 2
    upper = torch.sum(upper)
    down = torch.sum(prediction*mask) + torch.sum(target)

    loss = 1 - (upper / down)

    return loss