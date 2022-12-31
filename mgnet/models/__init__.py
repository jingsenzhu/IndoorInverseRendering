from .network import *
from .loss import PerceptualLoss, normal_loss

def get_model(type: str):
    if type == "dense":
        return {
            'encoder': EncoderDenseNet(),
            'albedo': DecoderDenseNet(mode=0),
            'normal': DecoderDenseNet(mode=1),
            'material': DecoderDenseNet(mode=3),
            'depth': DecoderDenseNet(mode=4)
        }
    elif type == "resnet":
        return {
            'encoder': EncoderResNet(),
            'albedo': DecoderResNet(mode=0),
            'normal': DecoderResNet(mode=1),
            'material': DecoderResNet(mode=3),
            'depth': DecoderResNet(mode=4)
        }
    else:
        return {
            'encoder': EncoderMG(),
            'albedo': DecoderMG(mode=0),
            'normal': DecoderMG(mode=1),
            'material': DecoderMG(mode=3),
            'depth': DecoderMG(mode=4)
        }

def LSregress(pred, gt, origin):
    nb = pred.size(0)
    origSize = pred.size()
    pred = pred.reshape(nb, -1 )
    gt = gt.reshape(nb, -1 )

    coef = (torch.sum(pred * gt, dim = 1) / torch.clamp(torch.sum(pred * pred, dim=1), min=1e-5) ).detach()
    coef = torch.clamp(coef, 0.001, 1000)
    for n in range(0, len(origSize) -1 ):
        coef = coef.unsqueeze(-1)
    pred = pred.view(origSize )

    predNew = origin * coef.expand(origSize )

    return predNew