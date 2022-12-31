from .cfgnode import CfgNode
import torch
from matplotlib import cm
import torchvision.utils as vutils

def plot_images(gt, pred, image, mask, filename, colormap=None):
    B = gt.shape[0]
    image = image.clamp(0, 1) ** (1/2.2)
    img = torch.cat((gt * mask, pred * mask), dim=0) # (2B, [C,] H, W)
    if not colormap:
        img = torch.cat((image, img), dim=0)
    else:
        img = img.cpu().numpy()
        colormap = cm.get_cmap(colormap)
        img = colormap(img)[...,:-1]
        img = torch.from_numpy(img).permute(0, 3, 1, 2)
        img = torch.cat((image.cpu(), img), dim=0)
    vutils.save_image(img, filename, nrow=B)

def plot_images_wo_gt(pred, image, filename, colormap=None):
    B = pred.shape[0]
    if not colormap:
        img = torch.cat((image, pred), dim=0)
    else:
        img = pred.cpu().numpy()
        colormap = cm.get_cmap(colormap)
        img = colormap(img)[...,:-1]
        img = torch.from_numpy(img).permute(0, 3, 1, 2)
        img = torch.cat((image.cpu(), img), dim=0)
    vutils.save_image(img, filename, nrow=B)