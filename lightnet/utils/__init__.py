import os
import os.path as osp
from typing import Callable
import torch
import numpy as np
# from torch.tensor import Tensor
import math
import models
import cv2
import glob
from .cfgnode import CfgNode
import torch.nn.functional as F
import glm


class Metric(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_model_with_name(path, mods, modNames, epochs, is_parallel=False):
    path_save = os.path.join(path, f'{epochs}.pt')
    data_save = {
        "epoch": epochs
    }
    for modName in modNames:
        data_save[modName] = mods[modName].state_dict() if not is_parallel or modName == 'mtl' else mods[modName].module.state_dict()
    torch.save(data_save, path_save)
    print(f"Model saved to {path_save} successfully in epoch {epochs}")

def save_model(path, mods, epochs):
    path_save = os.path.join(path, f'{epochs}.pt')
    data_save = {
        "encoder": mods['encoder'].state_dict(),
        "albedo": mods['albedo'].state_dict(),
        "normal": mods['normal'].state_dict(),
        "roughness": mods['roughness'].state_dict(),
        "metallic": mods['metallic'].state_dict(),
        # "specular": mods['specular'].state_dict(),
        "depth": mods['depth'].state_dict(),
        "epoch": epochs
    }
    torch.save(data_save, path_save)
    print(f"Model saved to {path_save} successfully in epoch {epochs}")

def load_model(path, epoch=None):
    if epoch:
        path = os.path.join(path, f"{epoch}.pt")
        if not os.path.exists(path):
            raise ValueError("Invalid model path!")
        data_load = torch.load(path)
        if epoch != data_load["epoch"]:
            print("WARNING: Inconsistent load epochs")
        return data_load
    if not os.path.exists(path):
        raise ValueError("Invalid model path!")
    # pts = [
    #     pt.split('.')[0] for pt in os.listdir(path) if 'pt' in pt
    # ]
    pts = [
        pt.split('.')[0] for pt in os.listdir(path) if 'pt' in pt and 'pth' not in pt
    ]
    if len(pts) == 0:
        raise AssertionError("No existing models!")
    pts = [int(pt) for pt in pts]
    epoch = max(pts)
    path = os.path.join(path, f"{epoch}.pt")
    data_load = torch.load(path)
    if epoch != data_load["epoch"]:
        print("WARNING: Inconsistent load epochs")
    return data_load

def to8b(im, srgb=True):
    if srgb:
        im = im**(1/2.2)
    return (255*np.clip(im,0,1)).astype(np.uint8)

def save_img(imName, im):
    im = im.detach().cpu().numpy()
    im = np.transpose(im[::-1,...], [1,2,0])
    return cv2.imwrite(imName, im)

def save_ldr(imName, im):
    im = im.detach().cpu().numpy()
    im = np.transpose(im[::-1,...], [1,2,0])
    return cv2.imwrite(imName, to8b(im))

def vis_img(imName, im, colorMap = True):
    im = im.numpy()
    im = np.transpose(im, [1,2,0])
    im = np.clip(im, 0, 1)
    im = (im * 255).astype(np.uint8)
    if colorMap:
        im = cv2.applyColorMap(im, cv2.COLORMAP_JET)
    return cv2.imwrite(imName, im)

def loadMask(imName, imSize, interpolation=cv2.INTER_NEAREST, ch_dim=False):
    im = cv2.imread(imName, -1)
    if im is None:
        raise ValueError(f"ERROR: Open {imName} failed!")
    im = cv2.resize(im, imSize, interpolation = interpolation)
    if len(im.shape) == 3:
        im = im[:,:,0]
    if ch_dim:
        im = im[np.newaxis,:,:]
    return torch.from_numpy(im)


def loadHdr(imName, imSize=None, interpolation=cv2.INTER_AREA):
    # if not osp.isfile(imName):
    #     raise ValueError("Invalid image name: %s" % imName)
    im = cv2.imread(imName, -1)
    if im is None:
        raise ValueError(f"ERROR: Open {imName} failed!")
    if imSize:
        im = cv2.resize(im, imSize, interpolation = interpolation)
    im = np.transpose(im, [2, 0, 1])
    im = im[::-1, :, :].copy()
    # return im
    return torch.from_numpy(im)

def focal_from_fov(width, height, fov, device='cpu'):
    fx = 2*math.tan(fov/180*np.pi*0.5)
    fy = fx * height / width
    return torch.tensor([fx, fy], device=device)

def rot_phi(phi, device='cpu'):
    return torch.tensor(
        [
            [math.cos(phi), 0, -math.sin(phi), 0],
            [0, 1, 0, 0],
            [math.sin(phi), 0, math.cos(phi), 0],
            [0, 0, 0, 1],
        ],
        dtype=torch.float32,
        device=device
    )

def look_at_inv(origin, target, world_up=np.array([0, 1, 0], dtype=np.float32)):
    """
    Get 4x4 camera to world space matrix, for camera looking at target
    """
    back = origin - target
    back /= np.linalg.norm(back)
    right = np.cross(world_up, back)
    right /= np.linalg.norm(right)
    up = np.cross(back, right)

    cam_to_world = np.empty((4, 4), dtype=np.float32)
    cam_to_world[:3, 0] = right
    cam_to_world[:3, 1] = up
    cam_to_world[:3, 2] = back
    cam_to_world[:3, 3] = origin
    cam_to_world[3, :] = [0, 0, 0, 1]
    return cam_to_world

def hughes_moeller(n: torch.Tensor, eps:float = 1e-6):
    """
    Generate orthonormal coordinate system based on surface normal
    :param: n (bn, 3, ...)
    """
    z = F.normalize(n, dim=1, eps=eps)
    y = torch.where(
        torch.abs(n[:,0,...]) > torch.abs(n[:,2,...]),
        torch.stack([-n[:,1,...], n[:,0,...], torch.zeros_like(n[:,0,...])], dim=1),
        torch.stack([torch.zeros_like(n[:,0,...]), -n[:,2,...], n[:,1,...]], dim=1)
    )
    y = F.normalize(y, dim=1, eps=eps)
    x = F.normalize(torch.cross(y, z, dim=1), dim=1, eps=eps)
    return x, y, z

def create_frame(n: torch.Tensor, eps:float = 1e-6):
    """
    Generate orthonormal coordinate system based on surface normal
    [Duff et al. 17] Building An Orthonormal Basis, Revisited. JCGT. 2017.
    :param: n (bn, 3, ...)
    """
    z = F.normalize(n, dim=1, eps=eps)
    sgn = torch.where(z[:,2,...] >= 0, 1.0, -1.0)
    a = -1.0 / (sgn + z[:,2,...])
    b = z[:,0,...] * z[:,1,...] * a
    x = torch.stack([1.0 + sgn * z[:,0,...] * z[:,0,...] * a, sgn * b, -sgn * z[:,0,...]], dim=1)
    y = torch.stack([b, sgn + z[:,1,...] * z[:,1,...] * a, -z[:,1,...]], dim=1)
    return x, y, z

def to_global(d, x, y, z):
    """
    d, x, y, z: (*, 3)
    """
    return d[:,0:1] * x + d[:,1:2] * y + d[:,2:3] * z

def depth_to_vpos(depth: torch.Tensor, fov, permute=False) -> torch.Tensor:
    row, col = depth.shape
    fovx = math.radians(fov)
    fovy = 2 * math.atan(math.tan(fovx / 2) / (col / row))
    vpos = torch.zeros(row, col, 3, device=depth.device)
    dmax = torch.max(depth)
    depth = depth / dmax
    Y = 1 - (torch.arange(row, device=depth.device) + 0.5) / row
    Y = Y * 2 - 1
    X = (torch.arange(col, device=depth.device) + 0.5) / col
    X = X * 2 - 1
    Y, X = torch.meshgrid(Y, X)
    vpos[:,:,0] = depth * X * math.tan(fovx / 2)
    vpos[:,:,1] = depth * Y * math.tan(fovy / 2)
    vpos[:,:,2] = -depth
    return vpos if not permute else vpos.permute(2,0,1)
