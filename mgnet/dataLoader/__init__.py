import os.path as osp
from glob import glob
import numpy as np
import cv2
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision
from prefetch_generator import BackgroundGenerator


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__(), 64)


class MGDataset(Dataset):
    def __init__(
        self, dataRoot, 
        imHeight = 240, imWidth = 320, preload_level = 0,
        phase='TRAIN', sceneList:list = None, clamp_im = False, random_flip=False,
        **kwargs
        ):
        super().__init__()

        if phase.upper() == 'TRAIN':
            sceneFile = osp.join(dataRoot, 'train.txt')
        elif phase.upper() == 'TEST':
            sceneFile = osp.join(dataRoot, 'test.txt')
        elif phase.upper() == 'VAL':
            sceneFile = osp.join(dataRoot, 'val.txt')
        elif sceneList is None:
            raise ValueError('Unrecognized phase for data loader')

        if sceneList:
            self.sceneList = sceneList
        else:
            with open(sceneFile, 'r') as fIn:
                self.sceneList = fIn.readlines() 
            self.sceneList = [x.strip() for x in self.sceneList]
        self.imList = []
        for s in self.sceneList:
            self.imList += glob(osp.join(dataRoot, s, "*_im.exr"))
        print(f"{len(self.imList)} images for {phase}")
        self.imHeight = imHeight
        self.imWidth = imWidth
        self.clamp_im = clamp_im
        self.random_flip = random_flip

        print('Image Num: %d' % len(self.imList))

        self.albedoList = [x.replace('_im', '_albedo') for x in self.imList]
        self.matList = [x.replace('_im', '_material') for x in self.imList]
        self.segList = [x.replace('_im', '_mask') for x in self.imList]
        self.depthList = [x.replace('_im', '_depth') for x in self.imList]
        self.normalList = [x.replace('_im', '_normal') for x in self.imList]

        self.imNames = self.imList
        self.preload_level = preload_level

        if preload_level >= 1:
            self.imList = [self.loadHdr(x) for x in tqdm(self.imList, desc="Image Load")]
            self.segList = [self.loadMask(x).bool() for x in tqdm(self.segList, desc="Mask Load")]
            self.depthList = [self.loadHdr(x, cv2.INTER_NEAREST)[0:1,:,:] for x in tqdm(self.depthList, desc="Depth Load")]
        if preload_level >= 2:
            self.albedoList = [self.loadHdr(x) for x in tqdm(self.albedoList, desc="Albedo Load")]
            self.matList = [self.loadHdr(x, cv2.INTER_NEAREST) for x in tqdm(self.matList, desc="Material Load")]
            self.normalList = [self.loadHdr(x, cv2.INTER_NEAREST) for x in tqdm(self.normalList, desc="Normal Load")]

    def __len__(self):
        return len(self.imNames)
    
    def __getitem__(self, index):
        if self.preload_level >= 1:
            seg = self.segList[index]
        else:
            seg = self.loadMask(self.segList[index]).bool()
        segAlb = seg
        segGeo = seg
        
        if self.preload_level >= 1:
            im = self.imList[index]
        else:
            im = self.loadHdr(self.imList[index])
        im = torch.where(torch.isfinite(im), im, torch.tensor(0.0))

        if self.preload_level >= 2:
            albedo = self.albedoList[index]
        else:
            albedo = self.loadHdr(self.albedoList[index])
        
        if self.preload_level >= 2:
            mat_info = self.matList[index]
        else:
            mat_info = self.loadHdr(self.matList[index], cv2.INTER_NEAREST)
        # In case material info contains abnormal values
        segMat = seg & ~((mat_info[0:1,...] == 0.0) & (mat_info[1:2,...] == 0.0))
        segMat &= torch.all(torch.isfinite(mat_info), dim=0, keepdim=True)
        mat_info[~torch.isfinite(mat_info)] = 0

        if self.preload_level >= 1:
            depth = self.depthList[index]
        else:
            depth = self.loadHdr(self.depthList[index], cv2.INTER_NEAREST)[0:1,:,:]
        # In case depth info contains infinite values
        segGeo &= torch.isfinite(depth)
        segAlb &= segGeo
        depth[~torch.isfinite(depth)] = 0
        depth /= torch.clamp(torch.max(depth), min=1e-6) # Normalize depth

        if self.preload_level >= 2:
            normal = self.normalList[index]
        else:
            normal = self.loadHdr(self.normalList[index], cv2.INTER_NEAREST)
        segGeo &= torch.all(torch.isfinite(normal), dim=0, keepdim=True)
        segAlb &= segGeo
        normal[~torch.isfinite(normal)] = 0
        normal = F.normalize(normal, dim=0, eps=1e-6)

        assert(mat_info.shape[0] == 3)
        rough = mat_info[0:1,...]
        metal = mat_info[1:2,...]
        specl = mat_info[2:3,...]
        segMat = segMat.float()
        segAlb = segAlb.float()
        segGeo = segGeo.float()

        sceneName = self.imNames[index]
        sceneName = sceneName[sceneName.rfind('/')+1:sceneName.rfind('_')]

        if self.clamp_im:
            im = im.clamp_(0, 1)

        batchDict = {
            'albedo': albedo,
            'normal': normal,
            'roughness': rough,
            'metallic': metal,
            'specular': specl,
            'depth': depth,
            'segAlb': segAlb,
            'segMat': segMat,
            'segGeo': segGeo,
            'im': im,
            'scene': sceneName
        }
        if self.random_flip and torch.rand(1) < 0.5:
            import torchvision.transforms.transforms as transforms
            for k in batchDict:
                if k != 'scene':
                    batchDict[k] = transforms.F.hflip(batchDict[k])
                    if k == 'normal':
                        batchDict[k][0:1,:,:] = -batchDict[k][0:1,:,:]
        return batchDict

    def loadImage(self, imName, gamma:str=None):
        if not osp.isfile(imName):
            raise ValueError("Invalid image name: %s" % imName)
        if gamma:
            assert(gamma.lower() in ['srgb', 'kjl'])
        raise NotImplementedError

    def loadMask(self, imName):
        im = cv2.imread(imName, -1)
        if im is None:
            return None
        im = cv2.resize(im, (self.imWidth, self.imHeight), interpolation = cv2.INTER_NEAREST)
        im = im[np.newaxis,:,:]
        return torch.from_numpy(im)

    def loadHdr(self, imName, interpolation=cv2.INTER_AREA):
        im = cv2.imread(imName, -1)
        if im is None:
            raise ValueError(f"ERROR: Open {imName} failed!")
        im = cv2.resize(im, (self.imWidth, self.imHeight), interpolation = interpolation)
        im = np.transpose(im, [2, 0, 1])
        im = im[::-1, :, :].copy()
        return torch.from_numpy(im)
