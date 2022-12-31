import torch
import numpy as np
import torch.optim as optim
import argparse
import random
import os

import models
import torchvision.utils as vutils
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
import dataLoader
import utils
import time
import datetime
from tqdm import tqdm
import yaml
import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar


class MGNet(pl.LightningModule):
    def __init__(self, cfg, exp_dir, train=True, val=True, test=False) -> None:
        super().__init__()
        self.models = nn.ModuleDict(models.get_model(cfg.model))
        self.perceptual_loss = models.PerceptualLoss()
        self.cfg = cfg
        self.exp_dir = exp_dir
        if train:
            self.train_set = dataLoader.MGDataset(**self.cfg.dataset)
        if val:
            self.val_set = dataLoader.MGDataset(**self.cfg.dataset, phase='VAL')
        if test:
            self.test_set = dataLoader.MGDataset(**self.cfg.dataset, phase='TEST')
            os.makedirs(os.path.join(self.exp_dir, "test/albedo"), exist_ok=True)
            os.makedirs(os.path.join(self.exp_dir, "test/depth"), exist_ok=True)
            os.makedirs(os.path.join(self.exp_dir, "test/normal"), exist_ok=True)
            os.makedirs(os.path.join(self.exp_dir, "test/roughness"), exist_ok=True)
            os.makedirs(os.path.join(self.exp_dir, "test/metallic"), exist_ok=True)

    def forward(self):
        pass

    def configure_optimizers(self):
        optimizer = optim.Adam(self.models.parameters(), **self.cfg.optim)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 5, 0.5, -1, verbose=True)
        return [optimizer], [scheduler]
    
    def train_dataloader(self):
        return dataLoader.DataLoaderX(self.train_set, batch_size=self.cfg.train.batch_size, num_workers=self.cfg.dataset.num_workers, shuffle=True)
    
    def val_dataloader(self):
        return dataLoader.DataLoaderX(self.val_set, batch_size=self.cfg.val.batch_size, num_workers=self.cfg.dataset.num_workers, shuffle=False)
    
    def test_dataloader(self):
        return dataLoader.DataLoaderX(self.test_set, batch_size=self.cfg.val.batch_size, num_workers=self.cfg.dataset.num_workers, shuffle=False)
    
    def training_step(self, batch, batch_nb):
        albedo = batch['albedo']
        normal = batch['normal']
        rough = batch['roughness']
        metal = batch['metallic']
        depth = batch['depth']
        segAlb = batch['segAlb']
        segMat = batch['segMat']
        segGeo = batch['segGeo']
        im = batch['im']
        mat = torch.cat([rough, metal], dim=1)

        features = self.models['encoder'](im)
        albedoPred = self.models['albedo'](im, *features)
        normalPred = self.models['normal'](im, *features)
        matPred = self.models['material'](im, *features)
        depthPred = self.models['depth'](im, *features)

        albedoPred = torch.clamp(albedoPred, 0, 1)
        depthPred = models.LSregress(depthPred * segGeo, depth * segGeo, depthPred)
        depthPred = torch.clamp(depthPred, min=0)

        pixAlbNum = torch.sum(segAlb).item()
        pixMatNum = torch.sum(segMat).item()
        pixGeoNum = torch.sum(segGeo).item()

        L2Err = torch.sum((torch.log1p(albedoPred) - torch.log1p(albedo)) * (torch.log1p(albedoPred) - torch.log1p(albedo)) * segAlb) / pixAlbNum / 3.0
        percErr = self.perceptual_loss(albedoPred, albedo, segAlb, layers=self.cfg.train.perceptual.albedo)
        albedoErr = 0.5 * (L2Err + percErr * self.cfg.train.perceptual.weight)

        percErr = self.perceptual_loss(normalPred, normal, segGeo, layers=self.cfg.train.perceptual.normal)
        normalErr = models.normal_loss(normalPred, normal, segGeo)
        normalErr = 0.5 * (normalErr + percErr * self.cfg.train.perceptual.weight)
        matErr = torch.sum((matPred - mat) * (matPred - mat) * segMat) / pixMatNum

        roughPred = matPred[:,0:1,...]
        metalPred = matPred[:,1:2,...]
        percErrRough = self.perceptual_loss(roughPred.expand(-1, 3, -1, -1), rough.expand(-1, 3, -1, -1), segMat, layers=self.cfg.train.perceptual.material)
        percErrMetal = self.perceptual_loss(metalPred.expand(-1, 3, -1, -1), metal.expand(-1, 3, -1, -1), segMat, layers=self.cfg.train.perceptual.material)
        matErr = 0.5 * ((percErrRough + percErrMetal) * self.cfg.train.perceptual.weight * 0.5 + matErr)

        depthErr = torch.sum((torch.log(depthPred+1) - torch.log(depth+1)) * (torch.log(depthPred+1) - torch.log(depth+1)) * segGeo) / pixGeoNum

        loss = albedoErr + normalErr + matErr + depthErr

        with torch.no_grad():
            self.log_dict({
                'train/total_loss': loss,
                'train/albedo_loss': albedoErr,
                'train/normal_loss': normalErr,
                'train/material_loss': matErr,
                'train/depth_loss': depthErr
            })
        return {
            'loss': loss,
            'albedo_loss': albedoErr,
            'normal_loss': normalErr,
            'material_loss': matErr,
            'depth_loss': depthErr
        }

    def test_step(self, batch, batch_nb):
        albedo = batch['albedo']
        normal = batch['normal']
        rough = batch['roughness']
        metal = batch['metallic']
        depth = batch['depth']
        segAlb = batch['segAlb']
        segMat = batch['segMat']
        segGeo = batch['segGeo']
        im = batch['im']

        features = self.models['encoder'](im)
        albedoPred = self.models['albedo'](im, *features)
        normalPred = self.models['normal'](im, *features)
        matPred = self.models['material'](im, *features)
        depthPred = self.models['depth'](im, *features)
        albedoPred = torch.clamp(albedoPred, 0, 1)
        depthPred = models.LSregress(depthPred * segGeo, depth * segGeo, depthPred)
        depthPred = torch.clamp(depthPred, min=0)

        utils.plot_images(albedo, albedoPred, im, segAlb, os.path.join(self.exp_dir, f"test/albedo/{batch_nb:04d}.png"))
        utils.plot_images((normal + 1) * 0.5, (normalPred + 1) * 0.5, im, segGeo, os.path.join(self.exp_dir, f"test/normal/{batch_nb:04d}.png"))
        utils.plot_images(depth[:,0,...], depthPred[:,0,...].clamp(0, 1), im, segGeo[:,0,...], os.path.join(self.exp_dir, f"test/depth/{batch_nb:04d}.png"), colormap='magma')
        utils.plot_images(rough[:,0,...], matPred[:,0,...], im, segMat[:,0,...], os.path.join(self.exp_dir, f"test/roughness/{batch_nb:04d}.png"), colormap='jet')
        utils.plot_images(metal[:,0,...], matPred[:,1,...], im, segMat[:,0,...], os.path.join(self.exp_dir, f"test/metallic/{batch_nb:04d}.png"), colormap='jet')

        pixAlbNum = torch.sum(segAlb).item()
        pixMatNum = torch.sum(segMat).item()
        pixGeoNum = torch.sum(segGeo).item()

        albedoErr = torch.sum((albedoPred - albedo) * (albedoPred - albedo) * segAlb) / pixAlbNum / 3.0
        normalErr = torch.sum((normalPred - normal) * (normalPred - normal) * segGeo) / pixGeoNum / 3.0
        roughErr = torch.sum((matPred[:,:1,:,:] - rough) * (matPred[:,:1,:,:] - rough) * segMat) / pixMatNum
        metalErr = torch.sum((matPred[:,1:,:,:] - metal) * (matPred[:,1:,:,:] - metal) * segMat) / pixMatNum
        depthErr = torch.sum((torch.log(depthPred+1) - torch.log(depth+1)) * (torch.log(depthPred+1) - torch.log(depth+1)) * segGeo) / pixGeoNum

        return {
            'albedo_loss': albedoErr,
            'normal_loss': normalErr,
            'roughness_loss': roughErr,
            'metallic_loss': metalErr,
            'depth_loss': depthErr
        }

    def test_epoch_end(self, outputs) -> None:
        albedo_loss = torch.stack([x['albedo_loss'] for x in outputs]).mean()
        normal_loss = torch.stack([x['normal_loss'] for x in outputs]).mean()
        roughness_loss = torch.stack([x['roughness_loss'] for x in outputs]).mean()
        metallic_loss = torch.stack([x['metallic_loss'] for x in outputs]).mean()
        depth_loss = torch.stack([x['depth_loss'] for x in outputs]).mean()

        with open(os.path.join(self.exp_dir, "test/metrics.txt"), 'w') as f:
            f.write(f"[ALBEDO] {albedo_loss:.6f}\n")
            f.write(f"[NORMAL] {normal_loss:.6f}\n")
            f.write(f"[ROUGHNESS] {roughness_loss:.6f}\n")
            f.write(f"[METALLIC] {metallic_loss:.6f}\n")
            f.write(f"[DEPTH] {depth_loss:.6f}\n")

        
    
    def validation_step(self, batch, batch_nb):
        albedo = batch['albedo']
        normal = batch['normal']
        rough = batch['roughness']
        metal = batch['metallic']
        depth = batch['depth']
        segAlb = batch['segAlb']
        segMat = batch['segMat']
        segGeo = batch['segGeo']
        im = batch['im']
        mat = torch.cat([rough, metal], dim=1)

        features = self.models['encoder'](im)
        albedoPred = self.models['albedo'](im, *features)
        normalPred = self.models['normal'](im, *features)
        matPred = self.models['material'](im, *features)
        depthPred = self.models['depth'](im, *features)

        albedoPred = torch.clamp(albedoPred, 0, 1)
        depthPred = models.LSregress(depthPred * segGeo, depth * segGeo, depthPred)
        depthPred = torch.clamp(depthPred, min=0)

        pixAlbNum = torch.sum(segAlb).item()
        pixMatNum = torch.sum(segMat).item()
        pixGeoNum = torch.sum(segGeo).item()

        albedoErr = torch.sum((albedoPred - albedo) * (albedoPred - albedo) * segAlb) / pixAlbNum / 3.0

        normalErr = torch.sum((normalPred - normal) * (normalPred - normal) * segGeo) / pixGeoNum / 3.0
        
        roughErr = torch.sum((matPred[:,:1,:,:] - rough) * (matPred[:,:1,:,:] - rough) * segMat) / pixMatNum
        metalErr = torch.sum((matPred[:,1:,:,:] - metal) * (matPred[:,1:,:,:] - metal) * segMat) / pixMatNum

        depthErr = torch.sum((torch.log(depthPred+1) - torch.log(depth+1)) * (torch.log(depthPred+1) - torch.log(depth+1)) * segGeo) / pixGeoNum

        albLoss = albedoErr
        normLoss = normalErr
        matLoss = (roughErr + metalErr)
        depthLoss = depthErr

        totalErr = albLoss + normLoss + matLoss + depthLoss
        self.log('loss', totalErr, prog_bar=True, logger=False, on_step=True, on_epoch=False)
        return {
            'loss': totalErr,
            'albedo_loss': albedoErr,
            'normal_loss': normalErr,
            'material_loss': matLoss,
            'roughness_loss': roughErr,
            'metallic_loss': metalErr,
            'depth_loss': depthErr
        }
    
    def validation_epoch_end(self, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        albedo_loss = torch.stack([x['albedo_loss'] for x in outputs]).mean()
        normal_loss = torch.stack([x['normal_loss'] for x in outputs]).mean()
        material_loss = torch.stack([x['material_loss'] for x in outputs]).mean()
        roughness_loss = torch.stack([x['roughness_loss'] for x in outputs]).mean()
        metallic_loss = torch.stack([x['metallic_loss'] for x in outputs]).mean()
        depth_loss = torch.stack([x['depth_loss'] for x in outputs]).mean()
        # self.log('validate/loss', loss)
        # self.log('validate/albedo_loss', albedo_loss)
        # self.log('validate/normal_loss', normal_loss)
        # self.log('validate/material_loss', material_loss)
        # self.log('validate/roughness_loss', roughness_loss)
        # self.log('validate/metallic_loss', metallic_loss)
        # self.log('validate/depth_loss', depth_loss)
        self.log_dict({
            'validate/loss': loss,
            'validate/albedo_loss': albedo_loss,
            'validate/normal_loss': normal_loss,
            'validate/material_loss': material_loss,
            'validate/roughness_loss': roughness_loss,
            'validate/metallic_loss': metallic_loss,
            'validate/depth_loss': depth_loss,
            # 'step': self.current_epoch
        })


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config", type=str, help="Path to (.yml) config file."
    )
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--version", '-v', type=int, default=None)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = utils.CfgNode(cfg_dict)
    
    num_gpus = len(cfg.experiment.device_ids)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(lambda x: str(x), cfg.experiment.device_ids))
    torch.multiprocessing.set_sharing_strategy('file_system')

    logger = loggers.TensorBoardLogger(save_dir=cfg.experiment.path_logs, name=cfg.experiment.id, version=args.version)
    exp_dir = logger.log_dir
    checkpoint_callback = ModelCheckpoint(os.path.join(exp_dir, 'checkpoint'), save_last=True, every_n_epochs=cfg.train.save_epochs)
    system = MGNet(cfg, exp_dir, train=not args.test, val=not args.test, test=args.test)
    if args.test:
        trainer = pl.Trainer(
            logger=logger,
            gpus=[0],
            max_epochs=cfg.train.epochs,
            strategy=None,
            callbacks=[checkpoint_callback, RichProgressBar(leave=True)]
        )
        trainer.test(system, ckpt_path=os.path.join(exp_dir, 'checkpoint/last.ckpt'))
    else:
        trainer = pl.Trainer(
            logger=logger,
            gpus=num_gpus,
            check_val_every_n_epoch=cfg.val.interval,
            max_epochs=cfg.train.epochs,
            strategy='ddp',
            callbacks=[checkpoint_callback, RichProgressBar(leave=True)]
        )
        trainer.fit(system)
    torch.cuda.empty_cache()
