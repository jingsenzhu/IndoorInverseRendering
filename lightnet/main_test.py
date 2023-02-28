import argparse
import utils
import yaml
import torch
import random
import models
import torch.nn as nn
import torch.nn.functional as F
from tqdm.contrib import tenumerate
import os
import numpy as np
import cv2


def main_test(cfg):
    gpu_id = cfg.experiment.device_ids[0]
    device = f"cuda:{gpu_id}" if cfg.experiment.cuda else "cpu"
    torch.cuda.set_device(gpu_id)

    model = models.LightNet(cfg.model, models.MODE_MIX)
    checkpointdir = os.path.join(cfg.experiment.path_checkpoint, cfg.experiment.id)
    checkpoint = torch.load(os.path.join(checkpointdir, f"{cfg.eval.load_epoch}.pth"), map_location=device)
    model.load_state_dict(checkpoint)
    print(f"Load checkpoint at epoch {cfg.eval.load_epoch} success!")
    model.wrap_im = True
    model = nn.DataParallel(model, cfg.experiment.device_ids)
    model.to(device)
    out_dir = "edit"
    output = getattr(cfg.experiment, 'out_dir', os.path.join(cfg.experiment.path_logs, cfg.experiment.id, out_dir))
    os.makedirs(output, exist_ok=True)
    model.eval()
    render_kwargs = getattr(cfg, 'render', {})
    if 'chunk' not in render_kwargs:
        render_kwargs['chunk'] = 128*1024*len(cfg.experiment.device_ids)

    if not hasattr(cfg.eval, 'images'):
        assert hasattr(cfg.eval, 'image_list')
        with open(cfg.eval.image_list) as f:
            cfg.eval.images = f.readlines()
        cfg.eval.images = [x.strip() for x in cfg.eval.images]

    with torch.no_grad():
        for si, imName in tenumerate(cfg.eval.images):
            im = os.path.join(cfg.dataset.path, f"{imName}.exr")
            result = im
            im = cv2.imread(im, -1)[:,:,::-1]
            nh, nw = im.shape[0], im.shape[1]
            if nh < nw:
                newW = cfg.dataset.imWidth
                newH = int(float(cfg.dataset.imWidth) / float(nw) * nh)
                # col = cfg.dataset.imWidth // 2
                col = cfg.dataset.col
                row = int(float(col) / float(nw) * nh)
            else:
                newH = cfg.dataset.imHeight
                newW = int(float(cfg.dataset.imHeight) / float(nh) * nw)
                # row = cfg.dataset.imHeight // 2
                row = cfg.dataset.row
                col = int(float(row) / float(nh) * nw)
            im = cv2.resize(im, (newW, newH), interpolation = cv2.INTER_AREA)
            im = np.transpose(im, [2, 0, 1])[np.newaxis, :, :, :]
            im = torch.from_numpy(im).to(device).unsqueeze_(0).expand(len(cfg.experiment.device_ids),-1,-1,-1,-1)
            # im = torch.clamp(im, 0, 1)
            normal = os.path.join(cfg.dataset.path, f"{imName}_view_normal.exr")
            albedo = os.path.join(cfg.dataset.path, f"{imName}_albedo.exr")
            mat = os.path.join(cfg.dataset.path, f"{imName}_mat_info.exr")
            mask = os.path.join(cfg.dataset.path, f"{imName}_mask.exr")
            depth = os.path.join(cfg.dataset.path, f"{imName}_depth.exr")
            depth = cv2.imread(depth, -1)
            if len(depth.shape) == 3:
                depth = depth[:,:,-1]
            depth = torch.from_numpy(cv2.resize(depth, (col, row), interpolation=cv2.INTER_AREA))
            vpos = utils.depth_to_vpos(depth, cfg.eval.fov, True)
            vpos = vpos.unsqueeze_(0).to(device)

            renderer = models.RenderLayerClip(fov=cfg.eval.fov, spp=cfg.eval.spp, imWidth=col, imHeight=row, **render_kwargs)
            renderer.to(device)

            normal = utils.loadHdr(normal, (col, row), cv2.INTER_AREA)
            normal = normal.unsqueeze_(0).to(device)

            normal = F.normalize(normal, dim=1, eps=1e-6)
            
            albedo = utils.loadHdr(albedo, (col, row), cv2.INTER_AREA)
            albedo = albedo.unsqueeze_(0).to(device)
            
            mat = utils.loadHdr(mat, (col, row), cv2.INTER_AREA)
            mat = mat.unsqueeze_(0).to(device)
            rough = mat[:,0:1,...]
            metal = mat[:,1:2,...]

            center_x, center_y = cfg.eval.centers[si]
            radius = cfg.eval.radius[si]
            r_diff, r_spec, _ = renderer(center_x, center_y, radius, model, im, albedo, normal, rough, metal, vpos)
            rendered = r_diff + r_spec
            rendered = torch.where(torch.isfinite(rendered), rendered, torch.zeros_like(rendered))[0].cpu() # (3, row, col)

            result = utils.loadHdr(result, (col, row), cv2.INTER_AREA)
            mask = utils.loadMask(mask, (col, row), cv2.INTER_AREA, True)
            im_clip = result[:,center_y-radius:center_y+radius,center_x-radius:center_x+radius]
            mask_clip = mask[:,center_y-radius:center_y+radius,center_x-radius:center_x+radius]
            result[:,center_y-radius:center_y+radius,center_x-radius:center_x+radius] \
                = im_clip * (1 - mask_clip) + rendered * mask_clip

            path_save = os.path.join(output, f"result_{imName}.exr")
            utils.save_img(path_save, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", '-c', type=str, required=True, help="Path to (.yml) config file.")
    args = parser.parse_args()
    cfg = None
    with open(args.config) as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = utils.CfgNode(cfg_dict)
    
    seed = cfg.experiment.seed
    random.seed(seed)
    torch.manual_seed(seed)
    cfg.experiment.cuda &= torch.cuda.is_available()
    gpu_id = cfg.experiment.gpu_id if hasattr(cfg.experiment, "gpu_id") else cfg.experiment.device_ids[0]
    torch.cuda.set_device(gpu_id)
    main_test(cfg)