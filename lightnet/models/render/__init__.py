import torch
import torch.nn as nn
import numpy as np

def get_light_chunk(model, im, model_kwargs, bn, chunk, mode = 2):
    if chunk > 0:
        # print(uv.shape, ssrt_uv.shape, direction.shape, index.shape, normal.shape)
        splits = [{k : model_kwargs[k][i:i+chunk] for k in model_kwargs} for i in range(0, bn, chunk)]
        lights = []
        for split in splits:
            # print(*[x.shape for _, x in split.items()])
            light = model(mode, im, **split)
            lights.append(light)
        return torch.cat(lights, dim=0)
    else:
        return model(mode, im, **model_kwargs)

class RenderingLayerBase(nn.Module):
    def __init__(self, imWidth = 160, imHeight = 120, fov=120, cameraPos = [0, 0, 0], brdf_type = "ggx", spp = 1024):
        super(RenderingLayerBase, self).__init__()
        self.imHeight = imHeight
        self.imWidth = imWidth

        self.fov = fov/180.0 * np.pi
        self.cameraPos = np.array(cameraPos, dtype=np.float32).reshape([1, 3, 1, 1])
        self.xRange = 1 * np.tan(self.fov/2)
        self.yRange = float(imHeight) / float(imWidth) * self.xRange
        x, y = np.meshgrid(np.linspace(-self.xRange, self.xRange, imWidth),
                np.linspace(-self.yRange, self.yRange, imHeight ) )
        y = np.flip(y, axis=0)
        z = -np.ones( (imHeight, imWidth), dtype=np.float32)

        pCoord = np.stack([x, y, z]).astype(np.float32)
        self.pCoord = pCoord[np.newaxis, :, :, :]
        v = self.cameraPos - self.pCoord
        v = v / np.sqrt(np.maximum(np.sum(v*v, axis=1), 1e-12)[:, np.newaxis, :, :] )
        v = v.astype(dtype = np.float32)

        self.v = torch.from_numpy(v)
        self.pCoord = torch.from_numpy(self.pCoord)

        self.up = torch.Tensor([0,1,0])
        assert(brdf_type in ["disney", "ggx"])
        self.brdf_type = brdf_type
        self.spp = spp

        self.v = torch.nn.parameter.Parameter(self.v, False)
        self.pCoord = torch.nn.parameter.Parameter(self.pCoord, False)
        self.up = torch.nn.parameter.Parameter(self.up, False)