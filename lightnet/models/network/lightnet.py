import torch
import torch.nn as nn
from .hyper import GlobalNet
from .local import LocalNet

MODE_SSR = 0
MODE_GLOBAL = 1
MODE_MIX = 2

class LightNet(nn.Module):
    def __init__(self, cfg, mode = MODE_SSR, wrap_im = True) -> None:
        """
        mode: 0 -> use SSR only; 1 -> use global only; 2 -> mixed with uncertainty
        """
        super().__init__()
        self.wrap_im = wrap_im
        assert(mode in [0,1,2])
        if mode == MODE_SSR or mode == MODE_MIX:
            self.model = LocalNet(cfg)
        if mode > MODE_SSR:
            self.model_global = GlobalNet(**getattr(cfg, "model_global", {}))
    
    def forward_mix(
            self,
            im: torch.Tensor, uv: torch.LongTensor,
            directions: torch.Tensor, positions: torch.Tensor,
            index: torch.Tensor, uncertainty: torch.Tensor,
            normal: torch.Tensor, Kd: torch.Tensor, Ks: torch.Tensor, rough: torch.Tensor
        ):
        if self.wrap_im:
            im = im[0,...]
        result = self.model_global(im, directions, positions, index)
        self.model.encode(im)
        ssrt_mask = uv[:,0] >= 0
        result_local = self.model(uv, directions, index, normal, Kd, Ks, rough)
        result[ssrt_mask,:] = result[ssrt_mask,:] * uncertainty[ssrt_mask,:] + result_local[ssrt_mask,:] * (1 - uncertainty[ssrt_mask,:])
        return result
    
    def forward_ssr(
            self,
            im: torch.Tensor, uv: torch.LongTensor,
            directions: torch.Tensor, index: torch.Tensor,
            normal: torch.Tensor, Kd: torch.Tensor, Ks: torch.Tensor, rough: torch.Tensor
        ):
        if self.wrap_im:
            im = im[0,...]
        self.model.encode(im)
        ssrt_mask = uv[:,0] >= 0
        assert torch.all(ssrt_mask)
        return self.model(uv, directions, index, normal, Kd, Ks, rough)
    
    def forward_nossr(self, im: torch.Tensor, directions: torch.Tensor, positions: torch.Tensor, index: torch.LongTensor):
        if self.wrap_im:
            im = im[0,...]
        return self.model_global(im, directions, positions, index)
    
    def forward(self, mode = MODE_SSR, *args, **kwargs):
        assert(mode in [0,1,2])
        if mode == MODE_SSR:
            return self.forward_ssr(*args, **kwargs)
        elif mode == MODE_GLOBAL:
            return self.forward_nossr(*args, **kwargs)
        else: # MODE_MIX
            return self.forward_mix(*args, **kwargs)