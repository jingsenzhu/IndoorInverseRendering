from . import RenderingLayerBase, get_light_chunk
from .brdf import *
from .ssrt import SSRTEngine, ssrt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import utils
import glm
import math

class RenderLayerClip(RenderingLayerBase):
    """
    Render a clipped image based on the specified clip center and radius
    e.g. Used for object insertion to focus on the inserted object, saving computations 
    """
    def __init__(self, imWidth=160, imHeight=120, fov=120, cameraPos=[0,0,0], brdf_type="ggx", spp=1024, chunk = 128*1024, parallel = None, uncertainty_boundary = None):
        super().__init__(imWidth, imHeight, fov, cameraPos, brdf_type, spp)
        self.chunk = chunk
        self.use_parallel = False
        if parallel is not None:
            self.use_parallel = True
            self.parallel_devices = parallel
            self.parallel_ssrt = SSRTEngine()
            self.parallel_ssrt = nn.DataParallel(self.parallel_ssrt, parallel)
        self.ub = uncertainty_boundary
    
    def forward(
            self,
            center_x,
            center_y,
            radius,
            model: nn.Module,
            im: torch.Tensor,
            albedo: torch.Tensor,
            normal: torch.Tensor,
            rough: torch.Tensor,
            metal: torch.Tensor,
            vpos: torch.Tensor
        ):
        """
        Render according to material, normal and lighting conditions
        Params:
            model: NeRF model to predict lights
            im, albedo, normal, rough, metal, vpos: (bn, c, h, w)
        """
        bn, _, row, col = vpos.shape
        assert(bn == 1) # Only support bn = 1
        assert(row == self.imHeight and col == self.imWidth)
        dev = vpos.device

        ##############################
        ########## IS Start ##########
        ##############################
        cx, cy, cz = utils.create_frame(normal)
        wi_world = self.v
        wi_x = torch.sum(cx*wi_world, dim=1)
        wi_y = torch.sum(cy*wi_world, dim=1)
        wi_z = torch.sum(cz*wi_world, dim=1)
        wi = torch.stack([wi_x, wi_y, wi_z], dim=1)
        wi_mask = torch.where(wi[:,2:3,...] < 0.001, torch.zeros_like(wi[:,2:3,...]), torch.ones_like(wi[:,2:3,...]))

        wi[:,2,...] = torch.clamp(wi[:,2,...], min=1e-3)
        wi = F.normalize(wi, dim=1, eps=1e-6)
        wi = wi.unsqueeze(1) # (bn, 1, 3, h, w)
        
        left = max(center_x - radius, 0)
        right = center_x + radius
        top = max(center_y - radius, 0)
        bottom = center_y + radius
        wi = wi[:,:,:,top:bottom,left:right]
        wi_mask = wi_mask[:,:,top:bottom,left:right]
        cx = cx[:,:,top:bottom,left:right]
        cy = cy[:,:,top:bottom,left:right]
        cz = cz[:,:,top:bottom,left:right]
        albedo_clip = albedo[:,:,top:bottom,left:right]
        metal_clip = metal[:,:,top:bottom,left:right]
        rough_clip = rough[:,:,top:bottom,left:right]

        irow = wi.size(3)
        icol = wi.size(4)

        samples = torch.rand(1, self.spp, 3, irow, icol, device=dev)
        wo_diffuse = square_to_cosine_hemisphere(samples[:,:,1:,...])
        if self.brdf_type == "ggx":
            specularF0 = baseColorToSpecularF0(albedo_clip, metal_clip)
            diffuseReflectance = albedo_clip * (1 - metal_clip)
            kS = probabilityToSampleSpecular(diffuseReflectance, specularF0)
            sample_diffuse = samples[:,:,0,...] >= kS
            wo_specular = sample_ggx_specular(samples[:,:,1:,...], rough_clip, wi)
        else:
            diffuseRatio = 0.5 * (1.0 - metal_clip)
            sample_diffuse = samples[:,:,0,...] < diffuseRatio # (bn, spp, h, w)
            wo_specular = sample_disney_specular(samples[:,:,1:,...], rough_clip, wi)
        wo = torch.where(sample_diffuse.unsqueeze(2).expand(1, self.spp, 3, irow, icol), wo_diffuse, wo_specular) # (bn, spp, 3, h, w)
        if self.brdf_type == "ggx":
            pdfs = pdf_ggx(albedo_clip, rough_clip, metal_clip, wi, wo).unsqueeze(2)
            eval_diff, eval_spec, mask = eval_ggx(albedo_clip, rough_clip, metal_clip, wi, wo)
        else:
            pdfs = pdf_disney(rough_clip, metal_clip, wi, wo).unsqueeze(2)
            eval_diff, eval_spec, mask = eval_disney(albedo_clip, rough_clip, metal_clip, wi, wo)
        direction = wo[:,:,0:1,...] * cx.unsqueeze(1) + wo[:,:,1:2,...] * cy.unsqueeze(1) + wo[:,:,2:3,...] * cz.unsqueeze(1)
        direction = direction.permute(0, 1, 3, 4, 2) # (bn, spp, h, w, 3)
        direction = direction.reshape(-1, 3)

        pdfs = torch.clamp(pdfs, min=0.001)
        ndl = torch.clamp(wo[:,:,2:,...], min=0)
        brdfDiffuse = eval_diff.expand([1, self.spp, 3, irow, icol] ) * ndl / pdfs
        brdfSpec = eval_spec.expand([1, self.spp, 3, irow, icol] ) * ndl / pdfs
        del wo, ndl, pdfs, eval_diff, eval_spec, wi
        ##############################
        ########### IS End ###########
        ##############################

        ##############################
        ######### SSRT Start #########
        ##############################
        fovy = 2 * math.atan(math.tan(self.fov / 2) / (self.imWidth / self.imHeight))
        depth = -vpos[0,...].permute(1,2,0) # (h, w, 3)
        depth[:,:,:-1] = 0
        dmin = torch.min(depth[:,:,-1])
        dmax = torch.max(depth[:,:,-1])
        depth /= dmax
        proj = glm.perspective(fovy, self.imWidth / self.imHeight, (dmin.item() / dmax.item()), 1.0)
        proj = torch.from_numpy(np.array(proj)).to(dev)
        depth = -depth
        depth = depth.view(-1, 3, 1)
        pos4 = torch.cat([depth, torch.ones(col*row, 1, 1, device=dev)], dim=1)
        pos4 = (proj.unsqueeze(0) @ pos4)[:,:,0]
        pos4 = pos4 / pos4[:,-1:]
        depth = (pos4[:,:-1].view(row, col, 3) + 1) * 0.5
        depth = depth[:,:,-1].unsqueeze(0).unsqueeze(0) # (1, 1, h, w)
        depth[torch.where(torch.isinf(depth))] = 1
        depth_start = depth.expand(1, self.spp, row, col)
        depth_start = depth_start[:,:,top:bottom,left:right].flatten()
        Y = torch.arange(0, row, device=dev)
        X = torch.arange(0, col, device=dev)
        Y, X = torch.meshgrid(Y, X) # (h, w)
        Y = Y[top:bottom,left:right]
        X = X[top:bottom,left:right]
        Y = Y.unsqueeze(0).expand(self.spp, irow, icol).flatten()
        X = X.unsqueeze(0).expand(self.spp, irow, icol).flatten()
        N = direction.size(0)
        indices = torch.zeros(N, dtype=torch.long, device=dev)

        if not self.use_parallel:
            ssrt_uv, mask, dz = ssrt(depth, normal, indices, proj, X, Y, direction, depth_start)
        else:
            depth_expand = depth.unsqueeze(0).expand(len(self.parallel_devices), -1, -1, -1, -1)
            normal_expand = normal.unsqueeze(0).expand(len(self.parallel_devices), -1, -1, -1, -1)
            proj_expand = proj.unsqueeze(0).expand(len(self.parallel_devices), -1, -1)
            ssrt_uv, mask, dz = self.parallel_ssrt(depth_expand, normal_expand, indices, proj_expand, X, Y, direction, depth_start)

        ssrt_uv[~mask,...] = -1
        uncertainty = torch.tanh(10 * dz)
        if self.ub is not None:
            certainty_left = torch.sigmoid((ssrt_uv[...,0:1] - self.ub//2) / self.ub)
            certainty_right = torch.sigmoid((col - 1 - self.ub//2 - ssrt_uv[...,0:1]) / self.ub)
            certainty_top = torch.sigmoid((ssrt_uv[...,1:2] - self.ub//2) / self.ub)
            certainty_bottom = torch.sigmoid((row - 1 - self.ub//2 - ssrt_uv[...,1:2]) / self.ub)
            uncertainty = 1 - certainty_left * certainty_right * certainty_top * certainty_bottom * (1 - uncertainty)
        ssrt_uv = ssrt_uv.flip(1) # xy->ij: x = j, y = i
        uncertainty[~mask,...] = 1
        ##############################
        ########## SSRT End ##########
        ##############################

        ##############################
        ###### Integrator Start ######
        ##############################
        vpos = vpos[:,:,top:bottom,left:right]
        positions = vpos.unsqueeze(1).expand(1, self.spp, 3, irow, icol).permute(0, 1, 3, 4, 2).reshape(-1, 3) # (bn*spp*h*w, 3)
        u = ssrt_uv[:,0]
        v = ssrt_uv[:,1]
        normals = normal[indices,:,u,v]
        roughs = rough[indices,:,u,v]
        Kd = albedo * (1 - metal)
        Ks = torch.lerp(torch.empty_like(albedo).fill_(0.04), albedo, metal)
        Kd = Kd[indices,:,u,v]
        Ks = Ks[indices,:,u,v]
        model_kwargs = {
            'uv': ssrt_uv, 'directions': direction,
            'positions': positions, 'index': indices,
            'uncertainty': uncertainty, 'normal': normals,
            'Kd': Kd, 'Ks': Ks, 'rough': roughs
        }

        light = get_light_chunk(model, im, model_kwargs, direction.size(0), self.chunk)
        light = light.view(1, self.spp, irow, icol, 3)
        light = light.permute(0, 1, 4, 2, 3) # (bn, spp, 3, h, w)

        colorDiffuse = torch.mean(brdfDiffuse * light, dim=1)
        colorSpec = torch.mean(brdfSpec * light, dim=1)
        ##############################
        ####### Integrator End #######
        ##############################

        return colorDiffuse, colorSpec, wi_mask

