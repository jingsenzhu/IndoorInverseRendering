import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np

class LocalEncoder(nn.Module):
    def __init__(
        self,
        backbone="resnet34",
        pretrained=True,
        num_layers=4,
        latent_size=None,
        index_interp="bilinear",
        index_padding="border",
        upsample_interp="bilinear",
        use_first_pool=True,
        single_channel=False,
        use_normal=False,
        use_albedo=False,
        use_material=False
    ):
        super().__init__()
        self.backbone = getattr(torchvision.models, backbone)(pretrained=pretrained)
        c1_chan = 3 if not single_channel else 2
        if use_normal:
            c1_chan += 3
        if use_albedo:
            c1_chan += 3 if not single_channel else 2
        if use_material:
            c1_chan += 2
        if c1_chan != 3:
            self.backbone.conv1 = nn.Conv2d(c1_chan, 64, kernel_size=7, stride=2, padding=3, bias=False)
            nn.init.kaiming_normal_(self.backbone.conv1.weight, nonlinearity='relu')
        self.backbone.fc = nn.Identity()
        self.backbone.avgpool = nn.Identity()
        self.channels = [0, 64, 128, 256, 512, 1024][num_layers]
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.index_interp = index_interp
        self.index_padding = index_padding
        self.upsample_interp = upsample_interp
        self.use_first_pool = use_first_pool
        self.register_buffer("latent", torch.empty(1, 1, 1, 1), persistent=False)

    def forward(self, x):
        # assert x.size(0) == 1
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        latents = [x]
        if self.num_layers > 1:
            if self.use_first_pool:
                x = self.backbone.maxpool(x)
            x = self.backbone.layer1(x)
            latents.append(x)
        if self.num_layers > 2:
            x = self.backbone.layer2(x)
            latents.append(x)
        if self.num_layers > 3:
            x = self.backbone.layer3(x)
            latents.append(x)
        if self.num_layers > 4:
            x = self.backbone.layer4(x)
            latents.append(x)
        align_corners = None if self.upsample_interp == "nearest " else True
        latent_sz = latents[0].shape[-2:]
        for i in range(len(latents)):
            latents[i] = F.interpolate(
                latents[i],
                latent_sz,
                mode=self.upsample_interp,
                align_corners=align_corners,
            )
        self.latent = torch.cat(latents, dim=1)
        if self.latent_size is not None:
            self.latent = F.interpolate(
                self.latent,
                self.latent_size,
                mode=self.upsample_interp,
                align_corners=align_corners
            )
        return self.latent
    
    def index(self, uv, index, integer_uv=False):
        """
        :param uv (B, 2) float uv
        :param index (B,)
        :return (B, C)
        """
        H, W = self.latent.shape[2:]
        u, v = uv[:,0], uv[:,1]
        if not integer_uv:
            u = torch.clamp(torch.round(u * H - 0.5).long(), min=0, max=H)
            v = torch.clamp(torch.round(v * W - 0.5).long(), min=0, max=W)
        features = self.latent[index,:,u,v] # (B, C)
        if features.size(0) == 1:
            features = features.expand(uv.size(0), -1)
        return features

def init_weights_normal(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        nn.init.zeros_(m.bias)

class LocalMLP(nn.Module):
    def __init__(self, input_size, hidden_size=512, out_channel=3, num_layers=8, skips=[4]):
        super().__init__()
        self.skips = skips
        self.linears = nn.ModuleList(
            [nn.Linear(input_size, hidden_size)] + [nn.Linear(hidden_size, hidden_size) if i not in skips else nn.Linear(hidden_size + input_size, hidden_size) for i in range(num_layers)]
        )
        self.output_layer = nn.Linear(hidden_size, out_channel)

        nl, nl_weight_init, first_layer_init = nn.ReLU(inplace=True), init_weights_normal, None
        self.nl = nl
        self.output_activation = nn.ReLU(inplace=True)

        if nl_weight_init is not None:
            for layer in self.linears:
                layer.apply(nl_weight_init)
            self.output_layer.apply(nl_weight_init)
        if first_layer_init is not None:
            self.linears[0].apply(first_layer_init)
    
    def forward(self, x):
        h = x
        for i, _ in enumerate(self.linears):
            h = self.linears[i](h)
            h = self.nl(h)
            if i in self.skips:
                h = torch.cat([h, x], -1)
        h = self.output_layer(h)
        return self.output_activation(h)

class LocalNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = LocalEncoder(use_normal=cfg.encode_normal, **cfg.encoder)
        latent_size = self.encoder.channels
        
        self.use_pos = getattr(cfg, "use_pos", False)
        self.use_normal = getattr(cfg, "use_normal", False)
        self.use_rough = getattr(cfg, "use_rough", False)
        self.use_albedo = getattr(cfg, "use_albedo", False)

        def fourier_mapping(x: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
            xp = torch.matmul(2 * np.pi * x, B)
            return torch.cat([torch.sin(xp), torch.cos(xp)], dim=-1)
        self.encode_pos_fn = self.encode_dir_fn = fourier_mapping
        input_size = 2 * cfg.fourier.map_dim
        self.register_buffer('Bd', torch.randn(3, cfg.fourier.map_dim) * cfg.fourier.sigma, True)
        if self.use_pos:
            input_size += 2 * cfg.fourier.map_dim
            self.register_buffer('Bp', torch.randn(2, cfg.fourier.map_dim) * cfg.fourier.sigma, True)
        if self.use_normal:
            input_size += 2 * cfg.fourier.map_dim
            self.register_buffer('Bn', torch.randn(3, cfg.fourier.map_dim) * cfg.fourier.sigma, True)
        if self.use_albedo:
            input_size += 4 * cfg.fourier.map_dim
            self.register_buffer('Bkd', torch.randn(3, cfg.fourier.map_dim) * cfg.fourier.sigma, True)
            self.register_buffer('Bks', torch.randn(3, cfg.fourier.map_dim) * cfg.fourier.sigma, True)
        if self.use_rough:
            input_size += 2 * cfg.fourier.map_dim
            self.register_buffer('Br', torch.randn(1, cfg.fourier.map_dim) * cfg.fourier.sigma, True)
        
        self.mlp = LocalMLP(latent_size + input_size, out_channel=3)

    def encode(self, im):
        """
        :param im (S, 3 (or 6), H, W)
        """
        self.encoder(im)
    
    def forward(
        self, uv: torch.LongTensor, directions: torch.Tensor, index: torch.Tensor, 
        normal: torch.Tensor = None, Kd: torch.Tensor = None, Ks: torch.Tensor = None, rough: torch.Tensor = None
        ):
        """
        :param uv (B, 2)
        :param directions (B, 3)
        :param index (B,)
        :param (optional) normal, Kd, Ks, rough (B, 3)
        :return rgb (B, 3)
        """
        # uv_flipped = uv.flip((1,))
        # mask_ssrt = uv[:,0] >= 0 # (bn)
        # local_feature = torch.zeros(directions.size(0), self.encoder.latent_size, device=directions.device)
        local_feature = self.encoder.index(uv, index, True)
        if local_feature.size(0) == 1:
            local_feature = local_feature.expand(directions.size(0), -1)

        if hasattr(self, 'Bd'):
            # uv_encoded = self.encode_pos_fn(uv, self.Bp)
            dir_encoded = self.encode_dir_fn(directions, self.Bd)
        else:
            # uv_encoded = self.encode_pos_fn(uv)
            dir_encoded = self.encode_dir_fn(directions)
        # if uv_encoded.size(0) == 1:
        #     uv_encoded = uv_encoded.expand(directions.size(0), -1)
        input_pos = dir_encoded
        input_feature = local_feature
        if self.use_pos:
            if hasattr(self, 'Bp'):
                uv_encoded = self.encode_pos_fn(uv, self.Bp)
            else:
                uv_encoded = self.encode_pos_fn(uv)
            if uv_encoded.size(0) == 1:
                uv_encoded = uv_encoded.expand(input_pos.size(0), -1)
            input_pos = torch.cat([uv_encoded, input_pos], dim=-1)
        if self.use_normal:
            if hasattr(self, 'Bn'):
                normal_encoded = self.encode_dir_fn(normal, self.Bn)
            else:
                normal_encoded = self.encode_dir_fn(normal)
            if normal_encoded.size(0) == 1:
                normal_encoded = normal_encoded.expand(input_pos.size(0), -1)
            input_pos = torch.cat([input_pos, normal_encoded], dim=-1)
        if self.use_albedo:
            if hasattr(self, 'Bkd'):
                Kd_encoded = self.encode_dir_fn(Kd, self.Bkd)
                Ks_encoded = self.encode_dir_fn(Ks, self.Bks)
            else:
                Kd_encoded = self.encode_dir_fn(Kd)
                Ks_encoded = self.encode_dir_fn(Ks)
            if Kd_encoded.size(0) == 1:
                Kd_encoded = Kd_encoded.expand(input_pos.size(0), -1)
                Ks_encoded = Ks_encoded.expand(input_pos.size(0), -1)
            input_pos = torch.cat([input_pos, Kd_encoded, Ks_encoded], dim=-1)
        if self.use_rough:
            if hasattr(self, 'Br'):
                rough_encoded = self.encode_dir_fn(rough, self.Br)
            else:
                rough_encoded = self.encode_dir_fn(rough)
            if rough_encoded.size(0) == 1:
                rough_encoded = rough_encoded.expand(input_pos.size(0), -1)
            input_pos = torch.cat([input_pos, rough_encoded], dim=-1)

        inputs = torch.cat([input_pos, input_feature], dim=-1)
        return torch.clamp(self.mlp(inputs), max=100)