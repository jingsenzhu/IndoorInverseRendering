import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

def positional_encoding(
    tensor, num_encoding_functions=6, include_input=True, log_sampling=True
) -> torch.Tensor:
    r"""Apply positional encoding to the input.

    Args:
        tensor (torch.Tensor): Input tensor to be positionally encoded.
        encoding_size (optional, int): Number of encoding functions used to compute
            a positional encoding (default: 6).
        include_input (optional, bool): Whether or not to include the input in the
            positional encoding (default: True).

    Returns:
    (torch.Tensor): Positional encoding of the input tensor.
    """
    # TESTED
    # Trivially, the input tensor is added to the positional encoding.
    encoding = [tensor] if include_input else []
    frequency_bands = None
    if log_sampling:
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    else:
        frequency_bands = torch.linspace(
            2.0 ** 0.0,
            2.0 ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    for freq in frequency_bands:
        for func in [torch.sin, torch.cos]:
            encoding.append(func(tensor * freq))

    # Special case, for no positional encoding
    if len(encoding) == 1:
        return encoding[0]
    else:
        return torch.cat(encoding, dim=-1)


def intersect_sphere(positions: torch.Tensor, directions: torch.Tensor, radius):
    """
    Ray (p, d) intersect with sphere P \dot P = r^2
    """
    a = 1
    b = 2 * torch.sum(positions * directions, dim=1, keepdim=True)
    c = torch.sum(directions * directions, dim=1, keepdim=True) - radius * radius
    delta = b * b - 4 * a * c
    mask = delta[:,0] >= 0
    delta[~mask] = 0
    discriminant = torch.sqrt(delta)
    t0 = (-b + discriminant) / (2 * a)
    t1 = (-b - discriminant) / (2 * a)
    t = torch.where(t1 > 0, t1, t0)
    mask &= t[:,0] > 0
    return t, mask

class HyperBlock(nn.Module):
    def __init__(self, in_size, hidden_size, hyper_dim) -> None:
        super().__init__()
        self.hyper_dim = hyper_dim
        out_size = hyper_dim[0] * hyper_dim[1]
        self.layer = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, out_size),
            nn.LeakyReLU(0.1, True)
        )
    
    def forward(self, x):
        x = self.layer(x)
        return x.view(-1, self.hyper_dim[0], self.hyper_dim[1])

class HyperNetwork(nn.Module):
    def __init__(self, hyper_dims, in_size=512) -> None:
        super().__init__()
        self.hyper_dims = hyper_dims
        self.hyper_nets = nn.ModuleList()
        self.initial_block = nn.Sequential(
            nn.Linear(in_size, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True)
        )
        for dim in hyper_dims:
            self.hyper_nets.append(HyperBlock(1024, 4096, dim))
    
    def forward(self, x):
        x = self.initial_block(x)
        results = []
        for layer in self.hyper_nets:
            results.append(layer(x))
        return results

class UnweightedNeRF(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, weights, biases, index):
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)
        for w, b in zip(weights, biases):
            uidx = torch.unique_consecutive(index)
            if (len(uidx) == 1):
                h = torch.bmm(w[uidx,...].expand(x.size(0), -1, -1), x) + b[uidx,...].expand(x.size(0), -1, -1)
            else:
                h = torch.bmm(w[index,...], x) + b[index,...]
            x = F.relu(h, False)
        return x.squeeze(-1)

class GlobalEncoder(nn.Module):
    def __init__(
        self,
        pretrained=True,
        latent_size=512
    ):
        super().__init__()
        self.backbone = torchvision.models.resnet34(pretrained=pretrained)
        self.backbone.fc = nn.Identity()
        self.latent_size = latent_size
        if latent_size != 512:
            self.fc = nn.Linear(512, latent_size)
            nn.init.kaiming_normal_(self.fc.weight, nonlinearity='relu')
            nn.init.zeros_(self.fc.bias)
    
    def forward(self, x):
        # assert(x.size(0) == 1)
        x = self.backbone(x)
        if self.latent_size != 512:
            x = self.fc(x)
        return x

class HyperModel(nn.Module):
    def __init__(self, num_encoding_functions = 6) -> None:
        super().__init__()
        self.encoder = GlobalEncoder(latent_size=512)
        self.num_encoding_functions = num_encoding_functions
        hyper_dim_weights = [
            [64, 3 + 3 * 2 * num_encoding_functions], [64, 64], [64, 64], [4, 64]
        ]
        hyper_dim_biases = [
            [64, 1], [64, 1], [64, 1], [4, 1]
        ]
        self.hypernet_weight = HyperNetwork(hyper_dim_weights)
        self.hypernet_bias = HyperNetwork(hyper_dim_biases)
        self.decoder = UnweightedNeRF()

    def forward(self, im, positions, index):
        global_feature = self.encoder(im)
        weights = self.hypernet_weight(global_feature)
        biases = self.hypernet_bias(global_feature)
        pos_enc = positional_encoding(positions, self.num_encoding_functions)
        return self.decoder(pos_enc, weights, biases, index)

class GlobalNet(nn.Module):
    def __init__(self, n_samples=64, eval_batch_size=1000000, num_encoding_functions = 6) -> None:
        super().__init__()
        self.model = HyperModel(num_encoding_functions)
        self.n_samples = n_samples
        self.eval_batch_size = eval_batch_size
        self.imHeight = 240
        self.imWidth = 320
    
    def sample_rays(self, near, far):
        step = 1.0 / self.n_samples
        B = far.size(0)
        device = far.device
        z_steps = torch.linspace(0, 1 - step, self.n_samples, device=device)  # (Kc)
        z_steps = z_steps.unsqueeze(0).repeat(B, 1)  # (B, Kc)
        z_steps += torch.rand_like(z_steps) * step
        return near * (1 - z_steps) + far * z_steps  # (B, Kc)
    
    def composite(self, rays, z_samp, im, index):
        B, K = z_samp.shape

        deltas = z_samp[:, 1:] - z_samp[:, :-1]  # (B, K-1)
        delta_inf = rays[:, -1:] - z_samp[:, -1:]
        deltas = torch.cat([deltas, delta_inf], -1)  # (B, K)

        # (B, K, 3)
        points = rays[:, None, :3] + z_samp.unsqueeze(2) * rays[:, None, 3:6]
        points = points.view(-1, 3)  # (B*K, 3)
        index = index.unsqueeze(1).expand(B, K).reshape(-1) # (B*K)

        if self.eval_batch_size > 0:
            val_all = []
            split_points = torch.split(points, self.eval_batch_size, dim=0)
            split_indices = torch.split(index, self.eval_batch_size, dim=0)
            for pnts, idxs in zip(split_points, split_indices):
                val_all.append(self.model(im, pnts, idxs))
            out = torch.cat(val_all, dim=0)
        else:
            out = self.model(im, points, index)
        points = None
        out = out.view(B, K, -1)  # (B, K, 4)
        rgbs = out[..., :3]  # (B, K, 3)
        sigmas = out[..., 3]  # (B, K)

        alphas = 1 - torch.exp(-deltas * sigmas)  # (B, K)
        deltas = None
        sigmas = None
        alphas_shifted = torch.cat(
            [torch.ones_like(alphas[:, :1]), 1 - alphas + 1e-10], -1
        )  # (B, K+1) = [1, a1, a2, ...]
        T = torch.cumprod(alphas_shifted, -1)
        weights = alphas * T[:, :-1]  # (B, K)
        alphas = None
        alphas_shifted = None

        rgb_final = torch.sum(weights.unsqueeze(-1) * rgbs, -2)  # (B, 3)
        return rgb_final
    
    def forward(self, im, directions, positions, index):
        # im = im[0,...]
        t_far, mask = intersect_sphere(positions, directions, 3.0)
        if not torch.all(mask):
            if torch.any(~torch.isfinite(positions)):
                print("Pos failed")
            if torch.any(~torch.isfinite(directions)):
                print("Dir failed")
        assert(torch.all(mask))
        t_samples = self.sample_rays(0.05, t_far)
        ray = torch.cat([positions, directions, t_far], dim=1)
        im = F.interpolate(im, (self.imHeight, self.imWidth), mode='area')
        return self.composite(ray, t_samples, im, index)