import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, X):
        h = F.relu(self.conv1_1(X))
        h = F.relu(self.conv1_2(h))
        relu1_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        relu2_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        relu3_3 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        relu4_3 = h
        return [relu1_2, relu2_2, relu3_3, relu4_3]

class MaskedL2Loss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor=None):
        loss = (output - target) ** 2
        if mask is not None:
            _, _, h, w = loss.shape
            _, _, hm, wm = mask.shape
            sh, sw = hm//h, wm//w
            mask0 = F.avg_pool2d(mask, kernel_size=(sh,sw), stride=(sh,sw)).expand_as(loss)
            loss = (loss * mask0).sum() / mask0.sum()
        else:
            loss = loss.mean()
        return loss

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = Vgg16()
        vgg_model = torchvision.models.vgg16(pretrained=True, progress=True)
        for (src, dst) in zip(vgg_model.parameters(), self.vgg.parameters()):
            dst.data[:] = src
            dst.requires_grad = False
        del vgg_model
        self.l2 = MaskedL2Loss()
    
    def forward(self, output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor=None, layers=[1]):
        conv_layers_output = self.vgg(output * mask)
        conv_layers_target = self.vgg(target * mask)
        loss_perception = 0
        for layer in layers:
            loss_perception += self.l2(conv_layers_output[layer], conv_layers_target[layer], mask)
        return loss_perception

def normal_loss(output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
    """
    Computes normal loss.
    L = | 1 - N \dot ~N |
    """
    dot = torch.sum((output * target), dim=1, keepdim=True)
    # loss = (1 - dot) ** 2
    # loss = torch.acos(dot)
    loss = torch.abs(1 - dot)
    if mask is not None:
        _, _, h, w = loss.shape
        _, _, hm, wm = mask.shape
        sh, sw = hm//h, wm//w
        mask0 = F.avg_pool2d(mask, kernel_size=(sh,sw), stride=(sh,sw)).expand_as(loss)
        loss = (loss * mask0).sum() / mask0.sum()
    else:
        loss = loss.mean()
    return loss