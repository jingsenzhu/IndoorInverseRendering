import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as tvm

class EncoderMG(nn.Module ):
    def __init__(self, cascadeLevel = 0, isSeg = False):
        super(EncoderMG, self).__init__()
        self.isSeg = isSeg
        self.cascadeLevel = cascadeLevel

        self.pad1 = nn.ReplicationPad2d(1)
        if self.cascadeLevel == 0:
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, bias =True )
        else:
            self.conv1 = nn.Conv2d(in_channels=12, out_channels = 64, kernel_size=4, stride =2, bias = True )

        self.gn1 = nn.GroupNorm(num_groups=4, num_channels=64)

        self.pad2 = nn.ZeroPad2d(1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, bias=True)
        self.gn2 = nn.GroupNorm(num_groups=8, num_channels=128)

        self.pad3 = nn.ZeroPad2d(1)
        self.conv3 = nn.Conv2d(in_channels = 128, out_channels=256, kernel_size=4, stride=2, bias=True)
        self.gn3 = nn.GroupNorm(num_groups=16, num_channels=256)

        self.pad4 = nn.ZeroPad2d(1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, bias=True)
        self.gn4 = nn.GroupNorm(num_groups=16, num_channels=256)

        self.pad5 = nn.ZeroPad2d(1)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, bias=True)
        self.gn5 = nn.GroupNorm(num_groups=32, num_channels=512)

        self.pad6 = nn.ZeroPad2d(1)
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, bias=True)
        self.gn6 = nn.GroupNorm(num_groups=64, num_channels=1024)

    def forward(self, x):
        x1 = F.relu(self.gn1(self.conv1(self.pad1(x) ) ), True)
        x2 = F.relu(self.gn2(self.conv2(self.pad2(x1) ) ), True)
        x3 = F.relu(self.gn3(self.conv3(self.pad3(x2) ) ), True)
        x4 = F.relu(self.gn4(self.conv4(self.pad4(x3) ) ), True)
        x5 = F.relu(self.gn5(self.conv5(self.pad5(x4) ) ), True)
        x6 = F.relu(self.gn6(self.conv6(self.pad6(x5) ) ), True)

        return x1, x2, x3, x4, x5, x6


class EncoderDenseNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.densenet = tvm.densenet121(pretrained=True)

    def forward(self, x):
        x1 = self.densenet.features[:3](x)
        x2 = self.densenet.features[4:6](x1)
        x3 = self.densenet.features[6:8](x2)
        x4 = self.densenet.features[8:10](x3)
        x5 = self.densenet.features[10:](x4)
        return x1, x2, x3, x4, x5


class EncoderResNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        resnet = tvm.resnet34(pretrained=True)
        self.layer0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
    
    def forward(self, x):
        x1 = self.layer0(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        return x1, x2, x3, x4, x5


class DecoderResNet(nn.Module):
    def __init__(self, mode=0):
        super().__init__()
        self.mode = mode

        self.dconv1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding = 1, bias=True)
        self.dgn1 = nn.GroupNorm(num_groups=8, num_channels=256 )

        self.dconv2 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, stride=1, padding = 1, bias=True)
        self.dgn2 = nn.GroupNorm(num_groups=8, num_channels=128 )

        self.dconv3 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.dgn3 = nn.GroupNorm(num_groups=4, num_channels=64 )

        self.dconv4 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding = 1, bias=True)
        self.dgn4 = nn.GroupNorm(num_groups=4, num_channels=64 )

        self.dconv5 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding = 1, bias=True)
        self.dgn5 = nn.GroupNorm(num_groups=4, num_channels=64 )

        self.dpadFinal = nn.ReplicationPad2d(1)
        self.dconvFinal = nn.Conv2d(in_channels=64, out_channels=(2 if mode == 3 else 3), kernel_size = 3, stride=1, bias=True)

    def forward(self, im, x1, x2, x3, x4, x5):
        dx1 = F.relu(self.dgn1(self.dconv1(x5 ) ) )

        if dx1.size(3) != x4.size(3) or dx1.size(2) != x4.size(2):
            dx1 = F.interpolate(dx1, [x4.size(2), x4.size(3)], mode='bilinear', align_corners=True)
        xin1 = torch.cat([dx1, x4], dim = 1)
        dx2 = F.relu(self.dgn2(self.dconv2(F.interpolate(xin1, scale_factor=2, mode='bilinear', align_corners=True) ) ), True)

        if dx2.size(3) != x3.size(3) or dx2.size(2) != x3.size(2):
            dx2 = F.interpolate(dx2, [x3.size(2), x3.size(3)], mode='bilinear', align_corners=True)
        xin2 = torch.cat([dx2, x3], dim=1 )
        dx3 = F.relu(self.dgn3(self.dconv3(F.interpolate(xin2, scale_factor=2, mode='bilinear', align_corners=True) ) ), True)

        if dx3.size(3) != x2.size(3) or dx3.size(2) != x2.size(2):
            dx3 = F.interpolate(dx3, [x2.size(2), x2.size(3)], mode='bilinear', align_corners=True)
        xin3 = torch.cat([dx3, x2], dim=1)
        dx4 = F.relu(self.dgn4(self.dconv4(F.interpolate(xin3, scale_factor=2, mode='bilinear', align_corners=True) ) ), True)

        if dx4.size(3) != x1.size(3) or dx4.size(2) != x1.size(2):
            dx4 = F.interpolate(dx4, [x1.size(2), x1.size(3)], mode='bilinear', align_corners=True)
        xin4 = torch.cat([dx4, x1], dim=1 )
        dx5 = F.relu(self.dgn5(self.dconv5(F.interpolate(xin4, scale_factor=2, mode='bilinear', align_corners=True) ) ), True)

        if dx5.size(3) != im.size(3) or dx5.size(2) != im.size(2):
            dx5 = F.interpolate(dx5, [im.size(2), im.size(3)], mode='bilinear', align_corners=True)
        x_orig = self.dconvFinal(self.dpadFinal(dx5 ) )

        if self.mode == 0 or self.mode == 3:
            x_out = torch.clamp(1.01 * torch.sigmoid(x_orig), 0, 1)
        elif self.mode == 1:
            x_orig = torch.clamp(1.01 * torch.tanh(x_orig ), -1, 1)
            norm = torch.sqrt(torch.sum(x_orig * x_orig, dim=1, keepdim=True)).expand_as(x_orig)
            x_out = x_orig / torch.clamp(norm, min=1e-6)
        elif self.mode == 2:
            x_out = torch.clamp(1.01 * torch.sigmoid(x_orig), 0, 1)
            x_out = torch.mean(x_orig, dim=1).unsqueeze(1)
        elif self.mode == 4:
            x_orig = torch.mean(x_orig, dim=1).unsqueeze(1)
            x_out = torch.clamp(1.01 * torch.sigmoid(x_orig), 0, 1)
        return x_out



class DecoderDenseNet(nn.Module):
    def __init__(self, mode=0):
        super().__init__()
        self.mode = mode

        self.dconv1 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding = 1, bias=True)
        self.dgn1 = nn.GroupNorm(num_groups=32, num_channels=512 )

        self.dconv2 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=3, stride=1, padding = 1, bias=True)
        self.dgn2 = nn.GroupNorm(num_groups=16, num_channels=256 )

        self.dconv3 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)
        self.dgn3 = nn.GroupNorm(num_groups=16, num_channels=128 )

        self.dconv4 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, stride=1, padding = 1, bias=True)
        self.dgn4 = nn.GroupNorm(num_groups=8, num_channels=64 )

        self.dconv5 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding = 1, bias=True)
        self.dgn5 = nn.GroupNorm(num_groups=4, num_channels=64 )

        self.dpadFinal = nn.ReplicationPad2d(1)
        self.dconvFinal = nn.Conv2d(in_channels=64, out_channels=(2 if mode == 3 else 3), kernel_size = 3, stride=1, bias=True)

    def forward(self, im, x1, x2, x3, x4, x5):
        dx1 = F.relu(self.dgn1(self.dconv1(x5 ) ) )

        xin1 = torch.cat([dx1, x4], dim = 1)
        dx2 = F.relu(self.dgn2(self.dconv2(F.interpolate(xin1, scale_factor=2, mode='bilinear', align_corners=True) ) ), True)

        if dx2.size(3) != x3.size(3) or dx2.size(2) != x3.size(2):
            dx2 = F.interpolate(dx2, [x3.size(2), x3.size(3)], mode='bilinear', align_corners=True)
        xin2 = torch.cat([dx2, x3], dim=1 )
        dx3 = F.relu(self.dgn3(self.dconv3(F.interpolate(xin2, scale_factor=2, mode='bilinear', align_corners=True) ) ), True)

        if dx3.size(3) != x2.size(3) or dx3.size(2) != x2.size(2):
            dx3 = F.interpolate(dx3, [x2.size(2), x2.size(3)], mode='bilinear', align_corners=True)
        xin3 = torch.cat([dx3, x2], dim=1)
        dx4 = F.relu(self.dgn4(self.dconv4(F.interpolate(xin3, scale_factor=2, mode='bilinear', align_corners=True) ) ), True)

        if dx4.size(3) != x1.size(3) or dx4.size(2) != x1.size(2):
            dx4 = F.interpolate(dx4, [x1.size(2), x1.size(3)], mode='bilinear', align_corners=True)
        xin4 = torch.cat([dx4, x1], dim=1 )
        dx5 = F.relu(self.dgn5(self.dconv5(F.interpolate(xin4, scale_factor=2, mode='bilinear', align_corners=True) ) ), True)

        if dx5.size(3) != im.size(3) or dx5.size(2) != im.size(2):
            dx5 = F.interpolate(dx5, [im.size(2), im.size(3)], mode='bilinear', align_corners=True)
        x_orig = self.dconvFinal(self.dpadFinal(dx5 ) )

        if self.mode == 0 or self.mode == 3:
            x_out = torch.clamp(1.01 * torch.sigmoid(x_orig), 0, 1)
        elif self.mode == 1:
            x_orig = torch.clamp(1.01 * torch.tanh(x_orig ), -1, 1)
            norm = torch.sqrt(torch.sum(x_orig * x_orig, dim=1, keepdim=True)).expand_as(x_orig)
            x_out = x_orig / torch.clamp(norm, min=1e-6)
        elif self.mode == 2:
            x_out = torch.clamp(1.01 * torch.sigmoid(x_orig), 0, 1)
            x_out = torch.mean(x_orig, dim=1).unsqueeze(1)
        # elif self.mode == 3: Unused
        #     x_out = F.softmax(x_orig, dim=1)
        elif self.mode == 4:
            x_orig = torch.mean(x_orig, dim=1).unsqueeze(1)
            x_out = torch.clamp(1.01 * torch.sigmoid(x_orig), 0, 1)
        return x_out



class DecoderMG(nn.Module ):
    def __init__(self, mode=0):
        super(DecoderMG, self).__init__()
        self.mode = mode

        self.dconv1 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding = 1, bias=True)
        self.dgn1 = nn.GroupNorm(num_groups=32, num_channels=512 )

        self.dconv2 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=3, stride=1, padding = 1, bias=True)
        self.dgn2 = nn.GroupNorm(num_groups=16, num_channels=256 )

        self.dconv3 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True)
        self.dgn3 = nn.GroupNorm(num_groups=16, num_channels=256 )

        self.dconv4 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, stride=1, padding = 1, bias=True)
        self.dgn4 = nn.GroupNorm(num_groups=8, num_channels=128 )

        self.dconv5 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, stride=1, padding = 1, bias=True)
        self.dgn5 = nn.GroupNorm(num_groups=4, num_channels=64 )

        self.dconv6 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding = 1, bias=True)
        self.dgn6 = nn.GroupNorm(num_groups=4, num_channels=64 )

        self.dpadFinal = nn.ReplicationPad2d(1)
        self.dconvFinal = nn.Conv2d(in_channels=64, out_channels=(2 if mode == 3 else 3), kernel_size = 3, stride=1, bias=True)

    def forward(self, im, x1, x2, x3, x4, x5, x6 ):
        dx1 = F.relu(self.dgn1(self.dconv1(x6 ) ) )

        xin1 = torch.cat([dx1, x5], dim = 1)
        dx2 = F.relu(self.dgn2(self.dconv2(F.interpolate(xin1, scale_factor=2, mode='bilinear', align_corners=True) ) ), True)

        if dx2.size(3) != x4.size(3) or dx2.size(2) != x4.size(2):
            dx2 = F.interpolate(dx2, [x4.size(2), x4.size(3)], mode='bilinear', align_corners=True)
        xin2 = torch.cat([dx2, x4], dim=1 )
        dx3 = F.relu(self.dgn3(self.dconv3(F.interpolate(xin2, scale_factor=2, mode='bilinear', align_corners=True) ) ), True)

        if dx3.size(3) != x3.size(3) or dx3.size(2) != x3.size(2):
            dx3 = F.interpolate(dx3, [x3.size(2), x3.size(3)], mode='bilinear', align_corners=True)
        xin3 = torch.cat([dx3, x3], dim=1)
        dx4 = F.relu(self.dgn4(self.dconv4(F.interpolate(xin3, scale_factor=2, mode='bilinear', align_corners=True) ) ), True)

        if dx4.size(3) != x2.size(3) or dx4.size(2) != x2.size(2):
            dx4 = F.interpolate(dx4, [x2.size(2), x2.size(3)], mode='bilinear', align_corners=True)
        xin4 = torch.cat([dx4, x2], dim=1 )
        dx5 = F.relu(self.dgn5(self.dconv5(F.interpolate(xin4, scale_factor=2, mode='bilinear', align_corners=True) ) ), True)

        if dx5.size(3) != x1.size(3) or dx5.size(2) != x1.size(2):
            dx5 = F.interpolate(dx5, [x1.size(2), x1.size(3)], mode='bilinear', align_corners=True)
        xin5 = torch.cat([dx5, x1], dim=1 )
        dx6 = F.relu(self.dgn6(self.dconv6(F.interpolate(xin5, scale_factor=2, mode='bilinear', align_corners=True) ) ), True)

        if dx6.size(3) != im.size(3) or dx6.size(2) != im.size(2):
            dx6 = F.interpolate(dx6, [im.size(2), im.size(3)], mode='bilinear', align_corners=True)
        x_orig = self.dconvFinal(self.dpadFinal(dx6 ) )

        if self.mode == 0 or self.mode == 3:
            x_out = torch.clamp(1.01 * torch.sigmoid(x_orig), 0, 1)
        elif self.mode == 1:
            x_orig = torch.clamp(1.01 * torch.tanh(x_orig ), -1, 1)
            norm = torch.sqrt(torch.sum(x_orig * x_orig, dim=1).unsqueeze(1) ).expand_as(x_orig)
            x_out = x_orig / torch.clamp(norm, min=1e-6)
        elif self.mode == 2:
            x_out = torch.clamp(1.01 * torch.sigmoid(x_orig), 0, 1)
            x_out = torch.mean(x_orig, dim=1).unsqueeze(1)
        # elif self.mode == 3: Unused
        #     x_out = F.softmax(x_orig, dim=1)
        elif self.mode == 4:
            x_orig = torch.mean(x_orig, dim=1).unsqueeze(1)
            x_out = torch.clamp(1.01 * torch.sigmoid(x_orig), 0, 1)
        return x_out

