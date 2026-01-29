import torch
import torch.nn as nn
from torch.nn import functional as F

class SA(nn.Module):
    def __init__(self, in_channels, dilation=1):
        super(SA, self).__init__()
 
        self.s_split = nn.Conv2d(in_channels // 2, in_channels // 2 , 3, 1, 1)

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels //2, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(),
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels //2, in_channels // 2, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(),
        )

        self.conv = nn.Conv2d(2, 1, 7, padding=3)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        feat1 = x[:, 0::2, :, :]
        feat2 = x[:, 1::2, :, :]
        feat1 = self.s_split(feat1)
        feat2 = self.s_split(feat2)

        feat1 = self.branch1(feat1)
        feat2 = self.branch2(feat2)

        feat = torch.cat([feat1, feat2], dim=1)

        max_result, _ = torch.max(feat, dim=1, keepdim=True)
        mean_result = torch.mean(feat, dim=1, keepdim=True)

        result = torch.cat([max_result, mean_result], dim=1)

        result = self.conv(result)
        output = self.sigmoid(result)

        return feat * output

class CA(nn.Module):
    def __init__(self, in_channels, dilation=1):
        super(CA, self).__init__()

        self.c_split = nn.Conv2d(in_channels // 2, in_channels // 2 , 3, 1, 1)

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(),
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels // 2, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4 , 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, in_channels, 1),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        feat1 = x[:, 0::2, :, :]
        feat2 = x[:, 1::2, :, :]
        feat1 = self.c_split(feat1)
        feat2 = self.c_split(feat2)


        feat1 = self.branch1(feat1)
        feat2 = self.branch2(feat2)

        feat = torch.cat([feat1, feat2], dim=1)

        gap_result = self.mlp(self.gap(feat))
        gmp_result = self.mlp(self.gmp(feat))

        result = gap_result + gmp_result
        result = self.sigmoid(result)

        return feat * result



class SACA(nn.Module):
    def __init__(self, in_channels, dilation=1):
        super(SACA, self).__init__()

        self.sa = SA(in_channels, dilation)
        self.ca = CA(in_channels, dilation)

    def forward(self, x):

        s_x = self.sa(x)
        s_x = s_x + x

        c_x = self.ca(s_x)
        out = c_x + s_x

        return out


class CrossFusion(nn.Module):
    def __init__(self):
        super(CrossFusion, self).__init__()

        self.enhance_conv = nn.Sequential(
            nn.Conv2d(512, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 2, 3, padding=1),
            nn.ReLU(),
        )

    def forward(self, feat1, feat2):

        bs, c, h, w = feat1.shape

        feat1_first_half = feat1[:, : c//2, :]
        feat2_first_half = feat2[:, : c//2, :]

        feat1_swap = torch.cat(
            [feat1_first_half, feat2[:, c//2:, :]],
            1
        )

        feat2_swap = torch.cat(
            [feat2_first_half, feat1[:, c//2:, :]],
            1
        )

        feat_swap = torch.cat(
            [feat1_swap, feat2_swap], dim=1
        )


        W = F.softmax(self.enhance_conv(feat_swap), dim=1)
        x_fuse = feat1_swap * W[:, 0:1, :, :] + feat2_swap * W[:, 1:2, :, :]

        return x_fuse


class msf(nn.Module):
    def __init__(self):
        super(msf, self).__init__()

        self.saca1_1 = SACA(256, dilation=1)
        self.saca1_2 = SACA(256, dilation=1)

        self.saca2_1 = SACA(256, dilation=2)
        self.saca2_2 = SACA(256, dilation=2)

        self.saca3_1 = SACA(256, dilation=3)

        self.cf1 = CrossFusion()
        self.cf2 = CrossFusion()
        self.cf3 = CrossFusion()


    def forward(self, x):
        x1_1 = self.saca1_1(x)
        x1_2 = self.saca1_2(x1_1)

        x2_1 = self.saca2_1(x1_1)
        x2_2 = self.saca2_2(x2_1)

        x3_1 = self.saca3_1(x2_1)

        fuse1 = self.cf1(x1_2, x2_1)
        fuse2 = self.cf2(x2_2, x3_1)
        fuse = self.cf3(fuse1, fuse2)

        return fuse