import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import warp, get_robust_weight
from losses import Charbonnier_L1, Ternary, Charbonnier_Ada, Geometry


def resize(x, scale_factor):
    return F.interpolate(x, scale_factor=scale_factor, mode="bilinear", align_corners=False)


def conv_prelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias),
        nn.PReLU(out_channels)
    )



class ResBlock(nn.Module):
    def __init__(self, in_channels, side_channels):
        super().__init__()
        self.side_channels = side_channels
        self.main_path = nn.Sequential(
            conv_prelu(in_channels, in_channels, kernel_size=3, padding=1),
            conv_prelu(in_channels, in_channels, kernel_size=3, padding=1)
        )
        self.side_path = nn.Sequential(
            conv_prelu(side_channels, side_channels, kernel_size=3, padding=1),
            conv_prelu(side_channels, side_channels, kernel_size=3, padding=1)
        )
        self.out_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.out_prelu = nn.PReLU(in_channels)

    def forward(self, x):
        out = self.main_path[0](x)
        out[:, -self.side_channels:] = self.side_path[0](out[:, -self.side_channels:].clone())
        out = self.main_path[1](out)
        out[:, -self.side_channels:] = self.side_path[1](out[:, -self.side_channels:].clone())
        return self.out_prelu(x + self.out_conv(out))
    

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.pyramid1 = nn.Sequential(
            conv_prelu(1, 32, 3, 2, 1), 
            conv_prelu(32, 32, 3, 1, 1)
        )
        self.pyramid2 = nn.Sequential(
            conv_prelu(32, 48, 3, 2, 1), 
            conv_prelu(48, 48, 3, 1, 1)
        )
        self.pyramid3 = nn.Sequential(
            conv_prelu(48, 72, 3, 2, 1), 
            conv_prelu(72, 72, 3, 1, 1)
        )
        self.pyramid4 = nn.Sequential(
            conv_prelu(72, 96, 3, 2, 1), 
            conv_prelu(96, 96, 3, 1, 1)
        )
        
    def forward(self, img):
        f1 = self.pyramid1(img)
        f2 = self.pyramid2(f1)
        f3 = self.pyramid3(f2)
        f4 = self.pyramid4(f3)
        return f1, f2, f3, f4


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            conv_prelu(in_channels, mid_channels, kernel_size=3, padding=1),
            ResBlock(mid_channels, 32),
            nn.ConvTranspose2d(mid_channels, out_channels, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, *inputs):
        return self.net(torch.cat(inputs, dim=1))
    

class Model(nn.Module):
    def __init__(self, local_rank=-1, lr=1e-4):
        super(Model, self).__init__()
        self.encoder = Encoder()
        self.decoder4 = DecoderBlock(193, 192, 76)
        self.decoder3 = DecoderBlock(220, 216, 52)
        self.decoder2 = DecoderBlock(148, 144, 36)
        self.decoder1 = DecoderBlock(100, 96, 8)


        # Loss functions
        self.l1_loss = Charbonnier_L1()
        self.tr_loss = Ternary(7)
        self.rb_loss = Charbonnier_Ada()
        self.gc_loss = Geometry(3)

    
    def forward(self, img0, img1, embt, imgt, flow=None):

        mean_ = torch.cat([img0, img1], 2).mean(1, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
        img0 = img0 - mean_
        img1 = img1 - mean_
        imgt_ = imgt - mean_
        
        f0 = self.encoder(img0)
        f1 = self.encoder(img1)
        ft = self.encoder(imgt_)

        # Decode layer 4
        out4 = self.decoder4(f0[3], f1[3], embt.repeat(1, 1, *f0[3].shape[-2:]))
        u0_4, u1_4, ft_3_ = out4[:, :2], out4[:, 2:4], out4[:, 4:]

        # Decode layer 3
        out3 = self.decoder3(ft_3_, warp(f0[2], u0_4), warp(f1[2], u1_4), u0_4, u1_4)
        u0_3, u1_3, ft_2_ = out3[:, :2] + 2 * resize(u0_4, 2), out3[:, 2:4] + 2 * resize(u1_4, 2), out3[:, 4:]

        # Decode layer 2
        out2 = self.decoder2(ft_2_, warp(f0[1], u0_3), warp(f1[1], u1_3), u0_3, u1_3)
        u0_2, u1_2, ft_1_ = out2[:, :2] + 2 * resize(u0_3, 2), out2[:, 2:4] + 2 * resize(u1_3, 2), out2[:, 4:]

        # Decode layer 1
        out1 = self.decoder1(ft_1_, warp(f0[0], u0_2), warp(f1[0], u1_2), u0_2, u1_2)
        u0_1 = out1[:, :2] + 2 * resize(u0_2, 2)
        u1_1 = out1[:, 2:4] + 2 * resize(u1_2, 2)
        mask, res = torch.sigmoid(out1[:, 4:5]), out1[:, 5:]

        # Final merge
        img0_warp = warp(img0, u0_1)
        img1_warp = warp(img1, u1_1)
        merged = mask * img0_warp + (1 - mask) * img1_warp + mean_
        pred = torch.clamp(merged + res, 0, 1)

        # out4 = self.decoder4(f0[3], f1[3], embt)
        # u0_4, u1_4, ft_3_ = out4[:, :2], out4[:, 2:4], out4[:, 4:]

        # out3 = self.decoder3(ft_3_, f0[2], f1[2], u0_4, u1_4)
        # u0_3, u1_3, ft_2_ = out3[:, :2] + 2 * resize(u0_4, 2), out3[:, 2:4] + 2 * resize(u1_4, 2), out3[:, 4:]

        # out2 = self.decoder2(ft_2_, f0[1], f1[1], u0_3, u1_3)
        # u0_2, u1_2, ft_1_ = out2[:, :2] + 2 * resize(u0_3, 2), out2[:, 2:4] + 2 * resize(u1_3, 2), out2[:, 4:]

        # out1 = self.decoder1(ft_1_, f0[0], f1[0], u0_2, u1_2)
        # u0_1 = out1[:, :2] + 2 * resize(u0_2, 2)
        # u1_1 = out1[:, 2:4] + 2 * resize(u1_2, 2)
        # mask, res = torch.sigmoid(out1[:, 4:5]), out1[:, 5:]

        loss_rec = self.l1_loss(pred - imgt) + self.tr_loss(pred, imgt)
        loss_geo = 0.01 * (self.gc_loss(ft_1_, ft[0]) + self.gc_loss(ft_2_, ft[1]) + self.gc_loss(ft_3_, ft[2]))

        if flow is not None:
            w0, w1 = get_robust_weight(u0_1, flow[:, :2]), get_robust_weight(u1_1, flow[:, 2:4])
            loss_dis = sum([
                self.rb_loss(2 * resize(u0_2, 2) - flow[:, :2], w0),
                self.rb_loss(2 * resize(u1_2, 2) - flow[:, 2:4], w1),
                self.rb_loss(4 * resize(u0_3, 4) - flow[:, :2], w0),
                self.rb_loss(4 * resize(u1_3, 4) - flow[:, 2:4], w1),
                self.rb_loss(8 * resize(u0_4, 8) - flow[:, :2], w0),
                self.rb_loss(8 * resize(u1_4, 8) - flow[:, 2:4], w1),
            ]) * 0.01
            
        else:
            loss_dis = 0.0 * loss_geo

        return pred, loss_rec, loss_geo, loss_dis

    def inference(self, img0, img1, embt, scale_factor=1.0):
        mean_ = torch.cat([img0, img1], 2).mean(1, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
        img0 = img0 - mean_
        img1 = img1 - mean_
        img0_ = resize(img0, scale_factor=scale_factor)
        img1_ = resize(img1, scale_factor=scale_factor)

        f0 = self.encoder(img0_)
        f1 = self.encoder(img1_)

        out4 = self.decoder4(f0[3], f1[3], embt.repeat(1, 1, *f0[3].shape[-2:]))
        u0_4, u1_4, ft_3_ = out4[:, :2], out4[:, 2:4], out4[:, 4:]

        out3 = self.decoder3(ft_3_, warp(f0[2], u0_4), warp(f1[2], u1_4), u0_4, u1_4)
        u0_3, u1_3, ft_2_ = out3[:, :2] + 2 * resize(u0_4, 2), out3[:, 2:4] + 2 * resize(u1_4, 2), out3[:, 4:]

        out2 = self.decoder2(ft_2_, warp(f0[1], u0_3), warp(f1[1], u1_3), u0_3, u1_3)
        u0_2, u1_2, ft_1_ = out2[:, :2] + 2 * resize(u0_3, 2), out2[:, 2:4] + 2 * resize(u1_3, 2), out2[:, 4:]

        out1 = self.decoder1(ft_1_, warp(f0[0], u0_2), warp(f1[0], u1_2), u0_2, u1_2)
        u0_1 = resize(out1[:, :2] + 2 * resize(u0_2, 2), 1 / scale_factor) / scale_factor
        u1_1 = resize(out1[:, 2:4] + 2 * resize(u1_2, 2), 1 / scale_factor) / scale_factor
        mask = resize(torch.sigmoid(out1[:, 4:5]), 1 / scale_factor)
        res = resize(out1[:, 5:], 1 / scale_factor)

        merged = mask * warp(img0, u0_1) + (1 - mask) * warp(img1, u1_1) + mean_
        imgt_pred = torch.clamp(merged + res, 0, 1)
        return imgt_pred


