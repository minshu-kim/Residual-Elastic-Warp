import torch
import torch.nn as nn
import torch.nn.functional as F
import torchgeometry as tgm

from models import register

from .utils import *
from .corr import CorrBlock
from .update import kp_hcell

from .HCell_extractor import BasicEncoderQuarter

autocast = torch.cuda.amp.autocast

@register('H-Cell')
class HCell(nn.Module):
    def __init__(self, constrain=True):
        super().__init__()
        self.device = torch.device('cuda:0')
        self.encoder = BasicEncoderQuarter(output_dim=256, norm_fn='instance')
        self.update_block_32 = kp_hcell(input_dim=164, hidden_dim=96)

        self.feat_size = 32
        self.base_size = 512

#        self.constrain = constrain

    def get_flow_now_4(self, four_point):
        four_point = four_point / 16
        four_point_org = torch.zeros((2, 2, 2)).to(four_point.device)
        four_point_org[:, 0, 0] = torch.Tensor([0, 0])
        four_point_org[:, 0, 1] = torch.Tensor([self.sz[3]-1, 0])
        four_point_org[:, 1, 0] = torch.Tensor([0, self.sz[2]-1])
        four_point_org[:, 1, 1] = torch.Tensor([self.sz[3]-1, self.sz[2]-1])

        four_point_org = four_point_org.unsqueeze(0)
        four_point_org = four_point_org.repeat(self.sz[0], 1, 1, 1)
        four_point_new = four_point_org + four_point
        four_point_org = four_point_org.flatten(2).permute(0, 2, 1)
        four_point_new = four_point_new.flatten(2).permute(0, 2, 1)
        H = tgm.get_perspective_transform(four_point_org, four_point_new)
        gridy, gridx = torch.meshgrid(torch.linspace(0, self.sz[3]-1, steps=self.sz[3]), torch.linspace(0, self.sz[2]-1, steps=self.sz[2]))
        points = torch.cat((gridx.flatten().unsqueeze(0), gridy.flatten().unsqueeze(0), torch.ones((1, self.sz[3] * self.sz[2]))),
                           dim=0).unsqueeze(0).repeat(self.sz[0], 1, 1).to(four_point.device)
        points_new = H.bmm(points)
        points_new = points_new / points_new[:, 2, :].unsqueeze(1)
        points_new = points_new[:, 0:2, :]
        flow = torch.cat((points_new[:, 0, :].reshape(self.sz[0], self.sz[3], self.sz[2]).unsqueeze(1),
                          points_new[:, 1, :].reshape(self.sz[0], self.sz[3], self.sz[2]).unsqueeze(1)), dim=1)
        return flow


    def initialize_flow_4(self, img):
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//16, W//16).to(img.device)
        coords1 = coords_grid(N, H//16, W//16).to(img.device)

        return coords0, coords1


    def scaling_flow(self, flow, before, after, scale):
        return flow / (before - 1) * (after * scale - 1)


    def forward(self, image1, image2, iters_lev0=6, scale=1):
        image1 = image1.contiguous()
        image2 = image2.contiguous()

        with autocast(enabled=False):
            fmap1 = self.encoder(image1).float()
            fmap2 = self.encoder(image2).float()

        corr_fn = CorrBlock(fmap1, fmap2, num_levels=2, radius=4)
        coords0, coords1 = self.initialize_flow_4(image1)

        sz = fmap1.shape
        self.sz = sz
        four_point_disp = torch.zeros((sz[0], 2, 2, 2)).to(fmap1.device)

        grid_loss = 0
        disps, flows = [], []

        for itr in range(iters_lev0):
            corr = corr_fn(coords1)
            flow = coords1 - coords0
            with autocast(enabled=False):
                delta_four_point = self.update_block_32(corr, flow)

            four_point_disp =  four_point_disp + delta_four_point
            """
            if not self.constrain:
                four_point_disp =  four_point_disp + delta_four_point
            else:
                four_point_disp =  four_point_disp + torch.tanh(delta_four_point) * 64 * scale
            """

            coords1 = self.get_flow_now_4(four_point_disp)
            disps.append(four_point_disp)
            flow = self.scaling_flow(flow, self.feat_size, self.base_size, scale)
            flow_ = F.upsample_bilinear(flow, None, [16*scale,16*scale]).permute(0,2,3,1)
#            flow_ = F.upsample_bilinear(flow/31*(512*scale-1), None, [16*scale,16*scale]).permute(0,2,3,1)
            flows.append(flow_)

        return flows, disps, grid_loss
