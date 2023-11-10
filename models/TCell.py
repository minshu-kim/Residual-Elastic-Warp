import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import kornia.geometry.transform as transform

from models import register
from .utils import *
from .update import kp_tcell_dirichlet

from .corr import CorrBlock
from .TCell_extractor import BasicEncoderQuarter, SmallMotionEncoder
from .TPS_evaluator_dirichlet import TpsGridGen

@register('T-Cell')
class TCell(nn.Module):
    def __init__(self, lev=6, num_kp=16, threshold=4):
        super().__init__()

        self.iters_lev = lev
        self.num_kp = num_kp

        self.feat_size = 64
        self.base_size = 512
        threshold = 1 / np.sqrt(self.num_kp) / threshold
        self.tps_grid = TpsGridGen(out_h=64, out_w=64, grid_size=int(np.sqrt(num_kp)))

        tmp = torch.linspace(0, 1, steps=int(np.sqrt(self.num_kp)))
        X, Y = torch.meshgrid(tmp, tmp, indexing='xy')
        self.theta = torch.stack([X,Y], dim=-1).unsqueeze(0).view(1,-1,2).cuda()

        self.update_ctrlpts = kp_tcell_dirichlet(input_dim=164, hidden_dim=96, thres=threshold)
        self.encoder = BasicEncoderQuarter(output_dim=256, norm_fn='instance', inputdim=3)


    def initialize_flow(self, img, scale=1):
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//scale, W//scale).to(img.device)
        coords1 = coords_grid(N, H//scale, W//scale).to(img.device)

        return coords0, coords1


    def scaling_flow(self, flow, before, after, scale):
        return flow / (before - 1) * (after * scale - 1)


    def forward(self, ref, tgt, iters=6, scale=1):
        b, *_ = ref.shape

        feat1 = self.encoder(ref)
        feat2 = self.encoder(tgt)

        flows = []
        flow = torch.zeros(b, 2, *feat2.shape[-2:]).cuda()

        corr_fn = CorrBlock(feat2, feat1, num_levels=2, radius=4)
        coords0, coords1 = self.initialize_flow(feat1, scale=1)

        delta_stack = 0
        for itr in range(iters):
            corr = corr_fn(coords1)
            delta = self.update_ctrlpts(corr, flow)

            theta = self.theta.repeat(b,1,1)
            delta_stack = delta_stack + delta
            coords1 = self.tps_grid(delta_stack, theta, feat_size=self.feat_size)

            flow = coords1.permute(0,3,1,2) - coords0
            flow = self.scaling_flow(flow, self.feat_size, self.base_size, scale)
            flow_ = F.upsample_bilinear(flow, None, [int(8 * scale),int(8 * scale)]).permute(0,2,3,1)
            flows.append(flow_)

            coords1 = coords1.permute(0,3,1,2)

        return flows

