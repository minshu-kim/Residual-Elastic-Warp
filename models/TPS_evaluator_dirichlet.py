import torch
import numpy as np

from kornia.geometry import transform
from torch.nn.modules.module import Module

class TpsGridGen(Module):
    def __init__(self, out_h=128, out_w=128, grid_size=3):
        super(TpsGridGen, self).__init__()
        self.out_h, self.out_w = out_h, out_w

        self.grid_X, self.grid_Y = np.meshgrid(np.linspace(0,1,out_w),np.linspace(0,1,out_h))
        self.grid_X = torch.Tensor(self.grid_X).float().unsqueeze(0).unsqueeze(3)
        self.grid_Y = torch.Tensor(self.grid_Y).float().unsqueeze(0).unsqueeze(3)

        axis_coords = np.linspace(0, 1, grid_size)
        self.N = (grid_size-2)**2

    def forward(self, delta, theta, feat_size):
        warped_grid = self.apply_transformation(delta,torch.cat((self.grid_X,self.grid_Y),3), theta, feat_size=feat_size)

        return warped_grid

    def apply_transformation(self, delta, points, theta, feat_size=64):
        b = theta.shape[0]
        _,h,w = [*points.shape[:3]]

        ctrlpts_tgt = theta.contiguous()
        delta = delta.reshape(b,-1,1,1)
        delta_x=delta[:,:self.N,:,:].squeeze(3)
        delta_y=delta[:,self.N:,:,:].squeeze(3)
        delta = torch.cat([delta_x, delta_y], dim=-1).reshape(b,10,10,2)

        ctrlpts_ref = ctrlpts_tgt.reshape(b,12,12,2).detach().clone()
        ctrlpts_ref[:, 1:-1, 1:-1] = ctrlpts_ref[:, 1:-1, 1:-1] + delta
        ctrlpts_ref = ctrlpts_ref.reshape(b,-1,2)

        points = points.to(delta.device).expand(b,h,w,2).reshape(b,-1,2)
        weights, affines = transform.get_tps_transform(ctrlpts_tgt, ctrlpts_ref)

        mesh = transform.warp_points_tps(points, ctrlpts_ref, weights, affines)
#        mesh = mesh.reshape(b,h,w,2)
        mesh = mesh.reshape(b,h,w,2) * (feat_size-1)

        return mesh
