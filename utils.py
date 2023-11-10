import os
import cv2
import time
import math
import shutil

import torch
import random
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

import torchgeometry as tgm

from tensorboardX import SummaryWriter
from torch.optim import SGD, Adam, AdamW

from kornia.geometry.transform import get_perspective_transform as get_H_mat


def warp_coord(coord, H):
    ones = torch.ones_like(coord[..., 0].unsqueeze(-1)).cuda()
    coord_hom = torch.cat([coord, ones], dim=-1)

    coord_hom_w = (H @ coord_hom.permute(0,2,1)).permute(0,2,1)
    coord_hom_w = coord_hom_w[..., :2] / coord_hom_w[..., 2].unsqueeze(-1)

    return coord_hom_w


def warp(x, flo, grid=None):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid
    if grid is None:
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()

        if x.is_cuda:
            grid = grid.to(x.device)
    vgrid = torch.autograd.Variable(grid) + flo

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = nn.functional.grid_sample(x, vgrid, mode='bilinear', align_corners=True)

    if grid is None:
        mask = torch.autograd.Variable(torch.ones(x.size())).to(x.device)
        mask = nn.functional.grid_sample(mask, vgrid, align_corners=True, mode='nearest')

        mask[mask < 0.999] = 0
        mask[mask > 0] = 1

        return output * mask, vgrid #, mask

    return output, vgrid


def get_warped_coords(four_point, scale=(4,4), size=(512,512)):
    h, w = size
    scale_h, scale_w = scale
    scale = torch.Tensor([[[[scale_w]], [[scale_h]]]]).cuda()

    b, _, _, _ = four_point.shape
    four_point = four_point * scale
    four_point_org = torch.zeros((2, 2, 2)).to(four_point.device)

    four_point_org[:, 0, 0] = torch.Tensor([0, 0])
    four_point_org[:, 0, 1] = torch.Tensor([w-1, 0])
    four_point_org[:, 1, 0] = torch.Tensor([0, h-1])
    four_point_org[:, 1, 1] = torch.Tensor([w-1, h-1])

    four_point_org = four_point_org.unsqueeze(0)
    four_point_org = four_point_org.repeat(b, 1, 1, 1)
    four_point_new = four_point_org + four_point
    four_point_org = four_point_org.flatten(2).permute(0, 2, 1)
    four_point_new = four_point_new.flatten(2).permute(0, 2, 1)

    H = tgm.get_perspective_transform(four_point_org, four_point_new).cuda()
    H_inv = torch.inverse(H)
    new = warp_coord(four_point_org, H_inv.cuda().float())

    corners_x = torch.stack([four_point_org[..., 0], new[..., 0]], dim=-1)
    corners_y = torch.stack([four_point_org[..., 1], new[..., 1]], dim=-1)

    w_max = torch.ceil(torch.max(corners_x)); h_max = torch.ceil(torch.max(corners_y))
    w_min = torch.floor(torch.min(corners_x)); h_min = torch.floor(torch.min(corners_y))
    img_h = (h_max - h_min).int().item() + 1; img_w = (w_max - w_min).int().item() + 1
    offset = [-h_min.int(), -w_min.int()]

    return H_inv[0], img_h, img_w, offset


def gridy2gridx_homography(gridy, H, W, h, w, m, cpu=True):
    # scaling
    gridy += 1
    gridy[:, 0] *= (H-1) / 2
    gridy[:, 1] *= (W-1) / 2
    gridy = gridy.flip(-1)

    if cpu:
        gridy = torch.cat((gridy, torch.ones(gridy.shape[0], 1)), dim=-1).double()
    else:
        gridy = torch.cat((gridy, torch.ones(gridy.shape[0], 1).cuda()), dim=-1).double()

    gridx = torch.mm(m, gridy.permute(1,0)).permute(1,0)

    gridx[:, 0] /= gridx[:, -1]
    gridx[:, 1] /= gridx[:, -1]
    gridx = gridx[:, :2].flip(-1)

    gridx[:, 0] /= (h-1) / 2
    gridx[:, 1] /= (w-1) / 2
    gridx -= 1
    gridx = gridx.float()

    mask = torch.where(torch.abs(gridx) > 1., 0, 1)
    mask = mask[..., 0] * mask[..., 1]
    mask = mask.float()

    return gridx, mask


class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


def time_text(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    elif t >= 60:
        return '{:.1f}m'.format(t / 60)
    else:
        return '{:.1f}s'.format(t)


_log_path = None
def set_log_path(path):
    global _log_path
    _log_path = path


def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)


def ensure_path(path, remove=True):
    basename = os.path.basename(path.rstrip('/'))
    if os.path.exists(path):
        if remove and (basename.startswith('_')
                or input('{} exists, remove? (y/[n]): '.format(path)) == 'y'):
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


def set_save_path(save_path, remove=True):
    ensure_path(save_path, remove=remove)
    set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    return log, writer


def compute_num_params(model, text=False):
    tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


def make_optimizer(param_list, optimizer_spec, load_sd=False):
    Optimizer = {
        'sgd': SGD,
        'adam': Adam,
        'adamw': AdamW,
    }[optimizer_spec['name']]
    optimizer = Optimizer(param_list, **optimizer_spec['args'])
    if load_sd:
        optimizer.load_state_dict(optimizer_spec['sd'])
    return optimizer


def get_translation(h_min, w_min):
    trans = torch.Tensor([[1., 0., -w_min],
                          [0., 1., -h_min],
                          [0., 0., 1.]]).double()

    return trans


def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def make_coordinate_grid(spatial_size, type):
    h, w = spatial_size
    if type is not None:
        x = torch.arange(w).type(type)
        y = torch.arange(h).type(type)

    else:
        x = torch.arange(w).cuda()
        y = torch.arange(h).cuda()

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)

    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)

    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)

    return meshed


def to_pixel_samples(img, sizes=None):
    """ Convert the image to coord-RGB pairs.
        img: Tensor, (3, H, W)
    """
    if sizes is not None:
        coord = make_coord(sizes)
    else:
        coord = make_coord(img.shape[-2:])

    if img is not None:
        rgb = img.view(3, -1).permute(1, 0)
        return coord, rgb

    return coord


def quantize(x: torch.Tensor) -> torch.Tensor:
    x = 127.5 * (x + 1)
    x = x.clamp(min=0, max=255)
    x = x.round()
    x = x / 127.5 - 1
    return x


def compens_H(H, size):
    h, w = size

    M = torch.Tensor([[(h-1)/2., 0.,   (h-1)/2.],
                      [0.,   (w-1)/2., (w-1)/2.],
                      [0.,   0.,   1.]]).cuda()

    M_inv = torch.inverse(M)
    return M_inv @ H @ M


def get_H(shift, size):
    h, w  = size
    b, _, _ = shift.shape

    src_corner = torch.Tensor([[[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]]])
    src_corner = src_corner.repeat(b, 1, 1)

    tgt_corner = src_corner + shift.cpu()
    H_src2tgt = get_H_mat(src_corner, tgt_corner).cuda()

    return H_src2tgt


def STN(image2_tensor, H_tf=None, grid=None, offsets=()):
    """Spatial Transformer Layer"""

    def _repeat(x, n_repeats):
        rep = torch.ones(1, n_repeats, dtype=x.dtype)
        x = torch.mm(x.reshape(-1, 1), rep)
        return x.reshape(-1)

    def _interpolate(im, x, y, out_size):
        # constants
        num_batch, channels, height, width = im.shape
        device = im.device

        x, y = x.float().to(device), y.float().to(device)
        height_f, width_f = torch.tensor(height).float(), torch.tensor(width).float()
        out_height, out_width = out_size

        # scale indices from [-1, 1] to [0, width/height]
        # effect values will exceed [-1, 1], so clamp is unnecessary or even incorrect
        x = (x + 1.0) * width_f / 2.0
        y = (y + 1.0) * height_f / 2.0

        # do sampling
        x0 = x.floor().int()
        x1 = x0 + 1
        y0 = y.floor().int()
        y1 = y0 + 1

        x0 = torch.clamp(x0, 0, width - 1)
        x1 = torch.clamp(x1, 0, width - 1)
        y0 = torch.clamp(y0, 0, height - 1)
        y1 = torch.clamp(y1, 0, height - 1)

        dim2 = width
        dim1 = width * height
        base = _repeat(torch.arange(num_batch) * dim1, out_height * out_width).to(device)
        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # use indices to lookup pixels in the flat image and restore
        # channels dim
        im_flat = im.permute(0, 2, 3, 1).reshape(-1, channels).float()
        Ia, Ib, Ic, Id = im_flat[idx_a], im_flat[idx_b], im_flat[idx_c], im_flat[idx_d]

        # and finally calculate interpolated values
        x0_f = x0.float()
        x1_f = x1.float()
        y0_f = y0.float()
        y1_f = y1.float()
        wa = torch.unsqueeze(((x1_f - x) * (y1_f - y)), 1)
        wb = torch.unsqueeze(((x1_f - x) * (y - y0_f)), 1)
        wc = torch.unsqueeze(((x - x0_f) * (y1_f - y)), 1)
        wd = torch.unsqueeze(((x - x0_f) * (y - y0_f)), 1)

        output = wa * Ia + wb * Ib + wc * Ic + wd * Id

        return output

    def _meshgrid(height, width):
        x_t = torch.mm(torch.ones(height, 1), torch.linspace(-1.0, 1.0, width).unsqueeze(0))
        y_t = torch.mm(torch.linspace(-1.0, 1.0, height).unsqueeze(1), torch.ones(1, width))

        x_t_flat = x_t.reshape(1, -1)
        y_t_flat = y_t.reshape(1, -1)

        ones = torch.ones_like(x_t_flat)
        grid = torch.cat([x_t_flat, y_t_flat, ones], dim=0)
        return grid

    bs, nc, height, width = image2_tensor.shape
    device = image2_tensor.device

    if grid == None:
        is_nan = torch.isnan(H_tf.reshape(bs, 9)).any(dim=1)
        assert is_nan.sum() == 0, f'{image2_tensor.shape} {len(offsets)}, {[off.view(-1, 8)[is_nan] for off in offsets]}'
        H_tf = H_tf.reshape(-1, 3, 3).float()
        grid = _meshgrid(height, width).unsqueeze(0).expand(bs, -1, -1).to(device)

        T_g = torch.bmm(H_tf, grid)
        x_s, y_s, t_s = torch.chunk(T_g, 3, dim=1)

        t_s_flat = t_s.reshape(-1)
        eps, maximal = 1e-2, 10.
        t_s_flat[t_s_flat.abs() < eps] = eps

        x_s_flat = x_s.reshape(-1) / t_s_flat
        y_s_flat = y_s.reshape(-1) / t_s_flat

    else:
        x_s_flat = grid[..., 0].reshape(-1)
        y_s_flat = grid[..., 1].reshape(-1)

    input_transformed = _interpolate(image2_tensor, x_s_flat, y_s_flat, (height, width))
    output = input_transformed.reshape(bs, height, width, nc).permute(0, 3, 1, 2)
    mesh = torch.stack([x_s_flat, y_s_flat], dim=-1).reshape(bs, height, width, 2)

    return output, mesh
