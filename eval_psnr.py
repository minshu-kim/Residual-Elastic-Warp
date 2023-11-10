import os
import gc
import yaml
import utils
import torch
import models
import argparse
import datasets
import torch.nn as nn
import torch.nn.functional as F

from torchmetrics import StructuralSimilarityIndexMeasure
import torchgeometry as tgm

from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

torch.manual_seed(2022)

def make_data_loader(spec, tag=''):
    if spec is None: return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'], shuffle=False, num_workers=8, pin_memory=True)

    return loader


def eval(loader, IHN, model, args):
    model.eval()

    failures = 0
    tot_mpsnr = 0

    hcell_iter = 6
    tcell_iter = 3
    model.iters_lev = tcell_iter

    pbar = tqdm(range(len(loader)), smoothing=0.9)
    loader = iter(loader)

    for b_id in pbar:
        batch = next(loader)

        for k, v in batch.items():
            batch[k] = v.cuda()

        ref_ = batch['inp_ref']
        tgt_ = batch['inp_tgt']
        b, c, h, w = ref_.shape

        if h != 512 or w != 512:
            ref = F.interpolate(ref_, size=(512,512), mode='bilinear')
            tgt = F.interpolate(tgt_, size=(512,512), mode='bilinear')
        else:
            ref = ref_
            tgt = tgt_
        scale = h/ref.shape[-2]

        _, disps, hinp = IHN(ref, tgt, iters_lev0=6)

        H_ = utils.get_H(disps[-1].reshape(ref.shape[0],2,-1).permute(0,2,1), [*ref.shape[-2:]])
        H_ = utils.compens_H(H_, [*ref.shape[-2:]])

        grid = utils.make_coordinate_grid([*ref.shape[-2:]], type=H_.type())
        grid = grid.reshape(1, -1, 2).repeat(ref.shape[0], 1, 1)
        mesh_homography = utils.warp_coord(grid, H_.cuda()).reshape(b,*ref.shape[-2:],-1)
        tgt_w = F.grid_sample(tgt, mesh_homography, align_corners=True)

        H_ = utils.get_H(disps[-1].reshape(b,2,-1).permute(0,2,1)*scale, [*ref_.shape[-2:]])
        H_ = utils.compens_H(H_, [*ref_.shape[-2:]])

        grid = utils.make_coordinate_grid([*ref_.shape[-2:]], type=H_.type())
        grid = grid.reshape(1, -1, 2).repeat(ref_.shape[0], 1, 1)
        mesh_homography = utils.warp_coord(grid, H_.cuda()).reshape(b,h,w,-1)
        flow_hom = (utils.warp_coord(grid, H_).reshape(b,h,w,-1) - grid.reshape(b,h,w,-1)) * (512*scale-1) / 2

        flows = model(tgt_w, ref, iters=6, scale=scale)
        flow = flow_hom + flows[-1]

        ones = torch.ones_like(ref_).cuda()
        mask, _ = utils.warp(ones, flow.permute(0,3,1,2))
        tgt_ovl, _ = utils.warp(tgt_, flow.permute(0,3,1,2))
        ref_ovl = ref_ * mask

        ref_ovl_samples = (ref_ovl)[0].permute(1,2,0).cpu().numpy()
        tgt_ovl_samples = (tgt_ovl)[0].permute(1,2,0).cpu().numpy()

        psnr = compare_psnr(ref_ovl_samples, tgt_ovl_samples, data_range=2.)
        tot_mpsnr = tot_mpsnr + psnr

        pbar.set_description_str(
            desc="[Evaluation] PSNR:{:.4f}, failures:{}".format(
                tot_mpsnr/(b_id+1-failures), failures), refresh=True
        )


def main(config, args):
    sv_file = torch.load(config['resume'])
    H_model = models.make(sv_file['H_model'], load_sd=True).cuda()
    model = models.make(sv_file['T_model'], load_sd=True).cuda()

    num_params = utils.compute_num_params(H_model, text=False)
    num_params += utils.compute_num_params(model, text=False)
    print('Model Params: {}'.format(str(num_params)))

    loader = make_data_loader(config.get('eval_dataset'), tag='eval')

    with torch.no_grad():
        eval(loader, H_model, model, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    main(config, args)
