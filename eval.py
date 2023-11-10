import os
import yaml
import utils
import torch
import models
import argparse
import datasets

import torchgeometry as tgm
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms import GaussianBlur

from skimage.metrics import peak_signal_noise_ratio as compare_psnr

torch.manual_seed(2022)

def make_data_loader(spec, tag=''):
    if spec is None: return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'], shuffle=False, num_workers=8, pin_memory=True)

    return loader


def linear_blender(ref, tgt, ref_m, tgt_m, mask=False):
    blur = GaussianBlur(kernel_size=(21,21), sigma=20)
    r1, c1 = torch.nonzero(ref_m[0, 0], as_tuple=True)
    r2, c2 = torch.nonzero(tgt_m[0, 0], as_tuple=True)

    center1 = (r1.float().mean(), c1.float().mean())
    center2 = (r2.float().mean(), c2.float().mean())

    vec = (center2[0] - center1[0], center2[1] - center1[1])

    ovl = (ref_m * tgt_m).round()[:, 0].unsqueeze(1)
    ref_m_ = ref_m[:, 0].unsqueeze(1) - ovl
    r, c = torch.nonzero(ovl[0, 0], as_tuple=True)

    ovl_mask = torch.zeros_like(ref_m_).cuda()
    proj_val = (r - center1[0]) * vec[0] + (c - center1[1]) * vec[1]
    ovl_mask[ovl.bool()] = (proj_val - proj_val.min()) / (proj_val.max() - proj_val.min() + 1e-3)

    mask1 = (blur(ref_m_ + (1-ovl_mask)*ref_m[:,0].unsqueeze(1)) * ref_m + ref_m_).clamp(0,1)
    if mask: return mask1

    mask2 = (1-mask1) * tgt_m
    stit = ref * mask1 + tgt * mask2

    return stit


def eval(loader, IHN, model, args):
    model.eval()

    failures = 0
    tot_mpsnr = 0

    collection_30 = []
    collection_60 = []
    collection_100 = []

    hcell_iter = 6
    tcell_iter = 3
    model.iters_lev = tcell_iter
    blur = GaussianBlur(kernel_size=(151,151), sigma=200)

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

        _, disps, hinp = IHN(ref, tgt, iters_lev0=hcell_iter)


        # Preparation of warped inputs
        H, img_h, img_w, offset = utils.get_warped_coords(disps[-1], scale=(h/512, w/512), size=(h,w))
        H_, *_ = utils.get_H(disps[-1].reshape(ref.shape[0],2,-1).permute(0,2,1), [*ref.shape[-2:]])
        H_ = utils.compens_H(H_, [*ref.shape[-2:]])

        grid = utils.make_coordinate_grid([*ref.shape[-2:]], type=H_.type())
        grid = grid.reshape(1, -1, 2).repeat(ref.shape[0], 1, 1)

        mesh_homography = utils.warp_coord(grid, H_.cuda()).reshape(b,*ref.shape[-2:],-1)
        ones = torch.ones_like(ref_).cuda()
        tgt_w = F.grid_sample(tgt, mesh_homography, align_corners=True)


        # Warp Field estimated by TPS
        flows = model(tgt_w, ref, iters=tcell_iter, scale=scale)
        translation = utils.get_translation(*offset)
        T_ref = translation.clone()
        T_tgt = torch.inverse(H).double() @ translation.cuda()

        sizes = (img_h, img_w)
        if img_h > 5000 or img_w > 5000:
            failures += 1
            print('Fail; Evaluated Size: {}X{}'.format(img_h, img_w))
            flows, disps= None, None
            continue


        # Image Alignment
        coord1 = utils.to_pixel_samples(None, sizes=sizes).cuda()
        mesh_r, _ = utils.gridy2gridx_homography(
            coord1.contiguous(), *sizes, *tgt_.shape[-2:], T_ref.cuda(), cpu=False
        )
        mesh_r = mesh_r.reshape(b, img_h, img_w, 2).cuda().flip(-1)

        coord2 = utils.to_pixel_samples(None, sizes=sizes).cuda()
        mesh_t, _ = utils.gridy2gridx_homography(
            coord2.contiguous(), *sizes, *tgt_.shape[-2:], T_tgt.cuda(), cpu=False
        )
        mesh_t = mesh_t.reshape(b, img_h, img_w, 2).cuda().flip(-1)

        mask_r = F.grid_sample(ones, mesh_r, mode='nearest', align_corners=True)
        mask_t = F.grid_sample(ones, mesh_t, mode='nearest', align_corners=True)

        ovl = torch.round(mask_r * mask_t)
        if ovl.sum() == 0:
            failures += 1
            continue

        flow = flows[-1]/511
        if flow.shape[-2] != 512 or flow.shape[-1] != 512:
            flow = F.interpolate(flow.permute(0,3,1,2), size=(h, w), mode='bilinear').permute(0,2,3,1) * 2

        mesh_t[:, offset[0]:offset[0]+h, offset[1]:offset[1]+w, :] += flow
        ref_w = F.grid_sample(ref_, mesh_r, mode='bilinear', align_corners=True)
        mask_r = F.grid_sample(ones, mesh_r, mode='nearest', align_corners=True)
        tgt_w = F.grid_sample(tgt_, mesh_t, mode='bilinear', align_corners=True)
        mask_t = F.grid_sample(ones, mesh_t, mode='nearest', align_corners=True)

        ref_w = (ref_w + 1)/2 * mask_r
        tgt_w = (tgt_w + 1)/2 * mask_t


        # Image Stitching
        stit = linear_blender(ref_w, tgt_w, mask_r, mask_t)


        # Evaluation
        ovl = (mask_r * mask_t).round().bool()
        pixels = img_h * img_w
        ovls = ovl[:, 0].sum()
        ovl_ratio = ovls / pixels

        ref_ovl = ref_w[ovl].cpu().numpy()
        tgt_ovl = tgt_w[ovl].cpu().numpy()

        psnr = compare_psnr(ref_ovl, tgt_ovl, data_range=1.)
        stit += (1-(mask_r + mask_t).clamp(0,1))

        if ovl_ratio <= 0.3: collection_30.append(psnr)
        elif ovl_ratio > 0.3 and ovl_ratio <= 0.6 : collection_60.append(psnr)
        elif ovl_ratio > 0.6: collection_100.append(psnr)
        collections = collection_30 + collection_60 + collection_100

        pbar.set_description_str(
            desc="[Evaluation] PSNR:{:.4f}/{:.4f}/{:.4f}, Avg PSNR: {:.4f}, Failures:{}".format(
                (sum(collection_30) + 1e-10)/(len(collection_30) + 1e-7),
                (sum(collection_60) + 1e-10)/(len(collection_60) + 1e-7),
                (sum(collection_100) + 1e-10)/(len(collection_100) + 1e-7),
                (sum(collections) + 1e-10)/(len(collections) + 1e-7),
                failures), refresh=True
        )

        flows, disps= None, None

def main(config, args):
    sv_file = torch.load(config['resume'])

    model = models.make(sv_file['T_model'], load_sd=True).cuda()
    H_model = models.make(sv_file['H_model'], load_sd=True).cuda()

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
