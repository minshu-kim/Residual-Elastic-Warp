import os
import gc
import yaml
import utils
import torch
import models
import argparse
import datasets
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

torch.manual_seed(2022)

def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})

    log('{} dataset: size={}'.format(tag, len(dataset)))
    for k, v in dataset[0].items():
        log('  {}: shape={}'.format(k, tuple(v.shape)))

    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        shuffle=False, num_workers=8, pin_memory=True)

    return loader


def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    valid_loader = make_data_loader(config.get('valid_dataset'), tag='valid')

    return train_loader, valid_loader


def prepare_train(resume=False):
    if resume:
        sv_file = torch.load(config['resume'])
        HCell = models.make(sv_file['H_model'], load_sd=True).cuda()
        TCell = models.make(sv_file['T_model'], load_sd=True).cuda()

        optimizer = utils.make_optimizer(
            TCell.parameters(),
            sv_file['optimizer'], load_sd=True
        )
        epoch_start = sv_file['epoch'] + 1
        lr_scheduler = ExponentialLR(optimizer, gamma=0.98)

    else:
        sv_file = torch.load(config['load_from'])
        HCell = models.make(sv_file['model'], load_sd=True).cuda()
        TCell = models.make(config['T_model']).cuda()

        optimizer = utils.make_optimizer(
            TCell.parameters(),
            config['optimizer']
        )

        epoch_start = 1
        lr_scheduler = ExponentialLR(optimizer, gamma=0.98)

    num_params = utils.compute_num_params(TCell, text=False) + utils.compute_num_params(HCell, text=False)
    log('Model Params: {}'.format(str(num_params)))

    return HCell, TCell, optimizer, epoch_start, lr_scheduler


def train(loader, HCell, TCell, optimizer, epoch, finetune=False):
    TCell.train()

    tot_loss = 0
    tot_mpsnr = 0
    l1_loss = nn.L1Loss(reduction='mean')

    grad_clip = torch.nn.utils.clip_grad_norm_
    pbar = tqdm(range(len(loader)), smoothing=0.9)
    loader = iter(loader)

    grid = utils.make_coordinate_grid([512,512], None)
    grid = grid.reshape(1, 512, 512, 2)

    for b_id in pbar:
        batch = next(loader)
        for k, v in batch.items():
            batch[k] = v.cuda()

        ref = batch['inp_ref']
        tgt = batch['inp_tgt']

        b, c, h, w = ref.shape

        _, disps, hinp = HCell(tgt, ref, iters_lev0=6)
        H = utils.get_H(disps[-1].reshape(ref.shape[0],2,-1).permute(0,2,1), [*ref.shape[-2:]])
        H = utils.compens_H(H, [*ref.shape[-2:]])
        grid_ = grid.reshape(1, -1, 2).repeat(ref.shape[0], 1, 1)

        mesh_homography = utils.warp_coord(grid_, H).reshape(b,h,w,-1)
        ref_w = F.grid_sample(ref, mesh_homography, align_corners=True)
        flows = TCell(ref_w, tgt, iters=6)

        ones = torch.ones_like(ref).cuda()
        flow_homography = (mesh_homography - grid.repeat(b,1,1,1)) * 512 / 2

        loss = 0
        for i in range(len(flows)):
            flow = flow_homography + flows[i]

            grid_ = (grid.repeat(b,1,1,1).permute(0,3,1,2) + 1) / 2
            mask, _ = utils.warp(ones, flow.permute(0,3,1,2), grid = grid_ * (h-1))
            ref_ovl, _ = utils.warp(ref, flow.permute(0,3,1,2), grid = grid_ * (h-1))
            tgt_ovl = tgt * mask

            ref_ovl_samples = ref_ovl[mask.bool()]
            tgt_ovl_samples = tgt_ovl[mask.bool()]

            loss = loss + l1_loss(ref_ovl_samples, tgt_ovl_samples) * (0.85 ** (len(flows)-1-i))

        if torch.isnan(loss):
            flow_loss, flows = None, None

            gc.collect()
            torch.cuda.empty_cache()
            continue

        tot_loss += loss.item()

        optimizer.zero_grad()
        loss.cpu().backward()

        grad_clip(TCell.parameters(), 0.1, norm_type=2)
        optimizer.step()

        pbar.set_description_str(
            desc="[Train] Epoch:{}/{}, Loss: {:.3f}".format(
                epoch, config['epoch_max'], tot_loss / (b_id+1)
            ), refresh=True
        )

        loss, flows, disps = None, None, None

    return tot_loss / (b_id+1)


def valid(loader, HCell, TCell):
    TCell.eval()
    tot_psnr = 0

    pbar = tqdm(range(len(loader)), smoothing=0.9)
    loader = iter(loader)

    for b_id in pbar:
        batch = next(loader)
        for k, v in batch.items():
            batch[k] = v.cuda()

        ref = batch['inp_ref']
        tgt = batch['inp_tgt']

        b, c, h, w = ref.shape

        _, disps, hinp = HCell(tgt, ref, iters_lev0=6)
        H = utils.get_H(disps[-1].reshape(ref.shape[0],2,-1).permute(0,2,1), [*ref.shape[-2:]])
        H = utils.compens_H(H, [*ref.shape[-2:]])

        grid = utils.make_coordinate_grid([*ref.shape[-2:]], type=H.type())
        grid = grid.reshape(1, -1, 2).repeat(ref.shape[0], 1, 1)
        mesh_homography = utils.warp_coord(grid, H).reshape(b,h,w,-1)
        ref_w = F.grid_sample(ref, mesh_homography, align_corners=True)

        flows = TCell(ref_w, tgt, iters=6)
        ones = torch.ones_like(ref).cuda()

        # Manipulate Flow Range
        flow_homography = (mesh_homography - grid.reshape(b,h,w,-1)) * 512 / 2
        flow = flow_homography + flows[-1]

        mask, _ = utils.warp(ones, flow.permute(0,3,1,2))
        ref_ovl, _ = utils.warp(ref, flow.permute(0,3,1,2))
        tgt_ovl = tgt * mask

        psnr = compare_psnr(ref_ovl.cpu().numpy(), tgt_ovl.cpu().numpy(), data_range=2.)
        tot_psnr += psnr

        pbar.set_description_str(desc="[Valid] PSNR: {:.4f}".format(tot_psnr/(b_id+1)), refresh=True)
        disps, flows = None, None

    return tot_loss/(b_id+1)


def main(config_, save_path, args):
    global config, log, writer
    config = config_
    log, writer = utils.set_save_path(save_path, remove=False)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    train_loader, valid_loader = make_data_loaders()
    HCell, TCell, optimizer, epoch_start, lr_scheduler = prepare_train(resume=args.resume)

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))

    best_loss = 1e4
    best_loss_e = 1

    timer = utils.Timer()

    epoch_max = config['epoch_max']
    epoch_save = config.get('epoch_save')

    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()

        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]
        print('[Scheduler] lr: {}'.format(optimizer.param_groups[0]['lr']))

        tr_loss = train(train_loader, HCell, TCell, optimizer, epoch)
        with torch.no_grad():
            val_loss = valid(valid_loader, HCell, TCell)

        lr_scheduler.step()

        model_spec = config['T_model']
        model_spec['sd'] = TCell.state_dict()

        model_spec_ = config['H_model']
        model_spec_['sd'] = HCell.state_dict()

        optimizer_spec = config['optimizer']
        optimizer_spec['sd'] = optimizer.state_dict()

        sv_file = {
            'T_model': model_spec,
            'H_model': model_spec_,
            'optimizer': optimizer_spec,
            'epoch': epoch
        }
        torch.save(sv_file, os.path.join(save_path, 'epoch-last.pth'))

        if (epoch_save is not None) and (epoch % epoch_save == 0):
            torch.save(sv_file, os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))

        if val_loss < best_loss:
            best_loss = val_loss
            best_e = epoch
            torch.save(sv_file, os.path.join(save_path, 'best.pth'))

        print('[{}/{} Summary] Avg Loss (Tr/Val): {:.4f}/{:.4f}'.format(epoch, epoch_max, tr_loss, val_loss))
        record = '[Record] Best Valid Loss: {:.4f} | {} Epoch'.format(best_loss, best_loss_e)

        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)

        record += '\t{} {}/{}'.format(t_epoch, t_elapsed, t_all)
        log(record)
        print('=' * os.get_terminal_size().columns)
        writer.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    save_path = os.path.join('./save', save_name)
    main(config, save_path, args)
