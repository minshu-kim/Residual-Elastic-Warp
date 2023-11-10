import os
import gc
import yaml
import utils

import argparse
import datasets
import numpy as np

import models

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR

from tqdm import tqdm
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
        HCell = models.make(sv_file['model'], load_sd=True).cuda()
        optimizer = utils.make_optimizer(HCell.parameters(), sv_file['optimizer'], load_sd=True)

        epoch_start = sv_file['epoch'] + 1
        lr_scheduler = ExponentialLR(optimizer, gamma=0.99)

    else:
        HCell = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(HCell.parameters(), config['optimizer'])

        epoch_start = 1
        lr_scheduler = ExponentialLR(optimizer, gamma=0.99)

    num_params = utils.compute_num_params(HCell, text=False)
    log('Model Params: {}'.format(str(num_params)))

    return HCell, optimizer, epoch_start, lr_scheduler


def train(loader, HCell, optimizer, epoch, finetune=False):
    HCell.train()

    failures = 0
    tot_loss = 0

    l1_loss = nn.L1Loss(reduction='mean')
    grad_clip = torch.nn.utils.clip_grad_norm_

    pbar = tqdm(range(len(loader)), smoothing=0.9)
    loader = iter(loader)

    dropout = [i for i in range(6)]

    for b_id in pbar:
        batch = next(loader)
        for k, v in batch.items():
            batch[k] = v.cuda()

        ref = batch['inp_ref']
        tgt = batch['inp_tgt']
        b, c, h, w = ref.shape

        flows, disps, grid_loss = HCell(tgt, ref, iters_lev0=6)

        loss = 0
        ones = torch.ones_like(ref).cuda()
        dropout = sorted(np.random.choice(len(disps)-1, 3, replace=False))

        for i in range(len(disps)):
            if i in dropout:
                continue

            shift = disps[i].view(b, 2, -1).permute(0, 2, 1)
            H = utils.get_H(shift, (h, w))
            H = utils.compens_H(H, size=(h, w))

            tgt_ovl, _ = utils.STN(tgt, torch.inverse(H))
            mask, _ = utils.STN(ones, torch.inverse(H))

            ref_ovl = ref * mask.round()
            ref_ovl_samples = ref_ovl[mask.bool()]
            tgt_ovl_samples = tgt_ovl[mask.bool()]

            loss += l1_loss(ref_ovl, tgt_ovl) * (0.85 ** (len(disps)-i-1))

        if torch.isnan(loss) or mask.sum() == 0:
            failures += 1
            flows, disps, loss = None, None, None
            torch.cuda.empty_cache()
            continue

        optimizer.zero_grad()
        loss.backward()
        grad_clip(HCell.parameters(), 0.01, norm_type=2)
        optimizer.step()

        tot_loss += loss.item()

        pbar.set_description_str(
            desc="[Train] Epoch:{}/{}, Loss: {:.3f}, Failures: {}".format(
                epoch, config['epoch_max'], tot_loss/(b_id-failures+1), failures
            ), refresh=True
        )

        flows, disps, loss = None, None, None

    return tot_loss/(b_id-failures+1)


def valid(loader, HCell):
    HCell.eval()
    tot_mpsnr = 0

    pbar = tqdm(range(len(loader)), smoothing=0.9)
    loader = iter(loader)

    failures = 0

    for b_id in pbar:
        batch = next(loader)
        for k, v in batch.items():
            batch[k] = v.cuda()

        ref = batch['inp_ref']
        tgt = batch['inp_tgt']
        b, c, h, w = ref.shape

        flows, disps, _ = HCell(tgt, ref, iters_lev0=6)
        ones = torch.ones_like(ref).cuda()

        shift = disps[-1].reshape(b, 2, -1).permute(0, 2, 1)
        H = utils.get_H(shift, (h, w))
        H = utils.compens_H(H, size=(h, w))

        tgt_ovl, _ = utils.STN(tgt, torch.inverse(H))
        mask, _ = utils.STN(ones, torch.inverse(H))
        ref_ovl = ref * mask.round()

        ref_ovl_samples = ref_ovl[mask.bool()].cpu().numpy()
        tgt_ovl_samples = tgt_ovl[mask.bool()].cpu().numpy()

        mpsnr = compare_psnr(ref_ovl_samples, tgt_ovl_samples, data_range=2.)
        if mask.sum() == 0 or np.isnan(mpsnr):
            failures += 1

            gc.collect()
            torch.cuda.empty_cache()
            continue

        tot_mpsnr += mpsnr
        pbar.set_description_str(desc="[Valid] mPSNR: {:.4f}, Failures: {}".format(tot_mpsnr/(b_id-failures+1), failures), refresh=True)

        flows, disps = None, None

    return tot_mpsnr/(b_id-failures+1)


def main(config_, save_path, args):
    global config, log, writer
    config = config_
    log, writer = utils.set_save_path(save_path, remove=False)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    train_loader, valid_loader = make_data_loaders()
    HCell, optimizer, epoch_start, lr_scheduler = prepare_train(resume=args.resume)

    best_mpsnr = 1e-4
    best_mpsnr_e = 1
    timer = utils.Timer()

    epoch_max = config['epoch_max']
    epoch_save = config.get('epoch_save')

    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()

        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]
        print('[Scheduler] lr: {}'.format(optimizer.param_groups[0]['lr']))

        tr_loss = train(train_loader, HCell, optimizer, epoch)
        with torch.no_grad():
            val_mpsnr = valid(valid_loader, HCell)

        lr_scheduler.step()

        model_spec = config['model']
        model_spec['sd'] = HCell.state_dict()
        optimizer_spec = config['optimizer']
        optimizer_spec['sd'] = optimizer.state_dict()

        sv_file = {
            'model': model_spec,
            'optimizer': optimizer_spec,
            'epoch': epoch
        }
        torch.save(sv_file, os.path.join(save_path, 'epoch-last.pth'))

        if (epoch_save is not None) and (epoch % epoch_save == 0):
            torch.save(sv_file, os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))

        if val_mpsnr > best_mpsnr:
            best_mpsnr = val_mpsnr
            best_mpsnr_e = epoch
            torch.save(sv_file, os.path.join(save_path, 'best.pth'))

        print('Avg Loss: {:.4f}, mPSNR: {:.4f}'.format(tr_loss, val_mpsnr))
        record = '[Record] Best PSNR: {:.4f} | {} Epoch'.format(best_mpsnr, best_mpsnr_e)

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
