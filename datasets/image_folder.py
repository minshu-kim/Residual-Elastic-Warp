import os
import json
import pickle
import imageio

import torch
import numpy as np

from PIL import Image
from glob import glob
from datasets import register

from torchvision import transforms
from torch.utils.data import Dataset

@register('image-folder')
class ImageFolder(Dataset):
    def __init__(
        self, root_path, inp_size=None, split_file=None,
        split_key=None, first_k=None, repeat=1, cache='none'
    ):
        self.repeat = repeat
        self.cache = cache
        self.inp_size = inp_size

        if split_file is None:
            filenames = sorted(os.listdir(root_path))
        else:
            with open(split_file, 'r') as f:
                filenames = json.load(f)[split_key]
        if first_k is not None:
            filenames = filenames[:first_k]

        self.files = []
        for filename in filenames:
            file = os.path.join(root_path, filename)

            if cache == 'none':
                self.files.append(file)

            elif cache == 'bin':
                bin_root = os.path.join(os.path.dirname(root_path),
                    '_bin_' + os.path.basename(root_path))
                if not os.path.exists(bin_root):
                    os.mkdir(bin_root)
                    print('mkdir', bin_root)
                bin_file = os.path.join(
                    bin_root, filename.split('.')[0] + '.pkl')
                if not os.path.exists(bin_file):
                    with open(bin_file, 'wb') as f:
                        pickle.dump(imageio.imread(file), f)
                    print('dump', bin_file)
                self.files.append(bin_file)

            elif cache == 'in_memory':
                img = transforms.ToTensor()(Image.open(file).convert('RGB'))
                img = transforms.Resize(self.inp_size, transforms.InterpolationMode.BILINEAR)(img)
                self.files.append(img.numpy())

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]

        if self.cache == 'none':
            img = transforms.ToTensor()(Image.open(x).convert('RGB'))
            if self.inp_size is not None:
                img = transforms.Resize(self.inp_size, transforms.InterpolationMode.BILINEAR)(img)
            return img.permute(1, 2, 0).numpy()

        elif self.cache == 'bin':
            with open(x, 'rb') as f:
                x = pickle.load(f)
            x = np.ascontiguousarray(x.transpose(2, 0, 1))
            x = torch.from_numpy(x).float() / 255
            return x

        elif self.cache == 'in_memory':
            return x


@register('paired-image-folders')
class PairedImageFolders(Dataset):
    def __init__(self, root_path_1, root_path_2, inp_size=None, **kwargs):
        self.dataset_1 = ImageFolder(root_path_1, inp_size, **kwargs)
        self.dataset_2 = ImageFolder(root_path_2, inp_size, **kwargs)

    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        return self.dataset_1[idx], self.dataset_2[idx]
