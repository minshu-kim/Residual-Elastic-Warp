import numpy as np
from datasets import register
from torch.utils.data import Dataset


@register('paired-images')
class SRImplicitHomographyPaired(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_ref, img_tgt = self.dataset[idx]
        img_ref = (img_ref - 0.5) * 2
        img_tgt = (img_tgt - 0.5) * 2

        img_ref = np.transpose(img_ref, (2,0,1))
        img_tgt = np.transpose(img_tgt, (2,0,1))

        return {
            'inp_ref'  : img_ref,
            'inp_tgt'  : img_tgt,
        }
