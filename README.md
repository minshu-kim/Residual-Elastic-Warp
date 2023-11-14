# Recurrent Elastic Warps
This repository contains the official implementation of [REwarp](https://arxiv.org/abs/2309.01406), 24' WACV.

## Requirement
1) Python packages
```
conda env create --file environment.yaml
conda activate rewarp
```

## Train & Evaluation
```
bash scripts/train.sh 0
bash scripts/eval.sh 0
```

## Pretrained Models
You can download below models on this [link](https://drive.google.com/file/d/1T4G2qDTwvSWCPyxx7Q-tM0F64qwvvGke/view?usp=share_link).
1. hcell.pth: Pretrained HCell,
2. rewarp.pth: Pretrained HCell and TCell.

## Acknowlegment
This work is mainly based on [LTEW](https://github.com/jaewon-lee-b/ltew) and [IHN](https://github.com/imdumpl78/IHN), we thank the authors for their contributions.
