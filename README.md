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

## Acknowlegment
This work is mainly based on [LTEW](https://github.com/jaewon-lee-b/ltew) and [IHN](https://github.com/imdumpl78/IHN), we thank the authors for their contributions.
