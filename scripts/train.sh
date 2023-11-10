CUDA_VISIBLE_DEVICES=$1 python train_hcell.py --config=configs/train/HCell.yaml
CUDA_VISIBLE_DEVICES=$1 python train_tcell.py --config=configs/train/TCell.yaml
