CUDA_VISIBLE_DEVICES=$1 python eval.py --config=configs/test/rewarp.yaml
CUDA_VISIBLE_DEVICES=$1 python eval_psnr.py --config=configs/test/rewarp.yaml
