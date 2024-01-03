# python main.py -exp_type oodd -DS_pair BZR+COX2 -num_epoch 400 -num_cluster 2 -alpha 0
# CUDA_VISIBLE_DEVICES=2 python main.py -DS_pair BZR+COX2  --exp_type oodd --num_epoch 5000 --type_learner gnn --is_adaptive 1 --alpha 0

# ablation
CUDA_VISIBLE_DEVICES=2 python main.py -DS_pair BZR+COX2 --exp_type oodd --num_epoch 800 --type_learner gnn --is_adaptive 1 --alpha 0.2
