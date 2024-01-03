# python main.py -exp_type ad -DS BZR -num_epoch 400 -num_cluster 2 -alpha 0.8
CUDA_VISIBLE_DEVICES=5 python main.py -DS PROTEINS_full --exp_type ad --num_epoch 2000 --type_learner gnn --is_adaptive 1 --alpha 0.2
