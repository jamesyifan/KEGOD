# python main.py -exp_type ad -DS COX2 -num_epoch 150 -num_cluster 3 -alpha 0.4
CUDA_VISIBLE_DEVICES=3 python main.py -DS COX2 --exp_type ad --num_epoch 2000 --type_learner gnn --is_adaptive 1 --alpha 0.4
