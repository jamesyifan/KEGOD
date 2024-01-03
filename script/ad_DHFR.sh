# python main.py -exp_type ad -DS DHFR -num_epoch 20 -num_cluster 2 -alpha 0
CUDA_VISIBLE_DEVICES=1 python main.py -DS DHFR --exp_type ad --num_epoch 2000 --type_learner gnn --is_adaptive 1 --alpha 0
