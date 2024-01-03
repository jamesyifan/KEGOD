# python main.py -exp_type ad -DS ENZYMES -num_epoch 400 -num_cluster 10 -alpha 0.2
CUDA_VISIBLE_DEVICES=2 python main.py -DS ENZYMES --exp_type ad --num_epoch 2000 --type_learner gnn --is_adaptive 1 --alpha 0.2
