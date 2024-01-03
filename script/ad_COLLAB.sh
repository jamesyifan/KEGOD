# python main.py -exp_type ad -DS COLLAB -batch_size 64 -batch_size_test 64 -num_epoch 100 -num_cluster 2 -alpha 0.8
CUDA_VISIBLE_DEVICES=2 python main.py -DS COLLAB --exp_type ad --num_epoch 3000 --type_learner gnn --is_adaptive 1 --alpha 0.8
