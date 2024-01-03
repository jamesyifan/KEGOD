# python main.py -exp_type ad -DS REDDIT-BINARY -batch_size 16 -batch_size_test 16 -num_epoch 80 -num_cluster 30 -alpha 0.8
CUDA_VISIBLE_DEVICES=3 python main.py -DS REDDIT-BINARY --exp_type ad --num_epoch 3000 --type_learner gnn --is_adaptive 1 --alpha 0.8
