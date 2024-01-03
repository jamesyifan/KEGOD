# python main.py -exp_type ad -DS DD -batch_size 16 -batch_size_test 16 -num_epoch 50 -num_cluster 2 -alpha 1.0
CUDA_VISIBLE_DEVICES=5 python main.py -DS DD --exp_type ad --num_epoch 5000 --type_learner gnn --is_adaptive 1 --alpha 0.0
