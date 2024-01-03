# python main.py -exp_type ad -DS NCI1 -batch_size 64 -batch_size_test 64 -num_epoch 400 -num_cluster 20 -alpha 1.0
CUDA_VISIBLE_DEVICES=6 python main.py -DS NCI1 --exp_type ad --num_epoch 5000 --type_learner gnn --is_adaptive 1 --alpha 0.2
