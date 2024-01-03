# python main.py -exp_type oodd -DS_pair AIDS+DHFR -batch_size_test 128 -num_epoch 400 -num_cluster 10 -alpha 0.2
# CUDA_VISIBLE_DEVICES=4 python main.py -DS_pair AIDS+DHFR --exp_type oodd --num_epoch 5000 --type_learner gnn --is_adaptive 1 --alpha 0.2


CUDA_VISIBLE_DEVICES=4 python main.py -DS_pair AIDS+DHFR --exp_type oodd --num_epoch 800 --type_learner none --is_adaptive 1 --alpha 0.2
