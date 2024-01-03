# python main.py -exp_type oodd -DS_pair ogbg-molfreesolv+ogbg-moltoxcast -batch_size_test 128 -num_epoch 400 -num_cluster 2 -alpha 0.6
# CUDA_VISIBLE_DEVICES=5 python main.py -DS_pair ogbg-molfreesolv+ogbg-moltoxcast --exp_type oodd --num_epoch 5000 --type_learner gnn --is_adaptive 1 --alpha 0.6


CUDA_VISIBLE_DEVICES=4 python main.py -DS_pair ogbg-molfreesolv+ogbg-moltoxcast --exp_type oodd --num_epoch 5000 --type_learner gnn --is_adaptive 1 --alpha 0.6
