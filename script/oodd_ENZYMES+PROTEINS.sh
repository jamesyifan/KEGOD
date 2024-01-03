# python main.py -exp_type oodd -DS_pair ENZYMES+PROTEINS -num_epoch 150 -num_cluster 15 -alpha 0.2
# CUDA_VISIBLE_DEVICES=4 python main.py -DS_pair ENZYMES+PROTEINS --exp_type oodd --num_epoch 5000 --type_learner gnn --is_adaptive 1 --alpha 0.2

CUDA_VISIBLE_DEVICES=3 python main.py -DS_pair ENZYMES+PROTEINS --exp_type oodd --num_epoch 5000 --type_learner gnn --is_adaptive 1 --alpha 0.8