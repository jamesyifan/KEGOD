# python main.py -exp_type oodd -DS_pair PTC_MR+MUTAG -num_epoch 400 -num_cluster 2 -alpha 0.8
# CUDA_VISIBLE_DEVICES=1 python main.py -DS_pair PTC_MR+MUTAG --exp_type oodd --num_epoch 5000 --type_learner gnn --is_adaptive 1 --alpha 0.8


CUDA_VISIBLE_DEVICES=5 python main.py -DS_pair PTC_MR+MUTAG --exp_type oodd --num_epoch 5000 --type_learner gnn --is_adaptive 1 --alpha 0.8
