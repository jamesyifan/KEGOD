# python main.py -exp_type ad -DS Tox21_PPAR-gamma -num_epoch 200 -num_cluster 10 -alpha 0.8
CUDA_VISIBLE_DEVICES=3 python main.py -DS Tox21_PPAR-gamma --exp_type ad --num_epoch 3000 --type_learner gnn --is_adaptive 1 --alpha 0.8
