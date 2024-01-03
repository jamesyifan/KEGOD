# python main.py -exp_type ad -DS Tox21_MMP -num_epoch 400 -num_cluster 5 -alpha 0.0
CUDA_VISIBLE_DEVICES=7 python main.py -DS Tox21_MMP --exp_type ad --num_epoch 3000 --type_learner gnn --is_adaptive 1 --alpha 0.0
