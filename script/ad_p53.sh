# python main.py -exp_type ad -DS Tox21_p53 -num_epoch 150 -num_cluster 5 -alpha 0.2
CUDA_VISIBLE_DEVICES=1 python main.py -DS Tox21_p53 --exp_type ad --num_epoch 3000 --type_learner gnn --is_adaptive 0 --alpha 0.2
