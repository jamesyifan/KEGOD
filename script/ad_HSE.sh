# python main.py -exp_type ad -DS Tox21_HSE -num_epoch 300 -num_cluster 2 -alpha 0.2
CUDA_VISIBLE_DEVICES=7 python main.py -DS Tox21_HSE --exp_type ad --num_epoch 5000 --type_learner gnn --is_adaptive 0 --alpha 0.2
