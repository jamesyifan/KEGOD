# python main.py -exp_type oodd -DS_pair ogbg-molclintox+ogbg-mollipo -batch_size_test 128 -num_epoch 300 -num_cluster 30 -alpha 1.0
# CUDA_VISIBLE_DEVICES=6 python main.py -DS_pair ogbg-molclintox+ogbg-mollipo --exp_type oodd --num_epoch 5000 --type_learner gnn --is_adaptive 1 --alpha 0.2


CUDA_VISIBLE_DEVICES=5 python main.py -DS_pair ogbg-molclintox+ogbg-mollipo --exp_type oodd --num_epoch 5000 --type_learner gnn --is_adaptive 1 --alpha 0.2 #1.0
