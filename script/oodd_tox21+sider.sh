# python main.py -exp_type oodd -DS_pair ogbg-moltox21+ogbg-molsider -batch_size_test 128 -num_epoch 400 -num_cluster 5 -alpha 0.2
# CUDA_VISIBLE_DEVICES=5 python main.py -DS_pair ogbg-moltox21+ogbg-molsider --exp_type oodd --num_epoch 5000 --type_learner gnn --is_adaptive 1 --alpha 0.2


CUDA_VISIBLE_DEVICES=2 python main.py -DS_pair ogbg-moltox21+ogbg-molsider --exp_type oodd --num_epoch 2000 --type_learner gnn --is_adaptive 1 --alpha 1.0
