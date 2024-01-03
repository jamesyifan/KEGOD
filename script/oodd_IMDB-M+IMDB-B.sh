# python main.py -exp_type oodd -DS_pair IMDB-MULTI+IMDB-BINARY -num_epoch 20 -num_cluster 5 -alpha 0.8
# CUDA_VISIBLE_DEVICES=1 python main.py -DS_pair IMDB-MULTI+IMDB-BINARY --exp_type oodd --num_epoch 5000 --type_learner gnn --is_adaptive 1 --alpha 1.0

# ablation
CUDA_VISIBLE_DEVICES=1 python main.py -DS_pair IMDB-MULTI+IMDB-BINARY --exp_type oodd --num_epoch 800 --type_learner none --is_adaptive 1 --alpha 0.8
