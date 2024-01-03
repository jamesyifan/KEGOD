import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='SIGNET')
    parser.add_argument('-DS', type=str, default='BZR')
    parser.add_argument('-DS_ood', help='Dataset', default='COX2')
    parser.add_argument('-DS_pair', default=None)

    parser.add_argument('--exp_type', type=str, default='ad', choices=['oodd', 'ad'])

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--batch_size_test', type=int, default=9999)
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--num_trials', type=int, default=5)
    parser.add_argument('--device', type=int, default=0)

    parser.add_argument('--lr', dest='lr', type=float, default=0.001)
    parser.add_argument('--num_epoch', type=int, default=500)
    parser.add_argument('--encoder_layers', type=int, default=3)
    parser.add_argument('--hidden_dim', type=int, default=16)
    parser.add_argument('--pooling', type=str, default='add', choices=['add', 'max'])
    parser.add_argument('--readout', type=str, default='last', choices=['concat', 'add', 'last'])
    parser.add_argument('--is_adaptive', type=int, default=1)

    parser.add_argument('--alpha', type=float, default=0)

    parser.add_argument('--type_learner', type=str, default='none', choices=["none", "att", "mlp", "gnn"])
    parser.add_argument('--activation_learner', type=str, default='relu', choices=["relu", "tanh"])
    parser.add_argument('--hidden_graphs', type=int, default=16)
    parser.add_argument('--size_hidden_graphs', type=int, default=10)
    parser.add_argument('--max_step', type=int, default=3)

    parser.add_argument('--compress_memory_size', type=int, default=12800)
    parser.add_argument('--compress_t', type=float, default=0.01)

    parser.add_argument('--k', type=int, default=30)
    parser.add_argument('--sim_function', type=str, default='cosine', choices=['cosine', 'minkowski'])

    parser.add_argument('--dropout', type=float, default=0.15)
    parser.add_argument('--normalize', action='store_true', default=False)

    parser.add_argument('--maskfeat_rate_learner', type=float, default=0.4)
    parser.add_argument('--reg_lambda', type=float, default=0.0001)
    # parser.add_argument('--reg_lambda', type=float, default=0.0)

    return parser.parse_args()