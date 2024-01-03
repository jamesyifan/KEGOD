import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from models import KBGAD
from arguments import arg_parse
# from get_data_loaders import get_data_loaders
from get_data_loaders_tuad import get_ad_split_TU, get_ad_dataset_TU, get_ad_dataset_Tox21, get_ood_dataset
import random
import torch_geometric
from torch_geometric.utils import to_networkx
import pickle
from compress_loss import CompReSSMomentum
from copy import deepcopy
from torch_geometric.data import Batch, Data

import warnings
warnings.filterwarnings("ignore")
# explainable_datasets = ['mutag', 'mnist0', 'mnist1', 'bm_mn', 'bm_ms', 'bm_mt']

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch_geometric.seed_everything(seed)


def run(args, seed, split=None):
    set_seed(seed)

    if args.exp_type == 'oodd':
        loaders, meta = get_ood_dataset(args)
    elif args.exp_type == 'ad' and not args.DS.startswith('Tox21'):
        loaders, meta = get_ad_dataset_TU(args, split)
    elif args.exp_type == 'ad' and args.DS.startswith('Tox21'):
        loaders, meta = get_ad_dataset_Tox21(args)
        
    n_feat = meta['num_feat']
    n_train = meta['num_train']

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    compress = CompReSSMomentum(args)

    model = KBGAD(n_feat, args, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_loader = loaders['train']
    test_loader = loaders['test']
    best_auc = 0.0
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.99, patience=5, min_lr=0.000001)
    for epoch in range(1, args.num_epoch+1):
        if args.is_adaptive:
            if epoch == 1:
                weight_imp, weight_exp, weight_imp_exp, weight_exp_imp = 1, 1, 1, 1
                # weight_imp, weight_exp, weight_imp_exp = 1, 1, 1
            else:
                weight_imp, weight_exp, weight_imp_exp, weight_exp_imp = std_imp ** args.alpha, std_exp ** args.alpha, std_imp_exp ** args.alpha, std_exp_imp ** args.alpha
                weight_sum = (weight_imp + weight_exp + weight_imp_exp + weight_exp_imp) / 4
                weight_imp, weight_exp, weight_imp_exp, weight_exp_imp = weight_imp/weight_sum, weight_exp/weight_sum, weight_imp_exp/weight_sum, weight_exp_imp/weight_sum

                # weight_imp, weight_exp, weight_imp_exp = std_imp ** args.alpha, std_exp ** args.alpha, std_imp_exp ** args.alpha
                # weight_sum = (weight_imp + weight_exp + weight_imp_exp) / 3
                # weight_imp, weight_exp, weight_imp_exp = weight_imp/weight_sum, weight_exp/weight_sum, weight_imp_exp/weight_sum

        model.train()
        loss_all = 0
        if args.is_adaptive:
            loss_imp_all, loss_exp_all, loss_imp_exp_all, loss_exp_imp_all = [], [], [], []
            # loss_imp_all, loss_exp_all, loss_imp_exp_all = [], [], []

        for data in train_loader:
            optimizer.zero_grad()
            data = data.to(device)

            y_implicit1, y_explicit1, y_implicit2, y_explicit2, reg, batch_aug_edge_weight = model(data)

            # loss_implicit = model.loss_nce_imp(y_implicit1, y_implicit2)
            # loss_explicit = model.loss_nce_exp(y_explicit1, y_explicit2)

            loss_implicit = model.loss_nce(y_implicit1, y_implicit2)
            loss_explicit = model.loss_nce(y_explicit1, y_explicit2)

            # loss_imp_exp = model.loss_kl(y_implicit1, y_explicit2)
            # loss_exp_imp = model.loss_kl(y_explicit1, y_implicit2)

            # loss_imp_exp = compress(y_implicit1, y_explicit2)
            # loss_exp_imp = compress(y_explicit1, y_implicit2)
            loss_imp_exp = compress(y_implicit1, y_explicit1)
            loss_exp_imp = compress(y_explicit2, y_implicit2)

            # loss_imp_exp = compress(y_implicit1, y_explicit2) + compress(y_explicit1, y_implicit2)


            if args.is_adaptive:
                # weight_exp, weight_exp_imp, weight_imp_exp = 0, 0, 0
                # weight_imp, weight_exp_imp, weight_imp_exp = 0, 0, 0

                # weight_imp, weight_exp = 0, 0
                # weight_exp_imp, weight_imp_exp = 0, 0

                # weight_exp = 0
                # weight_imp = 0
                if reg is None: 
                    loss = weight_imp * loss_implicit.mean() + weight_exp * loss_explicit.mean() + (weight_imp_exp * loss_imp_exp.mean() + weight_exp_imp * loss_exp_imp.mean()) * 0.001
                else:
                    loss = weight_imp * loss_implicit.mean() + weight_exp * loss_explicit.mean() + (weight_imp_exp * loss_imp_exp.mean() + weight_exp_imp * loss_exp_imp.mean()) * 0.001 + args.reg_lambda * reg.mean()
                # loss = weight_imp * loss_implicit.mean() + weight_exp * loss_explicit.mean() + weight_imp_exp * loss_imp_exp.mean() * 0.001 + args.reg_lambda * reg.mean()
                # loss = weight_imp * loss_implicit.mean() + weight_exp * loss_explicit.mean() + (weight_imp_exp * loss_imp_exp.mean() + weight_exp_imp * loss_exp_imp.mean()) * 0.001 + args.reg_lambda * reg.mean()
                loss_imp_all += loss_implicit.detach().cpu().tolist()
                loss_exp_all += loss_explicit.detach().cpu().tolist()
                loss_imp_exp_all += loss_imp_exp.detach().cpu().tolist()
                loss_exp_imp_all += loss_exp_imp.detach().cpu().tolist()
            else:
                loss = loss_implicit.mean() + loss_explicit.mean() + (loss_imp_exp.mean() + loss_exp_imp.mean()) * 0.001 + args.reg_lambda * reg.mean()
                # loss = loss_implicit.mean() + loss_explicit.mean() + loss_imp_exp.mean() + reg.mean()

            loss_all += loss.item() * data.num_graphs
            loss.backward()
            optimizer.step()

        print('[TRAIN] Epoch:{:03d} | Loss:{:.4f}'.format(epoch, loss_all / n_train))

        if args.is_adaptive:
            mean_imp, std_imp = np.mean(loss_imp_all), np.std(loss_imp_all)
            mean_exp, std_exp = np.mean(loss_exp_all), np.std(loss_exp_all)
            mean_imp_exp, std_imp_exp = np.mean(loss_imp_exp_all), np.std(loss_imp_exp_all)
            mean_exp_imp, std_exp_imp = np.mean(loss_exp_imp_all), np.std(loss_exp_imp_all)

        if epoch % args.log_interval == 0:
            model.eval()
            # anomaly detection
            all_ad_true = []
            all_ad_score = []
            all_implicit = []
            all_explicit = []
            all_edge_index = []
            all_edge_weight = []
            ori_graph = []
            aug_graph = []
            for data in test_loader:
                all_ad_true.append(data.y.cpu())
                data = data.to(device)
                with torch.no_grad():
                    y_implicit1, y_explicit1, y_implicit2, y_explicit2, reg, batch_aug_edge_weight = model(data)
                    # ano_score_implicit = model.loss_nce_imp(y_implicit1, y_implicit2)
                    # ano_score_explicit = model.loss_nce_exp(y_explicit1, y_explicit2)

                    # ano_score_implicit = model.loss_nce(y_implicit1, y_implicit2)
                    # ano_score_explicit = model.loss_nce(y_explicit1, y_explicit2)
                    ano_score_implicit = model.loss_nce(y_implicit1, y_implicit1)
                    ano_score_explicit = model.loss_nce(y_explicit2, y_explicit2)

                    # ano_score_imp_exp = model.loss_kl(y_implicit1, y_explicit2)
                    # ano_score_exp_imp = model.loss_kl(y_explicit1, y_implicit2)

                    ano_score_imp_exp = compress(y_implicit1, y_explicit2)
                    ano_score_exp_imp = compress(y_explicit1, y_implicit2)
                    # ano_score_imp_exp = compress(y_implicit1, y_explicit2) + compress(y_explicit1, y_implicit2)

                    all_implicit.append(y_implicit1.cpu().numpy())
                    all_explicit.append(y_explicit1.cpu().numpy())
                    all_edge_weight.append(batch_aug_edge_weight)
                    all_edge_index.append(data.edge_index)
                    ori_graph.extend([to_networkx(graph) for graph in data.to_data_list()])
                    # aug_data = torch_geometric.data.Data(edge_index=data.edge_index, edge_attr=batch_aug_edge_weight)
                    # aug_data = deepcopy(data)
                    # print(batch_aug_edge_weight.shape)
                    copy_data = deepcopy(data)

                    edge_batch = torch.tensor([copy_data.batch[edge[0]] for edge in copy_data.edge_index.t()])

                    for gid in range(copy_data.num_graphs):
                        graph = copy_data.get_example(gid)
                        # print('***')
                        # print(graph)
                        edge_attr_mask = edge_batch == gid
                        graph.edge_attr = batch_aug_edge_weight[edge_attr_mask]
                        edge_mask = graph.edge_attr >= 0.9
                        new_edge_index = graph.edge_index[:, edge_mask]
                        new_edge_attr = graph.edge_attr[edge_mask]
                        aug_data = Data(edge_index=new_edge_index, edge_attr=new_edge_attr) 
                        # print(aug_data)
                        # print('###')
                        aug_graph.append(to_networkx(aug_data))
                    # aug_data.edge_attr = batch_aug_edge_weight
                    # for gid in range(aug_data.num_graphs):
                        # aug_graph.extend([to_networkx(graph) for graph in aug_data.to_data_list()])

                    if args.is_adaptive:
                        # ano_score = (ano_score_implicit - mean_imp)/std_imp + (ano_score_explicit - mean_exp)/std_exp

                        ano_score = (ano_score_implicit - mean_imp)/std_imp + (ano_score_explicit - mean_exp)/std_exp + ((ano_score_imp_exp - mean_imp_exp)/std_imp_exp + (ano_score_exp_imp - mean_exp_imp)/std_exp_imp) * 0.001

                        # ano_score = (ano_score_implicit - mean_imp)/std_imp + (ano_score_explicit - mean_exp)/std_exp + (ano_score_imp_exp - mean_imp_exp)/std_imp_exp 

                        # ano_score = (ano_score_implicit - mean_imp)/std_imp 
                        # ano_score = (ano_score_explicit - mean_exp)/std_exp 

                        # ano_score = (ano_score_imp_exp - mean_imp_exp)/std_imp_exp + (ano_score_exp_imp - mean_exp_imp)/std_exp_imp
                        # ano_score = (ano_score_implicit - mean_imp)/std_imp + (ano_score_explicit - mean_exp)/std_exp

                        # ano_score = (ano_score_implicit - mean_imp)/std_imp + ((ano_score_imp_exp - mean_imp_exp)/std_imp_exp + (ano_score_exp_imp - mean_exp_imp)/std_exp_imp) * 0.001
                        # ano_score = (ano_score_explicit - mean_exp)/std_exp + ((ano_score_imp_exp - mean_imp_exp)/std_imp_exp + (ano_score_exp_imp - mean_exp_imp)/std_exp_imp) * 0.001
                    else:
                        ano_score = ano_score_implicit + ano_score_explicit + (ano_score_imp_exp + ano_score_exp_imp) * 0.001
                all_ad_score.append(ano_score.cpu())

            ad_true = torch.cat(all_ad_true)
            ad_score = torch.cat(all_ad_score)

            ad_auc = roc_auc_score(ad_true, ad_score)
            # scheduler.step(ad_auc)

            if ad_auc > best_auc:
                # if seed == 0:
                #     with open(args.DS_pair + '/ad_true_' + str(args.is_adaptive) + '_' + str(args.alpha) + '_wo_2.pickle', 'wb') as f:
                #         ad_true = ad_true.numpy()
                #         pickle.dump(ad_true, f)
                #     with open(args.DS_pair + '/ad_score_' + str(args.is_adaptive) + '_' + str(args.alpha) + '_wo_2.pickle', 'wb') as f:
                #         ad_score = ad_score.numpy()
                #     with open(args.DS_pair + '/all_implicit_wo_2.pickle', 'wb') as f:
                #         pickle.dump(all_implicit, f)
                #     with open(args.DS_pair + '/all_explicit_wo_2.pickle', 'wb') as f:
                #         pickle.dump(all_explicit, f)
                # if False:
                #     with open(args.DS_pair + '/ad_true_' + str(args.is_adaptive) + '_' + str(args.alpha) + '.pickle', 'wb') as f:
                #         ad_true = ad_true.numpy()
                #         pickle.dump(ad_true, f)
                #     with open(args.DS_pair + '/ad_score_' + str(args.is_adaptive) + '_' + str(args.alpha) + '.pickle', 'wb') as f:
                #         ad_score = ad_score.numpy()
                #         pickle.dump(ad_score, f)
                #     with open(args.DS_pair + '/all_implicit.pickle', 'wb') as f:
                #         pickle.dump(all_implicit, f)
                #     with open(args.DS_pair + '/all_explicit.pickle', 'wb') as f:
                #         pickle.dump(all_explicit, f)
                #     with open(args.DS_pair + '/adj_hidden.pickle', 'wb') as f:
                #         pickle.dump(model.encoder_explicit.adj_hidden.detach().cpu().numpy(), f)
                #     with open(args.DS_pair + '/edge_weight.pickle', 'wb') as f:
                #         pickle.dump(all_edge_weight, f)
                #     with open(args.DS_pair + '/edge_index.pickle', 'wb') as f:
                #         pickle.dump(all_edge_index, f)
                #     with open(args.DS_pair + '/ori_graph.pickle', 'wb') as f:
                #         pickle.dump(ori_graph, f)
                #     with open(args.DS_pair + '/aug_graph.pickle', 'wb') as f:
                #         pickle.dump(aug_graph, f)
                    
                best_auc = ad_auc
                print('[EVAL] Epoch: {:03d} | AUC:{:.4f}'.format(epoch, best_auc))
                # torch.save(model.state_dict(), 'model/' + args.DS + '_fold' + str(fold) + 'us.pt')

            # info_test = 'AD_AUC:{:.4f}'.format(ad_auc)

    print('[RESULT] Trial: {:02d} | AUC:{:.4f}'.format(seed, best_auc))
    return best_auc
    # aucs.append(ad_auc)


if __name__ == '__main__':
    args = arg_parse()
    ad_aucs = []
   
    if args.exp_type == 'ad':
        if args.DS.startswith('Tox21'):
            # loaders, meta = get_ad_dataset_Tox21(args)
            splits = [None]*args.num_trials
        else:
            splits = get_ad_split_TU(args, fold=args.num_trials)
    else:
       splits = [None]*args.num_trials 

    for trial in range(args.num_trials):
        results = run(args, seed=trial, split=splits[trial])
        ad_auc = results
        ad_aucs.append(ad_auc)
        print(ad_aucs)
        # break
    results = 'AUC: {:.2f}+-{:.2f}'.format(np.mean(ad_aucs) * 100, np.std(ad_aucs) * 100)

    print('[FINAL RESULTS] ' + results)