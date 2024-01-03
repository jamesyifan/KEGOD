# import dgl
import torch
import torch.nn as nn

from layers import Attentive
from utils import *
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv
from wgin_conv import WGINConv


# class FGP_learner(nn.Module):
#     def __init__(self, features, k, knn_metric, i, sparse):
#         super(FGP_learner, self).__init__()

#         self.k = k
#         self.knn_metric = knn_metric
#         self.i = i
#         self.sparse = sparse

#         self.Adj = nn.Parameter(
#             torch.from_numpy(nearest_neighbors_pre_elu(features, self.k, self.knn_metric, self.i)))

#     def forward(self, h):
#         if not self.sparse:
#             Adj = F.elu(self.Adj) + 1
#         else:
#             Adj = self.Adj.coalesce()
#             Adj.values = F.elu(Adj.values()) + 1
#         return Adj





# class MLP_learner(nn.Module):
#     def __init__(self, nlayers, isize, k, knn_metric, i, sparse, act):
#         super(MLP_learner, self).__init__()

#         self.layers = nn.ModuleList()
#         if nlayers == 1:
#             self.layers.append(nn.Linear(isize, isize))
#         else:
#             self.layers.append(nn.Linear(isize, isize))
#             for _ in range(nlayers - 2):
#                 self.layers.append(nn.Linear(isize, isize))
#             self.layers.append(nn.Linear(isize, isize))

#         self.input_dim = isize
#         self.output_dim = isize
#         self.k = k
#         self.knn_metric = knn_metric
#         self.non_linearity = 'relu'
#         self.param_init()
#         self.i = i
#         self.sparse = sparse
#         self.act = act

#     def internal_forward(self, h):
#         for i, layer in enumerate(self.layers):
#             h = layer(h)
#             if i != (len(self.layers) - 1):
#                 if self.act == "relu":
#                     h = F.relu(h)
#                 elif self.act == "tanh":
#                     h = F.tanh(h)
#         return h

#     def param_init(self):
#         for layer in self.layers:
#             layer.weight = nn.Parameter(torch.eye(self.input_dim))

#     def forward(self, features, edge_index):
#         if self.sparse:
#             embeddings = self.internal_forward(features)
#             rows, cols, values = knn_fast(embeddings, self.k, 1000)
#             rows_ = torch.cat((rows, cols))
#             cols_ = torch.cat((cols, rows))
#             values_ = torch.cat((values, values))
#             values_ = apply_non_linearity(values_, self.non_linearity, self.i)

#             # adj = dgl.graph((rows_, cols_), num_nodes=features.shape[0], device='cuda')
#             # adj.edata['w'] = values_
#             # edge_index_ = tuple(rows_, cols_)
#             edge_index_ = torch.stack([rows_, cols_], dim=0)
#             # return adj
#             return edge_index_, values_
#         else:
#             embeddings = self.internal_forward(features)
#             embeddings = F.normalize(embeddings, dim=1, p=2)
#             similarities = cal_similarity_graph(embeddings)
#             similarities = top_k(similarities, self.k + 1)
#             similarities = apply_non_linearity(similarities, self.non_linearity, self.i)
#             return similarities


# class GNN_learner(nn.Module):
#     def __init__(self, nlayers, isize, hidden_dim, k, knn_metric, i, sparse, mlp_act):
#         super(GNN_learner, self).__init__()

#         # self.adj = adj
#         # self.edge_index = edge_index
#         self.nlayers = nlayers
#         self.layers = nn.ModuleList()

#         for i in range(nlayers):
#             if i:
#                 mlp = Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
#             else:
#                 mlp = Sequential(Linear(isize, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
#             # mlp = Sequential(Linear(isize, isize), ReLU(), Linear(isize, isize))
#             conv = GINConv(mlp)
#             self.layers.append(conv)

#         # if nlayers == 1:
#         #     # self.layers.append(GCNConv_dgl(isize, isize))
#         #     self.layers.append()
#         # else:
#         #     self.layers.append(GCNConv_dgl(isize, isize))
#         #     for _ in range(nlayers - 2):
#         #         self.layers.append(GCNConv_dgl(isize, isize))
#         #     self.layers.append(GCNConv_dgl(isize, isize))

#         self.input_dim = isize
#         self.hidden_dim = hidden_dim
#         self.k = k
#         self.knn_metric = knn_metric
#         self.non_linearity = 'relu'
#         self.param_init()
#         self.i = i
#         self.sparse = sparse
#         self.mlp_act = mlp_act

#     def internal_forward(self, x, edge_index):
#         for i, layer in enumerate(self.layers):
#             x = layer(x, edge_index)
#             if i != (len(self.layers) - 1):
#                 if self.mlp_act == "relu":
#                     x = F.relu(x)
#                 elif self.mlp_act == "tanh":
#                     x = F.tanh(x)
#         return x

#     def param_init(self):
#         for layer in self.layers:
#             layer.weight = nn.Parameter(torch.eye(self.input_dim))

#     def forward(self, x, edge_index):
#         if self.sparse:
#             embeddings = self.internal_forward(x, edge_index)
#             rows, cols, values = knn_fast(embeddings, self.k, 1000)
#             rows_ = torch.cat((rows, cols))
#             cols_ = torch.cat((cols, rows))
#             values_ = torch.cat((values, values))
#             values_ = apply_non_linearity(values_, self.non_linearity, self.i)

#             # adj = dgl.graph((rows_, cols_), num_nodes=features.shape[0], device='cuda')
#             # adj.edata['w'] = values_
#             # edge_index_ = tuple(zip(rows_, cols_))
#             edge_index_ = torch.stack([rows_, cols_], dim=0)
#             # edge_index_ = torch.stack([rows_, cols_], dim=1)

#             # edge_index_ = torch.sparse_coo_tensor(indices=torch.stack([rows_, cols_], dim=0), values=values_).coalesce()

#             # print(edge_index_)
#             # print(x.shape)
#             # # import time
#             # time.sleep(100)
#             return edge_index_, values_
#         else:
#             embeddings = self.internal_forward(x, edge_index)
#             embeddings = F.normalize(embeddings, dim=1, p=2)
#             similarities = cal_similarity_graph(embeddings)
#             similarities = top_k(similarities, self.k + 1)
#             similarities = apply_non_linearity(similarities, self.non_linearity, self.i)
#             return similarities

# class ATT_learner(nn.Module):
#     def __init__(self, nlayers, isize, k, knn_metric, i, sparse, mlp_act):
#         super(ATT_learner, self).__init__()

#         self.i = i
#         self.layers = nn.ModuleList()
#         for _ in range(nlayers):
#             self.layers.append(Attentive(isize))
#         self.k = k
#         self.knn_metric = knn_metric
#         self.non_linearity = 'relu'
#         self.sparse = sparse
#         self.mlp_act = mlp_act

#     def internal_forward(self, h):
#         for i, layer in enumerate(self.layers):
#             h = layer(h)
#             if i != (len(self.layers) - 1):
#                 if self.mlp_act == "relu":
#                     h = F.relu(h)
#                 elif self.mlp_act == "tanh":
#                     h = F.tanh(h)
#         return h

#     def forward(self, features, edge_index):
#         if self.sparse:
#             embeddings = self.internal_forward(features)
#             rows, cols, values = knn_fast(embeddings, self.k, 1000)
#             rows_ = torch.cat((rows, cols))
#             cols_ = torch.cat((cols, rows))
#             values_ = torch.cat((values, values))
#             values_ = apply_non_linearity(values_, self.non_linearity, self.i)

#             # adj = dgl.graph((rows_, cols_), num_nodes=features.shape[0], device='cuda')
#             # adj.edata['w'] = values_
#             # return adj
#             # edge_index_ = tuple(rows_, cols_)
#             edge_index_ = torch.stack([rows_, cols_], dim=0)
#             return edge_index_, values_
#         else:
#             embeddings = self.internal_forward(features)
#             embeddings = F.normalize(embeddings, dim=1, p=2)
#             similarities = cal_similarity_graph(embeddings)
#             similarities = top_k(similarities, self.k + 1)
#             similarities = apply_non_linearity(similarities, self.non_linearity, self.i)
#             return similarities

class ATT_learner(nn.Module):
    def __init__(self, num_gc_layers, isize, hidden_dim, drop_ratio):
        super(ATT_learner, self).__init__()

        # self.i = i
        self.num_gc_layers = num_gc_layers
        self.drop_ratio = drop_ratio
        self.out_node_dim = hidden_dim
        self.layers = nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for _ in range(num_gc_layers):
            self.layers.append(Attentive(isize))
            bn = torch.nn.BatchNorm1d(isize)
            self.bns.append(bn)
        # self.k = k
        # self.knn_metric = knn_metric
        # self.non_linearity = 'relu'
        # self.sparse = sparse
        # self.mlp_act = mlp_act

    # def internal_forward(self, h):
    #     # for i, layer in enumerate(self.layers):
    #     for i in range(self.num_gc_layers):
    #         h = self.layers[i](h)
    #         h = self.bns[i](h)
    #         if i != (len(self.layers) - 1):
    #             if self.mlp_act == "relu":
    #                 h = F.relu(h)
    #             elif self.mlp_act == "tanh":
    #                 h = F.tanh(h)
    #     return h

    def forward(self, x, edge_index):
        for i in range(self.num_gc_layers):
            x = self.layers[i](x)
            x = self.bns[i](x)
            if i == self.num_gc_layers - 1:
            #     # remove relu for the last layer
                x = F.dropout(x, self.drop_ratio, training=self.training)
            else:
                x = F.dropout(F.relu(x), self.drop_ratio, training=self.training)
            # if i != (len(self.layers) - 1):
            #     if self.mlp_act == "relu":
            #         h = F.relu(h)
            #     elif self.mlp_act == "tanh":
            #         h = F.tanh(h)
        return x

class GNN_learner(nn.Module):
    def __init__(self, num_gc_layers, isize, hidden_dim, drop_ratio):
        super(GNN_learner, self).__init__()

        self.num_gc_layers = num_gc_layers
        self.drop_ratio = drop_ratio
        self.out_node_dim = hidden_dim
        self.convs  = nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        for i in range(num_gc_layers):
            if i:
                mlp = Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
            else:
                mlp = Sequential(Linear(isize, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
            conv = GINConv(mlp)
            # conv = WGINConv(mlp)
            bn = torch.nn.BatchNorm1d(hidden_dim)

            self.convs.append(conv)
            self.bns.append(bn)

    def forward(self, x, edge_index):
        for i in range(self.num_gc_layers):
            # x = F.relu(self.convs[i](x, edge_index))
            # x = F.relu(self.convs[i](x, edge_index, None))

            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            if i == self.num_gc_layers - 1:
                # remove relu for the last layer
                x = F.dropout(x, self.drop_ratio, training=self.training)
            else:
                x = F.dropout(F.relu(x), self.drop_ratio, training=self.training)
        return x


class MLP_learner(nn.Module):
    def __init__(self, num_gc_layers, isize, hidden_dim, drop_ratio):
        super(MLP_learner, self).__init__()

        self.num_gc_layers = num_gc_layers
        self.drop_ratio = drop_ratio
        self.out_node_dim = hidden_dim
        self.convs  = nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        for i in range(num_gc_layers):
            if i:
                mlp = Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
            else:
                mlp = Sequential(Linear(isize, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
            bn = torch.nn.BatchNorm1d(hidden_dim)
            self.convs.append(mlp)
            self.bns.append(bn)


    def forward(self, x, edge_index):
        for i in range(self.num_gc_layers):
            # x = F.relu(self.convs[i](x, edge_index))
            # x = F.relu(self.convs[i](x, edge_index, None))

            x = self.convs[i](x)
            x = self.bns[i](x)
            if i == self.num_gc_layers - 1:
            #     # remove relu for the last layer
                x = F.dropout(x, self.drop_ratio, training=self.training)
            else:
                x = F.dropout(F.relu(x), self.drop_ratio, training=self.training)
        return x