import torch
import torch.nn.functional as F
# import torch_scatter
# import torch.nn as nn
from graph_learners import *
from view_learner import ViewLearner
from torch.nn import Sequential, Linear, ReLU, Parameter, BatchNorm1d, Dropout, Sigmoid
from torch_geometric.nn import GINConv, global_add_pool, global_max_pool
from torch_geometric.nn.conv import MessagePassing
# from torch_geometric.utils import softmax
# from compress_loss import CompReSSMomentum
import copy
from torch_scatter import scatter

from wgin_conv import WGINConv

class KBGAD(nn.Module):
    def __init__(self, input_dim, args, device):
        super(KBGAD, self).__init__()
        self.device = device
        self.DS = args.DS
        self.embedding_dim = args.hidden_dim
        self.maskfeat_rate_learner = args.maskfeat_rate_learner

        if args.readout == 'concat':
            self.embedding_dim *= args.encoder_layers


        # if args.type_learner == 'mlp':
        #     self.graph_learner = MLP_learner(2, input_dim, args.hidden_dim, args.k, args.sim_function, 6, args.sparse, args.activation_learner)
        # elif args.type_learner == 'att':
        #     self.graph_learner = ATT_learner(2, input_dim, args.hidden_dim, args.k, args.sim_function, 6, args.sparse, args.activation_learner)
        # elif args.type_learner == 'gnn':
        #     self.graph_learner = GNN_learner(2, input_dim, args.hidden_dim, args.k, args.sim_function, 6, args.sparse, args.activation_learner)

        if args.type_learner == 'mlp':
            self.view_learner = ViewLearner(args.type_learner, MLP_learner(args.encoder_layers, input_dim, args.hidden_dim, args.dropout))
        elif args.type_learner == 'gnn':
            self.view_learner = ViewLearner(args.type_learner, GNN_learner(args.encoder_layers, input_dim, args.hidden_dim, args.dropout))
        elif args.type_learner == 'att':
            self.view_learner = ViewLearner(args.type_learner, ATT_learner(args.encoder_layers, input_dim, args.hidden_dim, args.dropout))
        else:
            self.view_learner = None

        self.encoder_implicit = GIN(input_dim, args.hidden_dim, args.encoder_layers, args.dropout, args.pooling, args.readout)
        self.encoder_explicit = RW_NN(input_dim, args.hidden_dim, args.hidden_graphs, args.size_hidden_graphs, 
                                    args.max_step, args.dropout, args.normalize, device)

        # self.encoder_implicit1 = GIN(input_dim, args.hidden_dim, args.encoder_layers, args.dropout, args.pooling, args.readout)
        # self.encoder_explicit1 = RW_NN(input_dim, args.hidden_dim, args.hidden_graphs, args.size_hidden_graphs, 
        #                               args.max_step, args.dropout, args.normalize, device)


        self.proj_head_implicit = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
                                       nn.Linear(self.embedding_dim, self.embedding_dim))
        
        self.proj_head_explicit = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
                                       nn.Linear(self.embedding_dim, self.embedding_dim))


        self.proj_head_implicit1 = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
                                       nn.Linear(self.embedding_dim, self.embedding_dim))
        
        self.proj_head_explicit1 = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
                                       nn.Linear(self.embedding_dim, self.embedding_dim))
        
        # self.compress = CompReSSMomentum(args)


        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, data):

        if self.view_learner:
            edge_logits = self.view_learner(data.x, data.edge_index)

            temperature = 1.0
            bias = 0.0 + 0.0001  # If bias is 0, we run into problems
            eps = (bias - (1 - bias)) * torch.rand(edge_logits.size()) + (1 - bias)
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs = gate_inputs.to(self.device)
            gate_inputs = (gate_inputs + edge_logits) / temperature
            batch_aug_edge_weight = torch.sigmoid(gate_inputs).squeeze()

            if self.maskfeat_rate_learner:
                mask, _ = get_feat_mask(data.x, self.maskfeat_rate_learner)
                features_v2 = data.x * (1 - mask)
            else:
                features_v2 = copy.deepcopy(data.x)
        else:
            features_v2 = data.x
            batch_aug_edge_weight = None



        y_implicit1 = self.encoder_implicit(data.x, data.edge_index, None, data.batch)
        y_explicit1 = self.encoder_explicit(data.x, data.edge_index, None, data.batch)
      
        # if self.DS in ['DD']:
        # y_implicit2 = self.encoder_implicit(features_v2, data.edge_index, batch_aug_edge_weight, data.batch)
        # y_explicit2 = self.encoder_explicit(features_v2, data.edge_index, batch_aug_edge_weight, data.batch)
        # else:
        y_implicit2 = self.encoder_implicit(data.x, data.edge_index, batch_aug_edge_weight, data.batch)
        y_explicit2 = self.encoder_explicit(data.x, data.edge_index, batch_aug_edge_weight, data.batch)

        # y_implicit2 = self.encoder_implicit(data.x, data.edge_index, None, data.batch)
        # y_explicit2 = self.encoder_explicit(data.x, data.edge_index, None, data.batch)
        

        y_implicit1 = self.proj_head_implicit(y_implicit1)
        y_explicit1 = self.proj_head_explicit(y_explicit1)

        y_implicit2 = self.proj_head_implicit1(y_implicit2)
        y_explicit2 = self.proj_head_explicit1(y_explicit2)

        # y_implicit2 = self.proj_head_implicit1(y_implicit2)
        # y_explicit2 = self.proj_head_explicit1(y_explicit2)

        if self.view_learner:
            row, col = data.edge_index
            edge_batch = data.batch[row]
            edge_drop_out_prob = 1 - batch_aug_edge_weight

            uni, edge_batch_num = edge_batch.unique(return_counts=True)
            sum_pe = scatter(edge_drop_out_prob, edge_batch, reduce="sum")

            reg = []
            for b_id in range(data.num_graphs):
                if b_id in uni:
                    num_edges = edge_batch_num[uni.tolist().index(b_id)]
                    reg.append(sum_pe[b_id] / num_edges)
                else:
                    # means no edges in that graph. So don't include.
                    pass
            num_graph_with_edges = len(reg)
            reg = torch.stack(reg)
        else:
            reg = None
        # reg = reg.mean()
        
        return y_implicit1, y_explicit1, y_implicit2, y_explicit2, reg, batch_aug_edge_weight

    @staticmethod
    def loss_nce(x1, x2, temperature=0.2):
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / temperature)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]

        loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim + 1e-10)
        loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim + 1e-10)

        loss_0 = - torch.log(loss_0 + 1e-10)
        loss_1 = - torch.log(loss_1 + 1e-10)
        loss = (loss_0 + loss_1) / 2.0
        return loss

    # # @staticmethod
    # def loss_nce_imp(self, x1, x2, temperature=0.2):
    #     batch_size, _ = x1.size()
    #     x1 = self.proj_head_implicit(x1)
    #     x2 = self.proj_head_implicit1(x2)
    #     x1_abs = x1.norm(dim=1)
    #     x2_abs = x2.norm(dim=1)

    #     sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
    #     sim_matrix = torch.exp(sim_matrix / temperature)
    #     pos_sim = sim_matrix[range(batch_size), range(batch_size)]

    #     loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim + 1e-10)
    #     loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim + 1e-10)

    #     loss_0 = - torch.log(loss_0 + 1e-10)
    #     loss_1 = - torch.log(loss_1 + 1e-10)
    #     loss = (loss_0 + loss_1) / 2.0
    #     return loss

    # # @staticmethod
    # def loss_nce_exp(self, x1, x2, temperature=0.2):
    #     batch_size, _ = x1.size()
    #     x1 = self.proj_head_explicit(x1)
    #     x2 = self.proj_head_explicit1(x2)
    #     x1_abs = x1.norm(dim=1)
    #     x2_abs = x2.norm(dim=1)

    #     sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
    #     sim_matrix = torch.exp(sim_matrix / temperature)
    #     pos_sim = sim_matrix[range(batch_size), range(batch_size)]

    #     loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim + 1e-10)
    #     loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim + 1e-10)

    #     loss_0 = - torch.log(loss_0 + 1e-10)
    #     loss_1 = - torch.log(loss_1 + 1e-10)
    #     loss = (loss_0 + loss_1) / 2.0
    #     return loss

    # def loss_kl(self, x1, x2):
    #     # x1_abs = x1.norm(dim=1)
    #     # x2_abs = x2.norm(dim=1)
    #     # return self.compress(x1_abs, x2_abs)
    #     return self.compress(x1, x2)

class RW_NN(MessagePassing):
    def __init__(self, num_features, hidden_dim, hidden_graphs, size_hidden_graphs, max_step, dropout, normalize, device):
        super(RW_NN, self).__init__()
        self.max_step = max_step
        self.hidden_graphs = hidden_graphs
        self.size_hidden_graphs = size_hidden_graphs
        self.normalize = normalize
        # self.device = device
        self.adj_hidden = Parameter(torch.FloatTensor(hidden_graphs, (size_hidden_graphs*(size_hidden_graphs-1))//2))
        self.features_hidden = Parameter(torch.FloatTensor(hidden_graphs, size_hidden_graphs, hidden_dim))
        self.fc = Linear(num_features, hidden_dim)
        self.bn = BatchNorm1d(hidden_graphs * max_step)
        self.fc1 = Linear(hidden_graphs * max_step, hidden_dim)
        # self.fc2 = Linear(args.penultimate_dim, args.num_classes)
        self.dropout = Dropout(p=dropout)
        self.relu = ReLU()
        self.sigmoid = Sigmoid()
        # self.proj_head = Sequential(Linear(penultimate_dim, hidden_dim), ReLU(inplace=True), Linear(hidden_dim, hidden_dim))
        
        self.device = device
        self.init_weights()

    def init_weights(self):
        self.adj_hidden.data.uniform_(-1, 1)
        self.features_hidden.data.uniform_(0, 1)
        
    # def forward(self, adj, features, graph_indicator):
    def forward(self, x, edge_index, edge_weight, batch):    
        # print(edge_index)
        # adj = data.edge_index
        # features = data.x
        # graph_indicator = data.batch
        # print(edge_index.indices())
        unique, counts = torch.unique(batch, return_counts=True)
        n_graphs = unique.size(0)
        # n_nodes = x.size(0)

        if self.normalize:
            norm = counts.unsqueeze(1).repeat(1, self.hidden_graphs)
        
        adj_hidden_norm = torch.zeros(self.hidden_graphs, self.size_hidden_graphs, self.size_hidden_graphs).to(self.device)
        idx = torch.triu_indices(self.size_hidden_graphs, self.size_hidden_graphs, 1)
        adj_hidden_norm[:,idx[0],idx[1]] = self.relu(self.adj_hidden)
        adj_hidden_norm = adj_hidden_norm + torch.transpose(adj_hidden_norm, 1, 2)
        x = self.sigmoid(self.fc(x))
        z = self.features_hidden
        zx = torch.einsum("abc,dc->abd", (z, x))
        # zx = torch.ones((zx.shape[0], zx.shape[1], zx.shape[2])).to(self.device)
        
        out = list()
        for i in range(self.max_step):
            if i == 0:
                eye = torch.eye(self.size_hidden_graphs, device=self.device)
                eye = eye.repeat(self.hidden_graphs, 1, 1)              
                o = torch.einsum("abc,acd->abd", (eye, z))
                t = torch.einsum("abc,dc->abd", (o, x))
            else:
                # x = torch.spmm(adj, x)
                x = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
                z = torch.einsum("abc,acd->abd", (adj_hidden_norm, z))
                t = torch.einsum("abc,dc->abd", (z, x))
            t = self.dropout(t)
            t = torch.mul(zx, t)
            t = torch.zeros(t.size(0), t.size(1), n_graphs, device=self.device).index_add_(2, batch, t)
            t = torch.sum(t, dim=1)
            t = torch.transpose(t, 0, 1)
            if self.normalize:
                t /= norm
            out.append(t)
            
        out = torch.cat(out, dim=1)
        # out = self.bn(out)
        out = self.relu(self.fc1(out))
        # out = self.dropout(out)
        # z = self.proj_head(out)
        # return z
        return out

class GIN(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers, drop_ratio, pooling, readout):
        super(GIN, self).__init__()

        self.num_gc_layers = num_gc_layers
        self.pooling = pooling
        self.readout = readout
        self.drop_ratio = drop_ratio

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.dim = dim
        self.pool = self.get_pool()

        for i in range(num_gc_layers):
            if i:
                mlp = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
            else:
                mlp = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
            # conv = GINConv(mlp)
            conv = WGINConv(mlp)
            bn = torch.nn.BatchNorm1d(dim)

            self.convs.append(conv)
            self.bns.append(bn)

    def forward(self, x, edge_index, edge_weight, batch):
        xs = []
        for i in range(self.num_gc_layers):

            # x = F.relu(self.convs[i](x, edge_index))
            x = F.relu(self.convs[i](x, edge_index, edge_weight))

            # x = self.convs[i](x, edge_index, edge_weight)
            # # x = self.bns[i](x)
            # if i == self.num_gc_layers - 1:
            #     x = F.dropout(x, self.drop_ratio, training=self.training)
            # else:
            #     x = F.dropout(F.relu(x), self.drop_ratio, training=self.training)

            xs.append(x)

        if self.readout == 'last':
            graph_emb = self.pool(xs[-1], batch)
        elif self.readout == 'concat':
            graph_emb = torch.cat([self.pool(x, batch) for x in xs], 1)
        elif self.readout == 'add':
            graph_emb = 0
            for x in xs:
                graph_emb += self.pool(x, batch)

        return graph_emb

    def get_pool(self):
        if self.pooling == 'add':
            pool = global_add_pool
        elif self.pooling == 'max':
            pool = global_max_pool
        else:
            raise ValueError("Pooling Name <{}> is Unknown".format(self.pooling))
        return pool