import torch
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.utils import degree
from torch_geometric.nn import GATConv
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import softmax
from torch_geometric.nn import GINEConv

import math

### GIN convolution along the graph structure
class GINConv(MessagePassing):
    def __init__(self, emb_dim):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GINConv, self).__init__(aggr = "add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        self.edge_encoder = torch.nn.Linear(7, emb_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.edge_encoder(edge_attr)
        out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out

### GCN convolution along the graph structure
class GCNConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GCNConv, self).__init__(aggr='add')

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        self.edge_encoder = torch.nn.Linear(7, emb_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.edge_encoder(edge_attr)

        row, col = edge_index

        #edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype = x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr = edge_embedding, norm=norm) + F.relu(x + self.root_emb.weight) * 1./deg.view(-1,1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


### GNN to generate node embedding
class GNN_node(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layer, emb_dim, drop_ratio = 0.5, JK = "last", residual = False, gnn_type = 'gin'):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers

        '''

        super(GNN_node, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.node_encoder = torch.nn.Embedding(1, emb_dim) # uniform input node embedding

        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(emb_dim))
            elif gnn_type == 'gat':
                self.convs.append(EdgeAwareGATv2Conv(emb_dim, emb_dim // 4, heads=4, concat=True, dropout=drop_ratio))
            elif gnn_type == 'gine':
                self.convs.append(GINEConv(emb_dim))
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, batched_data):
        device = batched_data.x.device

        x = batched_data.x.to(device)
        edge_index = batched_data.edge_index.to(device)
        edge_attr = batched_data.edge_attr.to(device)
        batch = batched_data.batch.to(device)

        # Compute input node embeddings
        h_list = [self.node_encoder(x)]

        for layer in range(self.num_layer):
            h_list[layer] = h_list[layer].to(device) 

            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)

        # JK: Jumping Knowledge connection
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = sum(h_list)

        return node_representation



### Virtual GNN to generate node embedding
class GNN_node_Virtualnode(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layer, emb_dim, drop_ratio = 0.5, JK = "last", residual = False, gnn_type = 'gin'):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GNN_node_Virtualnode, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.node_encoder = torch.nn.Embedding(1, emb_dim) # uniform input node embedding

        ### set the initial virtual node embedding to 0.
        self.virtualnode_embedding = torch.nn.Embedding(1, emb_dim)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        ### List of GNNs
        self.convs = torch.nn.ModuleList()
        ### batch norms applied to node embeddings
        self.batch_norms = torch.nn.ModuleList()

        ### List of MLPs to transform virtual node at every layer
        self.mlp_virtualnode_list = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(emb_dim))
            elif gnn_type == 'gat':
                self.convs.append(EdgeAwareGATv2Conv(emb_dim, emb_dim // 4, heads=4, concat=True, dropout=drop_ratio))
            elif gnn_type == 'gine':
                self.convs.append(GINEConv(emb_dim))
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_type))


            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

        for layer in range(num_layer - 1):
            self.mlp_virtualnode_list.append(torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), \
                                                    torch.nn.Linear(2*emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU()))


    def forward(self, batched_data):
        device = batched_data.edge_index.device

        x = batched_data.x.to(device)
        edge_index = batched_data.edge_index.to(device)
        edge_attr = batched_data.edge_attr.to(device)
        batch = batched_data.batch.to(device)

        device = x.device
        self.virtualnode_embedding = self.virtualnode_embedding.to(device)

        idx = torch.zeros(batch[-1].item() + 1, dtype=torch.long, device=device)
        virtualnode_embedding = self.virtualnode_embedding(idx)

        # Type check for node_encoder
        if isinstance(self.node_encoder, torch.nn.Embedding):
            if x.dtype != torch.long:
                x = x.to(torch.long)

        h_list = [self.node_encoder(x)]

        for layer in range(self.num_layer):
            # Make sure virtualnode_embedding is still on the same device
            virtualnode_embedding = virtualnode_embedding.to(device)
            h_list[layer] = h_list[layer].to(device)
            batch = batch.to(device)

            h_list[layer] = h_list[layer] + virtualnode_embedding[batch]

            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

            if self.residual:
                h = h + h_list[layer]

            h_list.append(h)

            if layer < self.num_layer - 1:
                virtualnode_embedding_temp = global_add_pool(h_list[layer], batch) + virtualnode_embedding
                if self.residual:
                    virtualnode_embedding = virtualnode_embedding + F.dropout(
                        self.mlp_virtualnode_list[layer](virtualnode_embedding_temp),
                        self.drop_ratio,
                        training=self.training
                    )
                else:
                    virtualnode_embedding = F.dropout(
                        self.mlp_virtualnode_list[layer](virtualnode_embedding_temp),
                        self.drop_ratio,
                        training=self.training
                    )

        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = sum(h_list)

        return node_representation

from torch.nn import TransformerEncoder, TransformerEncoderLayer

class GraphTransformer(torch.nn.Module):
    def __init__(self, input_dim, emb_dim, num_heads=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.input_proj = torch.nn.Linear(input_dim, emb_dim)
        encoder_layer = TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.emb_dim = emb_dim

    def forward(self, x, batch):
        # Group nodes by graphs for transformer input
        max_nodes = max((batch == i).sum().item() for i in batch.unique())
        padded_x = torch.zeros(batch.max().item() + 1, max_nodes, self.emb_dim, device=x.device)

        for i in range(batch.max().item() + 1):
            nodes = x[batch == i]
            padded_x[i, :nodes.size(0), :] = self.input_proj(nodes)

        out = self.transformer(padded_x)
        mask = (padded_x.sum(dim=2) != 0).float()
        graph_embeddings = (out * mask.unsqueeze(2)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
        return graph_embeddings


class EdgeAwareGATv2Conv(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=4, concat=True, dropout=0.0):
        super().__init__(aggr='add', node_dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout

        self.lin = torch.nn.Linear(in_channels, heads * out_channels, bias=False)
        self.edge_encoder = torch.nn.Linear(7, heads * out_channels, bias=False)
        self.att = torch.nn.Parameter(torch.Tensor(1, heads, out_channels))

        self.bias = torch.nn.Parameter(torch.Tensor(heads * out_channels if concat else out_channels))
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=0.2)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin.weight)
        torch.nn.init.xavier_uniform_(self.edge_encoder.weight)
        torch.nn.init.xavier_uniform_(self.att)
        torch.nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, edge_attr):
        H, C = self.heads, self.out_channels

        x = self.lin(x)  # [N, H * C]
        edge_attr = self.edge_encoder(edge_attr)  # [E, H * C]

        x = x.view(-1, H, C)  # [N, H, C]
        edge_attr = edge_attr.view(-1, H, C)  # [E, H, C]

        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr, index, ptr, size_i):
        # x_i, x_j: [E, H, C], edge_attr: [E, H, C]
        xj_e = x_j + torch.tanh(self.edge_encoder_output_scaler * edge_attr)
        alpha = (x_i * self.att).sum(dim=-1) + (xj_e * self.att).sum(dim=-1)
        alpha = self.leaky_relu(alpha)
        alpha = softmax(alpha, index)  # <-- this is critical
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return xj_e * alpha.unsqueeze(-1)

    def update(self, aggr_out):
        if self.concat:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)
        return aggr_out + self.bias


class GINEConv(MessagePassing):
    def __init__(self, emb_dim):
        super().__init__(aggr="add")

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, 2 * emb_dim),
            torch.nn.BatchNorm1d(2 * emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * emb_dim, emb_dim)
        )

        self.edge_encoder = torch.nn.Sequential(
            torch.nn.Linear(7, emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim, emb_dim)
        )

        self.eps = torch.nn.Parameter(torch.Tensor([0]))

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.edge_encoder(edge_attr)
        return self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out
