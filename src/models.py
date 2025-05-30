import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, Set2Set
from torch_geometric.nn.aggr import AttentionalAggregation
import torch.nn.functional as F
from torch_geometric.nn.inits import uniform
from torch_geometric.nn import GINEConv, global_mean_pool
from torch.nn import Linear, BatchNorm1d, ReLU, Dropout, Sequential

from src.conv import GNN_node, GNN_node_Virtualnode, GraphTransformer

class GNN(torch.nn.Module):

    def __init__(self, num_class, num_layer=5, emb_dim=300, gnn_type='gin', virtual_node=True,
            residual=False, drop_ratio=0.5, JK="last", graph_pooling="attention", input_dim=None):

        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''

        super(GNN, self).__init__()
        self.norm = torch.nn.LayerNorm(emb_dim)

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_class = num_class
        self.graph_pooling = graph_pooling

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        if gnn_type == 'transformer':
            assert input_dim is not None, "Transformer GNN requires input_dim"
            self.gnn_node = GraphTransformer(input_dim=input_dim, emb_dim=emb_dim)
        elif virtual_node:
            self.gnn_node = GNN_node_Virtualnode(num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type)
        else:
            self.gnn_node = GNN_node(num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type)


        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
                self.pool = AttentionalAggregation(gate_nn=torch.nn.Sequential(
                torch.nn.Linear(emb_dim, 2 * emb_dim),
                torch.nn.BatchNorm1d(2 * emb_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(2 * emb_dim, 1)
            ))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps = 2)
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2*self.emb_dim, self.num_class)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_class)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        if isinstance(self.gnn_node, GraphTransformer):
            # If using transformer, pass (x, batch) directly
            graph_embedding = self.gnn_node(x, batch)
        else:
            # Classic GNN pipeline: get node embeddings then pool
            node_embeddings = self.gnn_node(data)
            graph_embedding = self.pool(node_embeddings, batch)

        graph_embedding = self.norm(graph_embedding)
        graph_embedding = F.dropout(graph_embedding, p=self.drop_ratio, training=self.training)
        out = self.graph_pred_linear(graph_embedding)

        return out, graph_embedding  # logits, embeddings
    



class GINEModelWithVirtualNode(torch.nn.Module):
    def __init__(self, num_features=1, num_classes=6, num_layers=5, emb_dim=300, drop_ratio=0.0, edge_dim=7):
        super().__init__()
        
        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.emb_dim = emb_dim
        self.edge_dim = edge_dim

        # Node & edge encoders
        self.node_encoder = torch.nn.Embedding(num_features, emb_dim)
        self.edge_encoder = torch.nn.Linear(edge_dim, emb_dim)

        # Virtual node embedding (1 per graph)
        self.virtualnode_embedding = torch.nn.Embedding(1, emb_dim)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        # GINE layers
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.mlp_virtualnode_list = torch.nn.ModuleList()

        for i in range(num_layers):
            mlp = Sequential(
                Linear(emb_dim, emb_dim),
                BatchNorm1d(emb_dim),
                ReLU(),
                Linear(emb_dim, emb_dim)
            )
            self.convs.append(GINEConv(mlp, train_eps=True))
            self.batch_norms.append(BatchNorm1d(emb_dim))

        # MLPs to update virtual node
        for _ in range(num_layers - 1):
            self.mlp_virtualnode_list.append(Sequential(
                Linear(emb_dim, 2 * emb_dim),
                BatchNorm1d(2 * emb_dim),
                ReLU(),
                Linear(2 * emb_dim, emb_dim),
                BatchNorm1d(emb_dim),
                ReLU()
            ))

        # Classifier
        self.classifier = Sequential(
            Linear(emb_dim, emb_dim // 2),
            ReLU(),
            Dropout(drop_ratio),
            Linear(emb_dim // 2, num_classes)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Handle edge attributes - create default if missing
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            edge_attr = data.edge_attr.float()
        else:
            edge_attr = torch.ones(edge_index.size(1), self.edge_dim, device=edge_index.device)

        # Encode node and edge features
        if x.dtype != torch.long or x.ndim == 2:
            x = x.squeeze(-1).long()
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        # Ensure batch is on the correct device
        device = x.device
        batch = batch.to(device)

        # Prepare virtual node embedding (one per graph)
        idx = torch.zeros(batch.max().item() + 1, dtype=torch.long, device=device)
        virtualnode_embedding = self.virtualnode_embedding.weight[idx]

        # Apply GINE layers with virtual node updates
        for i in range(self.num_layers):
            # Add virtual node embedding to node features
            x = x + virtualnode_embedding[batch]

            x = self.convs[i](x, edge_index, edge_attr)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_ratio, training=self.training)

            if i < self.num_layers - 1:
                pooled = global_add_pool(x, batch)
                virtualnode_embedding_temp = virtualnode_embedding + pooled
                virtualnode_embedding = virtualnode_embedding + F.dropout(
                    self.mlp_virtualnode_list[i](virtualnode_embedding_temp),
                    self.drop_ratio,
                    training=self.training
                )

        graph_emb = global_add_pool(x, batch)
        return self.classifier(graph_emb), graph_emb

