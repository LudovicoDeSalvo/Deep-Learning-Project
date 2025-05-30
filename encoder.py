import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from torch_geometric.utils import to_networkx
import scipy.sparse.linalg


# Updated ContinuousNodeEncoder with proper initialization
class ContinuousNodeEncoder(nn.Module):
    """Encoder for continuous node features instead of discrete embeddings"""
    def __init__(self, input_dim, embedding_dim):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim // 2, embedding_dim)
        )
        self.embedding_dim = embedding_dim
        
        # Initialize weights properly
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Forward pass for continuous node encoder.
        Automatically reshapes 1D or transposed input tensors and ensures correct dtype and device.
        
        Args:
            x (torch.Tensor): Input tensor of shape [num_nodes, num_features] or [1, num_nodes] (transposed).
            
        Returns:
            torch.Tensor: Output tensor of shape [num_nodes, embedding_dim]
        """
        # Ensure tensor is float
        if x.dtype != torch.float32:
            x = x.float()

        # Fix transposed inputs [1, N] -> [N, 1]
        if x.dim() == 2 and x.shape[0] == 1 and x.shape[1] > 1000:
            x = x.T

        # Handle 1D case [N] -> [N, 1]
        if x.dim() == 1:
            x = x.unsqueeze(-1)

        # Move to the same device as the model parameters
        x = x.to(self.linear[0].weight.device)

        return self.linear(x)


    
def normalize_tensor(tensor):
    min_val = tensor.min(dim=0, keepdim=True)[0]
    max_val = tensor.max(dim=0, keepdim=True)[0]
    norm = (tensor - min_val) / (max_val - min_val + 1e-6)
    return norm

def add_enhanced_features(data):


    # Convert PyG data to undirected NetworkX graph
    G = to_networkx(data, to_undirected=True)

    # Compute structural features using NetworkX
    clustering_dict = nx.clustering(G)
    closeness_dict = nx.closeness_centrality(G)
    betweenness_dict = nx.betweenness_centrality(G, normalized=True)

    # Degree from PyG edge index
    degree = data.edge_index[0].bincount(minlength=data.num_nodes).float().unsqueeze(1)

    # Convert NetworkX features to torch tensors (Nx1)
    clustering = torch.tensor([clustering_dict[i] for i in range(data.num_nodes)], dtype=torch.float32).unsqueeze(1)
    closeness = torch.tensor([closeness_dict[i] for i in range(data.num_nodes)], dtype=torch.float32).unsqueeze(1)
    betweenness = torch.tensor([betweenness_dict[i] for i in range(data.num_nodes)], dtype=torch.float32).unsqueeze(1)

    # Combine and normalize
    features = torch.cat([degree, clustering, closeness, betweenness], dim=1)
    data.x = normalize_tensor(features)



    return data


def add_enhanced_features_fast(data, k=5):
    import scipy
    import scipy.sparse
    import scipy.sparse.linalg
    from torch_geometric.utils import to_networkx

    # Basic structural features
    deg = data.edge_index[0].bincount(minlength=data.num_nodes).float().unsqueeze(1)
    clustering = torch.rand(data.num_nodes, 1)  # placeholder
    centrality = torch.rand(data.num_nodes, 1)  # placeholder

    # Laplacian Positional Encoding
    G = to_networkx(data, to_undirected=True)
    laplacian = nx.normalized_laplacian_matrix(G)

    try:
        eigval, eigvec = scipy.sparse.linalg.eigsh(laplacian, k=min(k, G.number_of_nodes() - 2), which='SM')
        pe = torch.from_numpy(eigvec).float()
    except Exception:
        # Fallback: if eigsh fails (e.g. tiny graph)
        pe = torch.zeros(data.num_nodes, k)

    # Pad if not enough eigenvectors
    if pe.size(1) < k:
        pad = torch.zeros(data.num_nodes, k - pe.size(1))
        pe = torch.cat([pe, pad], dim=1)

    # Final feature matrix
    data.x = torch.cat([deg, clustering, centrality, pe], dim=1)
    return data
