import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv, global_mean_pool, global_max_pool
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU

# 自定义全局最小池化函数，兼容旧版本PyTorch Geometric
def global_min_pool(x, batch, size=None):
    r"""Returns the min pooling of nodes in each graph in the batch.
    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + N_2 + ... + N_B) \times F}`.
        batch (LongTensor): Batch vector :math:`\mathbf{b} \in {\{ 0, ..., B-1 \}}^N`,
            which assigns each node to a specific graph.
        size (int, optional): Number of graphs :math:`B`. Automatically determined
            if not given. (default: :obj:`None`)
    """
    if size is None:
        size = batch.max().item() + 1
    return torch.zeros(size, x.size(1), device=x.device).scatter_reduce_(
        0, batch.unsqueeze(1).expand_as(x), x, reduce='amin', include_self=False)

class GNNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, model_type, pooling_type):
        super(GNNModel, self).__init__()
        self.model_type = model_type
        self.pooling_type = pooling_type
        self.convs = torch.nn.ModuleList()
        
        # 输入层
        if model_type == 'GCN':
            self.convs.append(GCNConv(input_dim, hidden_dim))
        elif model_type == 'GAT':
            self.convs.append(GATConv(input_dim, hidden_dim))
        elif model_type == 'GraphSAGE':
            self.convs.append(SAGEConv(input_dim, hidden_dim))
        elif model_type == 'GIN':
            mlp = Sequential(Linear(input_dim, hidden_dim), BatchNorm1d(hidden_dim), ReLU(),
                             Linear(hidden_dim, hidden_dim))
            self.convs.append(GINConv(mlp))
        
        # 隐藏层
        for _ in range(num_layers - 1):
            if model_type == 'GCN':
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            elif model_type == 'GAT':
                self.convs.append(GATConv(hidden_dim, hidden_dim))
            elif model_type == 'GraphSAGE':
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            elif model_type == 'GIN':
                mlp = Sequential(Linear(hidden_dim, hidden_dim), BatchNorm1d(hidden_dim), ReLU(),
                                 Linear(hidden_dim, hidden_dim))
                self.convs.append(GINConv(mlp))
        
        # 输出层
        self.lin = Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):
        # 图卷积层
        for conv in self.convs:
            x = conv(x, edge_index)
            x = x.relu()
        
        # 池化层
        if self.pooling_type == 'AvgPooling':
            x = global_mean_pool(x, batch)
        elif self.pooling_type == 'MaxPooling':
            x = global_max_pool(x, batch)
        elif self.pooling_type == 'MinPooling':
            x = global_min_pool(x, batch)  # 使用自定义的最小池化
        
        # 输出层
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x
