import torch
from torch_geometric.nn import HeteroConv, GATConv, Linear
from torch_geometric.nn.aggr import MultiAggregation

try:
    convs = {('A', 'rel1', 'B'): Linear(16, 16), ('C', 'rel2', 'B'): Linear(16, 16)}
    conv = HeteroConv(convs, aggr=MultiAggregation(['mean', 'max', 'sum']))
    x_dict = {'A': torch.randn(10, 16), 'C': torch.randn(10, 16), 'B': torch.randn(10, 16)}
    edge_index_dict = {('A', 'rel1', 'B'): torch.randint(0, 10, (2, 20)), ('C', 'rel2', 'B'): torch.randint(0, 10, (2, 20))}
    out = conv(x_dict, edge_index_dict)
    print("MultiAggr shape:", out['B'].shape)
except Exception as e:
    print("Error:", e)
