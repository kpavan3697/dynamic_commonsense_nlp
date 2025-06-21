import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class GATModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GATModel, self).__init__()
        
        # Define the number of attention heads. A common value for GAT is 8.
        heads = 8 
        
        self.gat1 = GATConv(input_dim, hidden_dim, heads=heads)
        # The output of GATConv with multiple heads is usually hidden_dim * heads
        self.lin = nn.Linear(hidden_dim * heads, output_dim)

    def forward(self, x, edge_index, return_attention_weights=False):
        # Apply GAT layer
        
        # Ensure x is float
        x = x.float()

        if return_attention_weights:
            x, (edge_index_attn, alpha) = self.gat1(x, edge_index, return_attention_weights=True)
        else:
            x = self.gat1(x, edge_index)

        x = torch.relu(x) # Apply activation after GATConv
        x = self.lin(x)    # Final linear layer

        if return_attention_weights:
            return x, (edge_index_attn, alpha)
        return x