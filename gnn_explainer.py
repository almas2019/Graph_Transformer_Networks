import pickle
import torch
from torch_geometric.nn import GNNExplainer
from model_fastgtn import FastGTNs
# Load node features, edges, and labels
with open('node_features.pkl', 'rb') as f:
    node_features = pickle.load(f)

with open('edges.pkl', 'rb') as f:
    edges = pickle.load(f)

with open('labels.pkl', 'rb') as f:
    labels = pickle.load(f)

#

# Load the model
model = torch.load()

# Initialize GNNExplainer
explainer = GNNExplainer(model, epochs=200, return_type='both')

# Explain a single node prediction
node_idx = 0  # example node index
node_feat_mask, edge_mask = explainer.explain_node(node_idx, x=node_features, edge_index=edges)

# Get the most important features and edges
important_features = node_feat_mask.topk(10, largest=True)
important_edges = edge_mask.topk(10, largest=True)

# Get explanations for all nodes
for node_idx in range(node_features.shape[0]):
    node_feat_mask, edge_mask = explainer.explain_node(node_idx, x=node_features, edge_index=edges)
    important_features = node_feat_mask.topk(10, largest=True)
    important_edges = edge_mask.topk(10, largest=True)
    
    print(f"Node {node_idx} - Important Features: {important_features}")
    print(f"Node {node_idx} - Important Edges: {important_edges}")
