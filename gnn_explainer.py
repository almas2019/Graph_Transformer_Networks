import pickle
import torch
from torch_geometric.nn import GNNExplainer
from model_fastgtn import FastGTNs
import argparse

# Load node features, edges, and labels
with open('/home/almas/projects/def-gregorys/almas/human_lymph_node/for_fastgtn/data/highly_var/node_features.pkl', 'rb') as f:
    node_features = pickle.load(f)

with open('/home/almas/projects/def-gregorys/almas/human_lymph_node/for_fastgtn/data/highly_var/edges.pkl', 'rb') as f:
    edges = pickle.load(f)

with open('/home/almas/projects/def-gregorys/almas/human_lymph_node/for_fastgtn/data/highly_var/labels.pkl', 'rb') as f:
    labels = pickle.load(f)

# This is with model state save dict instead of whole model which I changed in the current main--> change back to save state?

# Ensure the data is in the correct format
# Ensure the data is in the correct format
if sp.issparse(node_features):
    node_features = torch.tensor(node_features.todense(), dtype=torch.float)
else:
    node_features = torch.tensor(node_features, dtype=torch.float)
edges = torch.tensor(edges, dtype=torch.long)
labels = torch.tensor(labels, dtype=torch.long)
# Define a function to load the model
def load_model():
    args = argparse.Namespace( # save args and load instead of configuring manually (waiting for it to run)
        model='FastGTN', dataset='lymph_node_highly_var_200_lr001', epoch=2500,
        node_dim=64, num_channels=1, lr=0.001, weight_decay=0.001, num_layers=2,
        runs=1, channel_agg='mean', remove_self_loops=False, non_local=False,
        non_local_weight=0, beta=0, K=1, pre_train=False, num_FastGTN_layers=1,
        save_metrics=True, layer_split='train', data_path='../data/for_fastgtn/data/highly_var'
    )

    model = FastGTNs(num_edge_type=226, w_in=200, num_class=2, num_nodes=1000, args=args)
    model.load_state_dict(torch.load("/home/almas/projects/def-gregorys/almas/Graph_Transformer_Networks/data/lymph_node_highly_var_200_lr001/07_25_2024_07_08_FastGTN_lymph_node_highly_var_200_lr001_best.pt",map_location=torch.device('cpu')))  # Replace with the actual path
    model.eval()
    return model

# Load the model
model = load_model()


# Initialize GNNExplainer
explainer = GNNExplainer(model, epochs=200, return_type='prob')

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


}")
