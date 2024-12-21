import pandas as pd
import networkx as nx
import numpy as np
import torch
from torch_geometric.utils import from_networkx
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from sklearn.metrics import classification_report

# Load Data
train_transactions = pd.read_csv('./output/filtered_train_transactions.csv')
test_transactions = pd.read_csv('./output/filtered_test_transactions.csv')

# Create mappings for customer_id and article_id
customer_ids = train_transactions['customer_id'].unique()
article_ids = train_transactions['article_id'].unique()
customer_map = {cid: i for i, cid in enumerate(customer_ids)}
article_map = {aid: i + len(customer_ids) for i, aid in enumerate(article_ids)}

# Map edges to unique indices
edges = train_transactions[['customer_id', 'article_id']].replace(
    {'customer_id': customer_map, 'article_id': article_map}
).values

# Map test edges for evaluation
test_edges = test_transactions[['customer_id', 'article_id']].replace(
    {'customer_id': customer_map, 'article_id': article_map}
).dropna().values

# Create Features
customer_features = np.eye(len(customer_ids))
article_features = np.eye(len(article_ids))
features = np.vstack([customer_features, article_features])

# Create Graph
G = nx.Graph()
G.add_edges_from(edges)
data = from_networkx(G)
data.x = torch.tensor(features, dtype=torch.float)

# Add labels and edge index
data.edge_index = torch.tensor(edges.T, dtype=torch.long)
data.y = torch.ones(data.edge_index.shape[1], dtype=torch.float)  # Labels for existing edges

# Define GCN
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Instantiate GCN
model = GCN(in_channels=data.x.shape[1], hidden_channels=16, out_channels=1)  # Out_channels = 1 for link prediction
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Prepare train/test split for edges
num_edges = data.edge_index.shape[1]
perm = torch.randperm(num_edges)
train_size = int(0.8 * num_edges)
train_edges = data.edge_index[:, perm[:train_size]]
test_edges = data.edge_index[:, perm[train_size:]]

train_labels = data.y[perm[:train_size]]
test_labels = data.y[perm[train_size:]]

# Training loop
for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, train_edges)
    loss = F.binary_cross_entropy_with_logits(out.view(-1), train_labels)
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Evaluate
model.eval()
with torch.no_grad():
    preds = model(data.x, test_edges).view(-1)
    preds = torch.sigmoid(preds).numpy()
    preds_binary = (preds > 0.5).astype(int)

# Generate ground truth and predictions for classification report
y_true = test_labels.numpy().astype(int)
y_pred = preds_binary

print("Classification Report:")
print(classification_report(y_true, y_pred))
