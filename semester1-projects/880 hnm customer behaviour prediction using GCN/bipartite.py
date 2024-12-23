import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# Data Preparation
def load_data():
    articles = pd.read_csv("../input/h-and-m-personalized-fashion-recommendations/articles.csv")
    customers = pd.read_csv("../input/h-and-m-personalized-fashion-recommendations/customers.csv")
    transactions = pd.read_csv("../input/h-and-m-personalized-fashion-recommendations/transactions_train.csv")

    # Filter data
    item_freq = transactions.groupby('article_id')['customer_id'].nunique()
    user_freq = transactions.groupby('customer_id')['article_id'].nunique()

    items = item_freq[item_freq >= 100].index
    users = user_freq[user_freq >= 100].index

    filtered_df = transactions[transactions['article_id'].isin(items) & transactions['customer_id'].isin(users)]

    freq = filtered_df.groupby(['customer_id', 'article_id']).size().reset_index(name='frequency')
    GraphTravel_HM = filtered_df.merge(freq, on=['customer_id', 'article_id'], how='left')
    GraphTravel_HM = GraphTravel_HM[GraphTravel_HM['frequency'] >= 10]

    unique_customer_ids = GraphTravel_HM['customer_id'].unique()
    customer_id_mapping = {id: i for i, id in enumerate(unique_customer_ids)}
    GraphTravel_HM['customer_id'] = GraphTravel_HM['customer_id'].map(customer_id_mapping)

    item_name_mapping = dict(zip(articles['article_id'], articles['prod_name']))

    return GraphTravel_HM, item_name_mapping, customers


# Cluster Customers Based on Properties
def cluster_customers(customers):
    # Select relevant features and preprocess
    customer_features = customers[['age', 'FN', 'Active']].fillna(0)
    scaler = StandardScaler()
    customer_features_scaled = scaler.fit_transform(customer_features)

    # Apply KMeans clustering
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    customers['cluster'] = kmeans.fit_predict(customer_features_scaled)

    print(f"Customer clusters:\n{customers['cluster'].value_counts()}")
    return customers, kmeans


# Generate Embeddings and Cluster Products
def generate_embeddings_and_cluster_products(GraphTravel_HM):
    G = nx.Graph()

    for _, row in GraphTravel_HM.iterrows():
        G.add_node(row['customer_id'], type='user')
        G.add_node(row['article_id'], type='item')
        G.add_edge(row['customer_id'], row['article_id'], weight=row['frequency'])

    x, edge_index, node_to_idx = prepare_graph_data(G)

    input_dim = x.size(1)
    hidden_dim = 128
    output_dim = 128

    model = GCN(input_dim, hidden_dim, output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(x, edge_index)

        # Loss: Compare output embeddings for connected nodes
        loss = 0
        for edge in edge_index.t():
            node_u, node_v = edge
            loss += F.mse_loss(out[node_u], out[node_v])

        loss /= edge_index.size(1)  # Normalize by the number of edges
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        final_embeddings = model(x, edge_index)

    # Convert embeddings to a dictionary
    embeddings = {node: final_embeddings[node_to_idx[node]].numpy() for node in G.nodes()}

    # Cluster products
    product_embeddings = [embeddings[node] for node in G.nodes if isinstance(node, int)]
    kmeans = KMeans(n_clusters=5, random_state=42)
    product_clusters = kmeans.fit_predict(product_embeddings)

    print("Product clustering completed.")
    return embeddings, product_clusters, kmeans


# Create Bipartite Graph
def create_bipartite_graph(customers, GraphTravel_HM, product_clusters):
    # Map products to clusters
    GraphTravel_HM['product_cluster'] = GraphTravel_HM['article_id'].map(
        lambda x: product_clusters[x] if x in product_clusters else -1
    )

    # Aggregate interactions between customer and product clusters
    cluster_edges = (
        GraphTravel_HM.groupby(['customer_id', 'product_cluster'])['frequency']
        .sum()
        .reset_index()
    )

    # Create bipartite graph
    B = nx.Graph()
    for _, row in cluster_edges.iterrows():
        B.add_edge(
            f"customer_cluster_{row['customer_id']}",
            f"product_cluster_{row['product_cluster']}",
            weight=row['frequency']
        )

    print(f"Bipartite graph created with {B.number_of_nodes()} nodes and {B.number_of_edges()} edges.")
    return B


# Main Execution
def main():
    GraphTravel_HM, item_name_mapping, customers = load_data()

    # Cluster customers based on properties
    customers, customer_kmeans = cluster_customers(customers)

    # Generate embeddings and cluster products
    embeddings, product_clusters, product_kmeans = generate_embeddings_and_cluster_products(GraphTravel_HM)

    # Create bipartite graph
    B = create_bipartite_graph(customers, GraphTravel_HM, product_clusters)

    # Visualize bipartite graph
    print(f"Bipartite graph: {B.edges(data=True)}")


if __name__ == "__main__":
    main()
