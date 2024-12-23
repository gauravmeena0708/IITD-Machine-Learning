import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import torch
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import umap.umap_ as umap

def visualize_graph_structure(G):
    """
    Visualize the overall graph structure
    """
    plt.figure(figsize=(20, 20))
    
    # Separate user and item nodes
    users = [node for node, data in G.nodes(data=True) if data.get('type') == 'user']
    items = [node for node, data in G.nodes(data=True) if data.get('type') == 'item']
    
    # Custom layout
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    # Draw users in blue, items in red
    nx.draw_networkx_nodes(G, pos, nodelist=users, node_color='blue', 
                            node_size=20, alpha=0.3, label='Users')
    nx.draw_networkx_nodes(G, pos, nodelist=items, node_color='red', 
                            node_size=20, alpha=0.3, label='Items')
    
    # Draw edges with low alpha
    nx.draw_networkx_edges(G, pos, alpha=0.05)
    
    plt.title('H&M Customer-Item Interaction Graph')
    plt.legend()
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_embeddings(embeddings, GraphTravel_HM, title='Embedding Visualization'):
    """
    Visualize embeddings using UMAP and t-SNE
    """
    # Convert embeddings to numpy array
    embed_array = np.array(list(embeddings.values()))
    nodes = list(embeddings.keys())
    
    # Determine node types
    node_types = []
    for node in nodes:
        if node in set(GraphTravel_HM['customer_id_mapped']):
            node_types.append('user')
        else:
            node_types.append('item')
    
    # UMAP Visualization
    plt.figure(figsize=(20, 10))
    
    # UMAP subplot
    plt.subplot(121)
    reducer_umap = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine')
    embedding_umap = reducer_umap.fit_transform(embed_array)
    
    # Color mapping
    color_map = {'user': 'blue', 'item': 'red'}
    colors = [color_map[t] for t in node_types]
    
    scatter = plt.scatter(embedding_umap[:, 0], embedding_umap[:, 1], 
                          c=colors, alpha=0.7, s=30)
    plt.title(f'{title} - UMAP Visualization')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    
    # t-SNE Visualization
    plt.subplot(122)
    reducer_tsne = TSNE(n_components=2, random_state=42)
    embedding_tsne = reducer_tsne.fit_transform(embed_array)
    
    scatter = plt.scatter(embedding_tsne[:, 0], embedding_tsne[:, 1], 
                          c=colors, alpha=0.7, s=30)
    plt.title(f'{title} - t-SNE Visualization')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    
    plt.tight_layout()
    plt.show()

def cluster_analysis(embeddings, GraphTravel_HM, n_clusters=5):
    """
    Perform clustering analysis on embeddings
    """
    # Convert embeddings to numpy array
    embed_array = np.array(list(embeddings.values()))
    nodes = list(embeddings.keys())
    
    # Determine node types
    node_types = []
    for node in nodes:
        if node in set(GraphTravel_HM['customer_id_mapped']):
            node_types.append('user')
        else:
            node_types.append('item')
    
    # K-Means Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embed_array)
    
    # Visualization
    plt.figure(figsize=(20, 10))
    
    # UMAP for clustering visualization
    reducer_umap = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine')
    embedding_umap = reducer_umap.fit_transform(embed_array)
    
    # Create subplot for cluster visualization
    plt.subplot(121)
    scatter = plt.scatter(embedding_umap[:, 0], embedding_umap[:, 1], 
                          c=cluster_labels, cmap='viridis', alpha=0.7, s=30)
    plt.colorbar(scatter)
    plt.title('Clustering Visualization')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    
    # Cluster distribution
    plt.subplot(122)
    cluster_counts = pd.Series(cluster_labels).value_counts()
    cluster_type_counts = pd.crosstab(cluster_labels, node_types)
    
    cluster_type_counts.plot(kind='bar', stacked=True)
    plt.title('Cluster Composition')
    plt.xlabel('Cluster')
    plt.ylabel('Count')
    plt.legend(title='Node Type')
    
    plt.tight_layout()
    plt.show()
    
    return cluster_labels, cluster_type_counts

def interaction_frequency_analysis(GraphTravel_HM):
    """
    Analyze and visualize interaction frequencies
    """
    # User interaction frequency
    user_interactions = GraphTravel_HM.groupby('customer_id')['article_id'].count()
    
    # Item interaction frequency
    item_interactions = GraphTravel_HM.groupby('article_id')['customer_id'].count()
    
    # Visualization
    plt.figure(figsize=(20, 10))
    
    # User interactions
    plt.subplot(121)
    sns.histplot(user_interactions, kde=True)
    plt.title('User Interaction Frequency')
    plt.xlabel('Number of Interactions')
    plt.ylabel('Count of Users')
    
    # Item interactions
    plt.subplot(122)
    sns.histplot(item_interactions, kde=True)
    plt.title('Item Interaction Frequency')
    plt.xlabel('Number of Interactions')
    plt.ylabel('Count of Items')
    
    plt.tight_layout()
    plt.show()

def main():
    articles_path = "../input/h-and-m-personalized-fashion-recommendations/articles.csv"
    customers_path = "../input/h-and-m-personalized-fashion-recommendations/customers.csv"
    transactions_path = "../input/h-and-m-personalized-fashion-recommendations/transactions_train.csv"

    # Load and preprocess data
    GraphTravel_HM, item_name_mapping = load_and_preprocess_data(
        articles_path, customers_path, transactions_path
    )
    G = create_interaction_graph(GraphTravel_HM)
    
    # Visualize graph structure
    visualize_graph_structure(G)
    
    # Generate embeddings
    input_dim = len(G.nodes())
    hidden_dim = 128
    output_dim = 128
    embeddings = generate_embeddings(G, input_dim, hidden_dim, output_dim)
    
    # Visualize embeddings
    visualize_embeddings(embeddings, GraphTravel_HM)
    
    # Cluster analysis
    cluster_labels, cluster_type_counts = cluster_analysis(embeddings, GraphTravel_HM)
    
    # Interaction frequency analysis
    interaction_frequency_analysis(GraphTravel_HM)
    
    # Print cluster composition
    print("Cluster Composition:")
    print(cluster_type_counts)

if __name__ == "__main__":
    main()