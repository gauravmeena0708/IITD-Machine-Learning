import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.metrics.pairwise import cosine_similarity


# Data Loading and Preprocessing
def load_and_preprocess_data(articles_path, transactions_path):
    articles = pd.read_csv(articles_path)
    transactions = pd.read_csv(transactions_path)

    # Convert dates to datetime
    transactions['t_dat'] = pd.to_datetime(transactions['t_dat'])

    # Split data into train and test based on date
    last_month = transactions['t_dat'].max().replace(day=1)
    train_data = transactions[transactions['t_dat'] < last_month]
    test_data = transactions[transactions['t_dat'] >= last_month]

    # Filter items and users with sufficient interactions in the training set
    item_freq = train_data.groupby('article_id')['customer_id'].nunique()
    user_freq = train_data.groupby('customer_id')['article_id'].nunique()

    items = item_freq[item_freq >= 100].index
    users = user_freq[user_freq >= 100].index

    train_data = train_data[
        train_data['article_id'].isin(items) & 
        train_data['customer_id'].isin(users)
    ]

    # Calculate interaction frequency for train data
    freq = train_data.groupby(['customer_id', 'article_id']).size().reset_index(name='frequency')
    GraphTravel_HM = train_data.merge(freq, on=['customer_id', 'article_id'], how='left')
    GraphTravel_HM = GraphTravel_HM[GraphTravel_HM['frequency'] >= 10]

    # Create mappings
    unique_customer_ids = GraphTravel_HM['customer_id'].unique()
    customer_id_mapping = {id: i for i, id in enumerate(unique_customer_ids)}
    GraphTravel_HM['customer_id_mapped'] = GraphTravel_HM['customer_id'].map(customer_id_mapping)

    # Create item name mapping
    item_name_mapping = dict(zip(articles['article_id'], articles['prod_name']))

    return GraphTravel_HM, test_data, item_name_mapping


# Create Interaction Graph
def create_interaction_graph(GraphTravel_HM):
    G = nx.Graph()
    for _, row in GraphTravel_HM.iterrows():
        G.add_node(row['customer_id_mapped'], type='user')
        G.add_node(row['article_id'], type='item')
        G.add_edge(row['customer_id_mapped'], row['article_id'], weight=row['frequency'])
    return G


# Recommendation Evaluation
def evaluate_recommendations(test_data, embeddings, item_name_mapping, top_k=25):
    # Map customer_id to the same mapping used in training
    test_data['customer_id_mapped'] = test_data['customer_id'].map(customer_id_mapping)

    total_correct = 0
    total_users = 0

    for customer_id, group in test_data.groupby('customer_id_mapped'):
        if customer_id not in embeddings:
            continue
        user_embedding = embeddings[customer_id]
        purchased_items = group['article_id'].unique()

        # Rank items by similarity
        similarities = []
        for item_id in set(test_data['article_id']):
            if item_id in embeddings:
                item_embedding = embeddings[item_id]
                similarity = cosine_similarity([user_embedding], [item_embedding])[0][0]
                similarities.append((item_id, similarity))

        # Get top K recommendations
        recommendations = [item for item, _ in sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]]

        # Calculate correct predictions
        correct_predictions = len(set(recommendations) & set(purchased_items))
        total_correct += correct_predictions
        total_users += 1

    # Calculate average precision
    avg_precision = total_correct / (total_users * top_k)
    print(f"Average Precision: {avg_precision:.4f}")
    return avg_precision


# Main Execution
def main():
    articles_path = "../input/h-and-m-personalized-fashion-recommendations/articles.csv"
    transactions_path = "../input/h-and-m-personalized-fashion-recommendations/transactions_train.csv"

    # Load and preprocess data
    GraphTravel_HM, test_data, item_name_mapping = load_and_preprocess_data(
        articles_path, transactions_path
    )

    # Create interaction graph
    G = create_interaction_graph(GraphTravel_HM)

    # Generate embeddings
    input_dim = len(G.nodes())
    hidden_dim = 128
    output_dim = 128
    embeddings = generate_embeddings(G, input_dim, hidden_dim, output_dim)

    # Evaluate recommendations
    evaluate_recommendations(test_data, embeddings, item_name_mapping, top_k=25)


if __name__ == "__main__":
    main()
