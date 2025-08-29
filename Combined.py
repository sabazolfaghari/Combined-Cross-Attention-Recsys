import os
import json
import time
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from datetime import datetime
from scipy.stats import rankdata
from collections import defaultdict
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, random_split

# Step 1: Load and preprocess the data
def load_from_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Function to save data to JSON
def save_to_json(data, output_path):
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)

# Ensure embeddings align by userId and itemId
def align_embeddings(sim_embeddings, temp_embeddings, actual_ratings):
    sim_dict = {(entry['userId'], entry['itemId']): entry['embedding'] for entry in sim_embeddings}
    temp_dict = {(entry['userId'], entry['itemId']): entry['embedding'] for entry in temp_embeddings}
    ratings_dict = {(entry['userId'], entry['itemId']): entry['rating'] for entry in actual_ratings}
    aligned_data = []
    for key in sim_dict.keys():
        if key in temp_dict:
            aligned_data.append({
                'userId': key[0],
                'itemId': key[1],
                'similarity_embedding': sim_dict[key],
                'temporal_embedding': temp_dict[key],
                'actual_rating': ratings_dict[key]
            })
    return aligned_data

# Step 2: Define the Dataset and Cross-Attention Mechanism
class RecommenderDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        return (
            torch.tensor(entry['similarity_embedding'], dtype=torch.float32),
            torch.tensor(entry['temporal_embedding'], dtype=torch.float32),
            torch.tensor(entry['actual_rating'], dtype=torch.float32),
            torch.tensor(entry['userId'], dtype=torch.int64),
            torch.tensor(entry['itemId'], dtype=torch.int64)
        )

# Cross-Attention with bias factor
class CrossAttention(nn.Module):
    def __init__(self, embed_dim, bias_factor=0.5):
        super(CrossAttention, self).__init__()
        self.query_layer = nn.Linear(embed_dim, embed_dim)
        self.key_layer = nn.Linear(embed_dim, embed_dim)
        self.value_layer = nn.Linear(embed_dim, embed_dim)
        self.scale = torch.sqrt(torch.tensor(embed_dim, dtype=torch.float32))
        self.bias_factor = bias_factor

    def forward(self, temporal, similarity):
        # Generate Query, Key, and Value
        query = self.query_layer(temporal)
        key = self.key_layer(similarity)
        value = self.value_layer(similarity)

        # Scaled Dot-Product Attention
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Weighted Sum of Values
        attended_similarity = torch.matmul(attention_weights, value)

        # Combine Temporal and Similarity Embeddings
        combined_embeddings = self.bias_factor * temporal + (1 - self.bias_factor) * attended_similarity

        return combined_embeddings

# MLP for Rating Prediction
class RatingPredictor(nn.Module):
    def __init__(self, embedding_dim):
        super(RatingPredictor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(64, 1)
        )

    def forward(self, embedding):
        return self.fc(embedding)

# Metric functions
def precision_at_k(recommended_items, relevant_items, k):
    recommended_at_k = set(recommended_items[:k])
    relevant_items = set(relevant_items)
    return len(recommended_at_k & relevant_items) / k

def recall_at_k(recommended_items, relevant_items, k):
    recommended_at_k = set(recommended_items[:k])
    relevant_items = set(relevant_items)
    return len(recommended_at_k & relevant_items) / len(relevant_items) if relevant_items else 0

def ndcg_at_k(recommended_items, relevant_items, k):
    dcg = 0
    idcg = 0
    relevant_items_set = set(relevant_items)
    for i, item in enumerate(recommended_items[:k]):
        if item in relevant_items_set:
            dcg += 1 / np.log2(i + 2)
    for i in range(min(len(relevant_items), k)):
        idcg += 1 / np.log2(i + 2)
    return dcg / idcg if idcg > 0 else 0

def hit_rate_at_k(recommended_items, relevant_items, k):
    recommended_at_k = set(recommended_items[:k])
    relevant_items = set(relevant_items)
    return 1 if recommended_at_k & relevant_items else 0

def calculate_coverage(recommended_items, total_items):
    unique_recommended = set(recommended_items)
    return len(unique_recommended) / total_items

# Training function
def train_model(train_loader, cross_attention, rating_predictor, optimizer, criterion, epochs):
    cross_attention.train()
    rating_predictor.train()

    for epoch in range(epochs):
        total_loss = 0
        for similarity, temporal, rating, _, _ in train_loader:
            optimizer.zero_grad()
            combined_embeddings = cross_attention(temporal, similarity)
            predictions = rating_predictor(combined_embeddings)
            loss = criterion(predictions.squeeze(), rating)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")  # Print loss per epoch

# Testing function
def test_model(test_loader, cross_attention, rating_predictor, k):
    cross_attention.eval()
    rating_predictor.eval()
    all_ratings, all_predictions, test_indices = [], [], []
    user_to_recommended, user_to_relevant = defaultdict(list), defaultdict(list)

    with torch.no_grad():
        for similarity, temporal, ratings, user_ids, item_ids in test_loader:
            combined_embeddings = cross_attention(temporal, similarity)
            predictions = rating_predictor(combined_embeddings).squeeze()
            all_ratings.extend(ratings.tolist())
            all_predictions.extend(predictions.tolist())
            test_indices.extend(zip(user_ids.tolist(), item_ids.tolist()))

            # Group predictions and relevance by user
            for user, item, pred, true_rating in zip(user_ids.tolist(), item_ids.tolist(), predictions.tolist(), ratings.tolist()):
                user_to_recommended[user].append((item, pred))
                if true_rating >= 3:  # Define relevance
                    user_to_relevant[user].append(item)

    # Evaluate per user
    precision, recall, ndcg, hit_rate = 0, 0, 0, 0
    for user in user_to_recommended:
        recommended = sorted(user_to_recommended[user], key=lambda x: -x[1])[:k]
        recommended_items = [item for item, _ in recommended]
        relevant_items = user_to_relevant[user]

        precision += precision_at_k(recommended_items, relevant_items, k)
        recall += recall_at_k(recommended_items, relevant_items, k)
        ndcg += ndcg_at_k(recommended_items, relevant_items, k)
        hit_rate += hit_rate_at_k(recommended_items, relevant_items, k)

    # Average across users
    num_users = len(user_to_recommended)
    precision /= num_users
    recall /= num_users
    ndcg /= num_users
    hit_rate /= num_users

    # Coverage
    unique_recommended = set(item for user_items in user_to_recommended.values() for item, _ in user_items)
    coverage = len(unique_recommended) / len(test_indices)

    return all_ratings, all_predictions, precision, recall, ndcg, hit_rate, coverage

def main():
    # Parameters
    EPOCH = 10
    METRIC_K = 10
    BATCH_SIZE = 64
    L2_LAMBDA = 0.001
    EMBEDDING_DIM = 32
    LEARNING_RATE = 0.0005

    start_time = time.time()
    start_time_readable = datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
    print(f"Execution started at: {start_time_readable}")

    aligned_train_file = 'C://Users//Saba//Documents//UNIVERSITY//Thesis//Thesis//Final Combined Code and Files//FINAL//Saved Files//aligned_train3.json'
    aligned_test_file = 'C://Users//Saba//Documents//UNIVERSITY//Thesis//Thesis//Final Combined Code and Files//FINAL//Saved Files//aligned_test3.json'

    if os.path.exists(aligned_train_file) and os.path.exists(aligned_test_file):
        print(f"Aligned train and test data found. Loading from files.")
        train_data = load_from_json(aligned_train_file)
        test_data = load_from_json(aligned_test_file)
    else:
        print("Aligned files not found. Creating train and test alignment...")

        # Load train and test embeddings
        similarity_train = load_from_json("C://Users//Saba//Documents//UNIVERSITY//Thesis//Thesis//Movie Similarity Code and Files//FINAL//Saved Files//train_embeddings.json")
        similarity_test = load_from_json("C://Users//Saba//Documents//UNIVERSITY//Thesis//Thesis//Movie Similarity Code and Files//FINAL//Saved Files//test_embeddings.json")
        temporal_train = load_from_json("C://Users//Saba//Documents//UNIVERSITY//Thesis//Thesis//Temporal Code and Files//FINAL//Saved Files//train_embeddings.json")
        temporal_test = load_from_json("C://Users//Saba//Documents//UNIVERSITY//Thesis//Thesis//Temporal Code and Files//FINAL//Saved Files//test_embeddings.json")
        actual_ratings_train = load_from_json("C://Users//Saba//Documents//UNIVERSITY//Thesis//Thesis//Final Combined Code and Files//FINAL//Saved Files//actual_ratings_train.json")
        actual_ratings_test = load_from_json("C://Users//Saba//Documents//UNIVERSITY//Thesis//Thesis//Final Combined Code and Files//FINAL//Saved Files//actual_ratings_test.json")
        
        # Align data
        train_data = align_embeddings(similarity_train, temporal_train, actual_ratings_train)
        test_data = align_embeddings(similarity_test, temporal_test, actual_ratings_test)
        
        # Save aligned data for faster reuse
        save_to_json(train_data, aligned_train_file)
        save_to_json(test_data, aligned_test_file)
        print(f"Aligned train and test data saved successfully.")
    
    # Create datasets and DataLoaders
    train_dataset = RecommenderDataset(train_data)
    test_dataset = RecommenderDataset(test_data)
    train_loader = DataLoader(train_dataset, BATCH_SIZE)
    test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False)

    # Model, optimizer, and loss
    cross_attention = CrossAttention(embed_dim=EMBEDDING_DIM)
    rating_predictor = RatingPredictor(EMBEDDING_DIM)
    optimizer = optim.Adam(
        list(cross_attention.parameters()) + list(rating_predictor.parameters()),
        lr=LEARNING_RATE,
        weight_decay=L2_LAMBDA
    )
    criterion = nn.MSELoss()

    # Training
    train_loss = train_model(train_loader, cross_attention, rating_predictor, optimizer, criterion, EPOCH)

    # Testing and metrics
    all_ratings, all_predictions, precision, recall, ndcg, hit_rate, coverage = test_model(
        test_loader, cross_attention, rating_predictor, METRIC_K
    )

    # Calculate overall regression metrics
    rmse = root_mean_squared_error(all_ratings, all_predictions)
    mae = mean_absolute_error(all_ratings, all_predictions)

    print(f"Metrics @K={METRIC_K}:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"Precision@{METRIC_K}: {precision:.4f}")
    print(f"Recall@{METRIC_K}: {recall:.4f}")
    print(f"NDCG@{METRIC_K}: {ndcg:.4f}")
    print(f"Hit Rate@{METRIC_K}: {hit_rate:.4f}")
    print(f"Coverage: {coverage:.4f}")
    
    end_time = time.time()
    end_time_readable = datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')
    execution_time = end_time - start_time
    minutes, seconds = divmod(execution_time, 60)

    print(f"Execution started at: {start_time_readable}")
    print(f"Execution ended at: {end_time_readable}")
    print(f"Total execution time: {int(minutes)} min {int(seconds)} sec")

if __name__ == '__main__':
    main()

