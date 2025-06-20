# train_gnn.py
import torch
import torch.nn as nn
from torch_geometric.data import Data
import networkx as nx
import os
import random

# Assume gnn_model.py and graph_builder.py are in the 'reasoning' directory
from reasoning.gnn_model import GATModel
from reasoning.graph_builder import nx_to_pyg_data 

# --- Dummy Data Generation (for demonstration of training process) ---
def generate_dummy_data(num_samples=100, max_nodes=20, max_edges_per_node=3):
    """Generates synthetic PyG Data objects and target persona scores."""
    dummy_dataset = []
    
    # Define some example persona targets for specific scenarios (high-level categories)
    # These are illustrative. In reality, you'd have real labeled data.
    persona_types = {
        "urgent_practical": [0.9, 0.3, 0.9, 0.5],  # Urgency, Emotional Distress, Practical Need, Empathy
        "emotional_support": [0.3, 0.9, 0.3, 0.9],
        "information_seeking": [0.2, 0.1, 0.8, 0.4],
        "general_inquiry": [0.1, 0.1, 0.5, 0.5],
        "boredom_activity": [0.2, 0.5, 0.9, 0.8],
    }

    all_node_names = [f"node_{i}" for i in range(max_nodes * 2)] # Larger pool of names
    
    for i in range(num_samples):
        num_nodes = random.randint(5, max_nodes)
        
        # Create a dummy NetworkX graph
        nx_graph = nx.Graph()
        nodes_for_graph = random.sample(all_node_names, num_nodes)
        nx_graph.add_nodes_from(nodes_for_graph)
        
        for u in nodes_for_graph:
            num_edges = random.randint(1, max_edges_per_node)
            possible_neighbors = [n for n in nodes_for_graph if n != u]
            if not possible_neighbors: continue

            for _ in range(num_edges):
                v = random.choice(possible_neighbors)
                nx_graph.add_edge(u, v, relation="hasA") # Dummy relation

        # Convert to PyG Data. Ensure feature_dim is consistent.
        max_possible_features = len(all_node_names) 
        pyg_data = nx_to_pyg_data(nx_graph, feature_dim=max_possible_features)

        # Assign a random persona type to each dummy sample
        target_persona_scores = random.choice(list(persona_types.values()))
        pyg_data.y = torch.tensor(target_persona_scores, dtype=torch.float)
        
        dummy_dataset.append(pyg_data)
    
    print(f"Generated {len(dummy_dataset)} dummy data samples.")
    return dummy_dataset, max_possible_features

# --- Main Training Function (Placeholder) ---
def train_model(model, dataset, epochs=10, learning_rate=0.01, model_save_path="models/trained_gnn_model.pth"):
    """
    Placeholder for the GNN training process.
    In a real scenario, this would involve a robust dataset, proper loss, and optimization.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss() 

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    print(f"Starting dummy training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train() # Set model to training mode
        total_loss = 0
        for data in dataset:
            # Ensure data has required attributes for GNN
            if data.x is None or data.x.numel() == 0 or data.edge_index is None or data.edge_index.numel() == 0:
                # print(f"Skipping malformed data sample: {data}")
                continue

            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = criterion(out.mean(dim=0), data.y) # Average node embeddings for graph-level prediction
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataset) if len(dataset) > 0 else 0
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    print("Dummy training complete. Saving dummy model weights.")
    # Save the state_dict to simulate a trained model
    torch.save(model.state_dict(), model_save_path)
    print(f"Dummy model weights saved to {model_save_path}")


if __name__ == "__main__":
    dummy_num_samples = 200
    dummy_max_nodes_per_graph = 15 
    total_possible_features = 50 

    dummy_dataset_pyg, total_feature_dim = generate_dummy_data(dummy_num_samples, dummy_max_nodes_per_graph)
    
    model = GATModel(input_dim=total_feature_dim, hidden_dim=8, output_dim=4)
    
    train_model(model, dummy_dataset_pyg, epochs=5, model_save_path="models/trained_gnn_model.pth")