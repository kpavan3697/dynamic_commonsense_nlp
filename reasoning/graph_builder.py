import networkx as nx
import requests
import json
import collections # For deque
import time # For rate limiting
import torch
from torch_geometric.data import Data
import certifi # Ensure this is imported

# !!! TEMPORARY SSL WORKAROUND FOR DEMO PURPOSES ONLY !!!
# In a production environment, you should resolve the underlying SSL certificate issue
# by ensuring your system's certificates are up-to-date and correctly configured,
# or by explicitly providing a trusted CA bundle (e.g., using certifi.where()).
# For this demo, to bypass persistent SSL errors on some systems:
_requests_ca_bundle = False # Setting to False disables SSL certificate verification
# If you resolve your SSL issue and want to re-enable verification, change this to:
# _requests_ca_bundle = certifi.where()


def fetch_conceptnet_relations(start_node, depth=2, max_nodes=50, max_edges=100):
    """
    Fetches relations from ConceptNet API for a given start node,
    performing a breadth-first search up to a specified depth.
    
    Args:
        start_node (str): The initial concept to start fetching from.
        depth (int): The maximum number of hops (relations) to explore.
        max_nodes (int): Maximum number of nodes to include in the graph.
        max_edges (int): Maximum number of edges to include in the graph.

    Returns:
        networkx.DiGraph: A directed graph representing the ConceptNet relations.
    """
    graph = nx.DiGraph()
    visited_nodes = set()
    queue = collections.deque([(start_node.lower(), 0)]) # (node, current_depth)
    
    # Simple rate limiting
    last_request_time = 0
    min_interval = 0.1 # seconds between requests

    print(f"Fetching ConceptNet relations for '{start_node}' up to depth {depth}...")

    while queue and len(graph.nodes()) < max_nodes and len(graph.edges()) < max_edges:
        current_node, current_depth = queue.popleft()

        if current_node in visited_nodes:
            continue

        visited_nodes.add(current_node)
        graph.add_node(current_node) # Add node even if no edges found yet

        if current_depth >= depth:
            continue

        # ConceptNet API endpoint
        url = f"http://api.conceptnet.io/c/en/{current_node}"
        
        # Implement basic rate limiting
        current_time = time.time()
        if current_time - last_request_time < min_interval:
            time.sleep(min_interval - (current_time - last_request_time))
        last_request_time = time.time()

        try:
            # Using _requests_ca_bundle (which is set to False for temporary workaround)
            response = requests.get(url, params={"limit": 50}, verify=_requests_ca_bundle)
            response.raise_for_status() # Raise an exception for HTTP errors
            data = response.json()

            for edge in data.get('edges', []):
                if len(graph.edges()) >= max_edges:
                    break

                # Extract relation, start node, and end node
                relation = edge['rel']['label']
                start_label = edge['start']['label'].lower()
                end_label = edge['end']['label'].lower()

                # Ensure nodes are relevant and in English
                if not (edge['start']['language'] == 'en' and edge['end']['language'] == 'en'):
                    continue
                
                graph.add_edge(start_label, end_label, relation=relation, weight=edge['weight'])

                # Add neighbors to queue if not visited and within limits
                if start_label not in visited_nodes and len(graph.nodes()) < max_nodes:
                    queue.append((start_label, current_depth + 1))
                if end_label not in visited_nodes and len(graph.nodes()) < max_nodes:
                    queue.append((end_label, current_depth + 1))
        
        except requests.exceptions.RequestException as e:
            print(f"Error fetching ConceptNet data for '{current_node}': {e}")
            continue
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON for '{current_node}': {e}")
            continue
    
    print(f"Finished ConceptNet fetching. Graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
    return graph

def nx_to_pyg_data(nx_graph, feature_dim=None):
    """
    Converts a NetworkX graph to a PyTorch Geometric Data object.
    
    Args:
        nx_graph (networkx.Graph or networkx.DiGraph): The NetworkX graph.
        feature_dim (int, optional): The dimension for one-hot node features.
                                     If None, max node ID + 1 is used, but it's
                                     safer to provide a consistent vocab size.

    Returns:
        torch_geometric.data.Data: The PyG Data object.
    """
    # Create a mapping from NetworkX node names to PyG integer indices
    node_mapping = {node: i for i, node in enumerate(nx_graph.nodes())}
    reverse_node_mapping = {i: node for node, i in node_mapping.items()}

    # Edge index
    edge_indices = []
    for u, v in nx_graph.edges():
        edge_indices.append([node_mapping[u], node_mapping[v]])
        # If the graph is undirected, add the reverse edge too for GNNs that expect it
        # (even if the original NX graph is DiGraph, for a GNN like GATConv, treating it
        # as undirected or adding reverse edges can be beneficial for message passing)
        if not nx_graph.is_directed(): # Assuming your NX graphs are mostly undirected for GNN processing
            edge_indices.append([node_mapping[v], node_mapping[u]])

    if not edge_indices:
        # Handle case with no edges (e.g., a single node graph, or disconnected)
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

    # Node features (one-hot encoding based on node index)
    num_nodes = nx_graph.number_of_nodes()
    
    if num_nodes == 0:
        return Data(x=torch.empty((0, feature_dim if feature_dim is not None else 1)), 
                    edge_index=torch.empty((2,0), dtype=torch.long),
                    node_id_mapping=node_mapping)

    # If feature_dim is not provided, use the number of nodes in the current graph
    # This assumes each node is unique and features are one-hot encoded based on its ID.
    # For a general solution across many graphs, feature_dim should be the size of a global vocabulary.
    if feature_dim is None:
        feature_dim = num_nodes
    
    # Create one-hot features
    x = torch.zeros(num_nodes, feature_dim)
    for original_node, pyg_idx in node_mapping.items():
        if pyg_idx < feature_dim: # Ensure index is within feature_dim bounds
            x[pyg_idx, pyg_idx] = 1.0 # One-hot based on PyG index

    data = Data(x=x, edge_index=edge_index)
    data.node_id_mapping = reverse_node_mapping # Store mapping for interpretation

    return data