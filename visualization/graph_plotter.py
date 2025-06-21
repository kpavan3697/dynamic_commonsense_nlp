#graph_plotter.py
import networkx as nx
import matplotlib.pyplot as plt

def visualize_attention(edge_index, attention_weights, node_labels):
    G = nx.DiGraph()
    for i in range(edge_index.shape[1]):
        src = edge_index[0][i]
        dst = edge_index[1][i]
        weight = attention_weights[i]
        G.add_edge(src, dst, weight=weight)

    pos = nx.spring_layout(G, seed=42)
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    norm_weights = [(w - min(weights)) / (max(weights) - min(weights) + 1e-6) for w in weights]

    nx.draw(G, pos, labels=node_labels, edge_color=norm_weights, edge_cmap=plt.cm.Blues,
            with_labels=True, node_size=600, font_size=10)
    edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title("GAT Attention Weights")
    plt.show()
