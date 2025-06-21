# visualizer.py
import networkx as nx
import matplotlib.pyplot as plt

def visualize_subgraph(graph, node_mapping, attention_scores=None, save_path="subgraph.png"):
    pos = nx.spring_layout(graph)
    plt.figure(figsize=(10, 7))

    node_colors = []
    for node in graph.nodes():
        if attention_scores and node in attention_scores:
            node_colors.append(attention_scores[node])
        else:
            node_colors.append(0.5)  # default gray

    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, cmap=plt.cm.Blues, node_size=800)
    nx.draw_networkx_edges(graph, pos, arrows=True)
    nx.draw_networkx_labels(graph, pos, font_size=10)
    
    edge_labels = nx.get_edge_attributes(graph, 'relation')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color='red')

    plt.title("Concept Subgraph with Attention Weights")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Graph saved to {save_path}")
