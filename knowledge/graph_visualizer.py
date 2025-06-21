import matplotlib.pyplot as plt
import networkx as nx

def visualize_subgraph(G, center_node=None, num_hops=1, title="Graph"):
    if center_node and center_node in G:
        nodes = set([center_node])
        for _ in range(num_hops):
            neighbors = set()
            for node in nodes:
                neighbors.update(G.neighbors(node))
            nodes.update(neighbors)
        subgraph = G.subgraph(nodes)
    else:
        subgraph = G.subgraph(list(G.nodes)[:50])  # default small graph

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(subgraph, seed=42)
    nx.draw(subgraph, pos, with_labels=True, node_size=500, node_color='lightblue', font_size=8, arrows=True)
    edge_labels = nx.get_edge_attributes(subgraph, 'relation')
    nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, font_size=7)
    plt.title(title)
    plt.show()