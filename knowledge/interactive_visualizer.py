#interactive_visualizer.py
import plotly.graph_objects as go
import networkx as nx

def visualize_interactive_graph(G, center_node=None, num_hops=1, title="Interactive Graph"):
    if center_node and center_node in G:
        nodes = set([center_node])
        for _ in range(num_hops):
            neighbors = set()
            for node in nodes:
                neighbors.update(G.neighbors(node))
            nodes.update(neighbors)
        subgraph = G.subgraph(nodes)
    else:
        subgraph = G.subgraph(list(G.nodes)[:50])

    pos = nx.spring_layout(subgraph, seed=42)

    edge_x = []
    edge_y = []
    for edge in subgraph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    node_x = []
    node_y = []
    node_text = []
    for node in subgraph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(str(node))

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        text=node_text,
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            showscale=False,
            color='lightblue',
            size=20,
            line=dict(width=2)
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=title,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, zeroline=False)
                    ))
    fig.show()
