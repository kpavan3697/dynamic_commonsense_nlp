import networkx as nx

def build_knowledge_graph(triples, directed=True):
    G = nx.DiGraph() if directed else nx.Graph()
    for head, relation, tail in triples:
        G.add_node(head)
        G.add_node(tail)
        G.add_edge(head, tail, relation=relation)
    return G