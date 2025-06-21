import spacy
import networkx as nx
import re


# Load once globally
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spacy model 'en_core_web_sm'...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


def extract_key_concepts(query):
    doc = nlp(query)
    concepts = set()

    # Include nouns, proper nouns, and named entities
    for token in doc:
        if token.pos_ in ['NOUN', 'PROPN'] and not token.is_stop:
            concepts.add(token.lemma_.lower())
    for ent in doc.ents:
        concepts.add(ent.text.lower())

    return concepts

def retrieve_subgraph(graph, concepts, k=2): # This function is not used by retrieve_combined_subgraph
    nodes_to_include = set()
    for concept in concepts:
        normalized_node = normalize_conceptnet_node(concept)
        matched_nodes = [n for n in graph.nodes if n.lower() == normalized_node]
        if matched_nodes:
            for node in matched_nodes:
                nodes_to_include.add(node)
                neighbors = nx.single_source_shortest_path_length(graph, node, cutoff=k).keys()
                nodes_to_include.update(neighbors)
        else:
            print(f"No match found for concept '{concept}' as '{normalized_node}'")
    return graph.subgraph(nodes_to_include).copy()


def retrieve_combined_subgraph(conceptnet_graph, atomic_graph, input_data, k=2, is_concept_list=False):
    """
    Retrieves a combined subgraph from ConceptNet and ATOMIC based on a query or matched concepts.
    If is_concept_list is True, input_data is treated as a list of matched concept nodes.
    """
    # If not a list of concepts, extract them first (this path is not currently used by main.py)
    if not is_concept_list:
        concepts = extract_key_concepts(input_data)
        from utils.graph_utils import normalize_and_match_concepts # Import here to avoid circular dependency
        matched_concepts = normalize_and_match_concepts(conceptnet_graph, concepts).values()
    else:
        # input_data is already the list of matched concept node names
        matched_concepts = input_data

    subgraph = nx.Graph() # Use a generic graph for combining, assuming edges might be bidirectional conceptually

    # Add nodes and edges from ConceptNet
    for concept in matched_concepts:
        if concept in conceptnet_graph:
            # Get neighbors within k hops
            nodes_in_range = nx.single_source_shortest_path_length(conceptnet_graph, concept, cutoff=k).keys()
            # Add nodes and their attributes
            for node in nodes_in_range:
                if node in conceptnet_graph: # Check if node still exists after potential sub-selection
                    subgraph.add_node(node, **conceptnet_graph.nodes[node]) # Add node with its attributes
            # Add edges and their attributes
            for u, v, data in conceptnet_graph.edges(nodes_in_range, data=True):
                 if u in nodes_in_range and v in nodes_in_range:
                     subgraph.add_edge(u, v, **data)

    # Add nodes and edges from ATOMIC (can merge into the same graph)
    for concept in matched_concepts:
        if concept in atomic_graph: # Check if concept exists in ATOMIC
            nodes_in_range = nx.single_source_shortest_path_length(atomic_graph, concept, cutoff=k).keys()
            for node in nodes_in_range:
                if node in atomic_graph:
                    subgraph.add_node(node, **atomic_graph.nodes[node])
            for u, v, data in atomic_graph.edges(nodes_in_range, data=True):
                if u in nodes_in_range and v in nodes_in_range:
                    subgraph.add_edge(u, v, **data)

    return subgraph


def normalize_conceptnet_node(concept):
    # No prefix, just return the raw concept string
    return concept.lower()  # Optional: lowercase for safer matching


def extract_concepts(text): # This is the function called by main.py
    # Basic keyword/concept extraction using spacy and stopword filtering
    doc = nlp(text)
    concepts = set()

    # Include nouns, proper nouns, and named entities
    for token in doc:
        if token.pos_ in ['NOUN', 'PROPN'] and not token.is_stop and token.is_alpha and len(token) > 2:
            concepts.add(token.lemma_.lower())
    for ent in doc.ents:
        if ent.label_ not in ["CARDINAL", "ORDINAL", "DATE", "TIME", "MONEY", "QUANTITY", "PERCENT"]: # Filter out numerical/temporal entities
            concepts.add(ent.text.lower())
    return concepts