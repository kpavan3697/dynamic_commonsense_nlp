import difflib

def find_closest_node(graph_nodes, concept, cutoff=0.8):
    """
    Find close matches for a concept in graph nodes using difflib.
    Returns a list of closest matches above cutoff similarity.
    """
    return difflib.get_close_matches(concept, graph_nodes, n=3, cutoff=cutoff)

def normalize_and_match_concepts(graph, concepts):
    """
    For each concept, check if it exists exactly in the graph.
    If not, find close matches and return a dict with best matches.
    Returns a dict: {original_concept: matched_node_or_None}
    """
    graph_nodes_lower = {n.lower(): n for n in graph.nodes()} # Map lowercased node to original cased node
    concept_matches = {}

    for concept in concepts:
        concept_lower = concept.lower()
        if concept_lower in graph_nodes_lower:
            # Exact match
            matched_node = graph_nodes_lower[concept_lower]
            concept_matches[concept] = matched_node
        else:
            # Try fuzzy matching on the lowercased graph nodes
            # difflib.get_close_matches expects a list of possibilities, not a dict keys view
            close_matches_lower = find_closest_node(list(graph_nodes_lower.keys()), concept_lower)
            if close_matches_lower:
                # Pick best close match (first one) and map back to original casing
                best_match_lower = close_matches_lower[0]
                matched_node = graph_nodes_lower[best_match_lower]
                concept_matches[concept] = matched_node
            else:
                concept_matches[concept] = None

    return concept_matches