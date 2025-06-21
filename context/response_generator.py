# response_generator.py

from transformers import pipeline
import re
import torch
import numpy as np

# Load the GPT-2 model (ensure you have it downloaded or an internet connection)
generator = pipeline("text-generation", model="gpt2-medium")

def format_dialogue_history(history, max_turns=2):
    return "\n".join(
        f"{turn['speaker'].capitalize()}: {turn['utterance']}" for turn in history[-max_turns:]
    )

def clean_response(text):
    text = text.strip()
    text = re.sub(r"(I'm sorry[.!]\s*){2,}", "I'm sorry. ", text, flags=re.IGNORECASE)
    text = re.sub(r"(User|Assistant|System|Response):.*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\b(Given|Based on|Here's|The response is|A good response would be|My advice is|Response would be|Assistant:)\s*$", "", text, flags=re.IGNORECASE).strip()
    if text and not text.endswith(('.', '!', '?')):
        text += '.'
    return text

def generate_response(query, gnn_output, node_mapping, context_history, real_time_data):
    # node_mapping is a dictionary from PyG node indices to original NetworkX node names
    # `relevant_concepts` are the original query concepts that matched nodes in the graph
    # gnn_output is the 'x' tensor from your GATModel, representing learned node features/scores

    # Filter out None values and create a string of matched concepts
    matched_strs = [str(node_mapping[i]) for i in range(len(node_mapping)) if i in node_mapping and node_mapping[i] is not None]
    relevant_concepts_str = ", ".join(matched_strs) if matched_strs else "None"

    weather = real_time_data.get("weather", "No data")
    mood = real_time_data.get("user_mood", "No data")
    time_of_day = real_time_data.get("time_of_day", "No data")

    gnn_insight = ""
    if gnn_output is not None and isinstance(gnn_output, torch.Tensor) and gnn_output.numel() > 0:
        # --- CUSTOMIZE THIS SECTION BASED ON WHAT YOUR GNN ACTUALLY PREDICTS ---
        # gnn_output is the result of your GATModel's forward pass (the 'x' tensor).
        # Its meaning depends on your GNN's training objective.

        if relevant_concepts_str != "None":
            # Scenario 1: GNN's purpose is to contextualize or refine understanding of concepts
            # This is a good generic starting point.
            gnn_insight = f"The graph neural network analyzed common sense relationships involving concepts like {relevant_concepts_str}. "

            # --- Advanced GNN Interpretation Examples (UNCOMMENT & ADAPT IF APPLICABLE) ---
            # Scenario 2: If your GNN performs node classification (e.g., predicting if a node is an "action", "cause", "effect")
            # This assumes gnn_output is [num_nodes, num_classes] and you have a mapping
            # if gnn_output.dim() == 2 and gnn_output.size(1) > 1:
            #     # Assuming a hypothetical mapping of output classes to types
            #     node_type_labels = {0: "action", 1: "object", 2: "property", 3: "event"}
            #     # For simplicity, let's just interpret the first few matched nodes' predicted types
            #     interpreted_elements = []
            #     for i, node_name in enumerate(matched_strs[:3]): # Consider top 3 matched
            #         if i < gnn_output.size(0): # Check if node index exists in gnn_output
            #             predicted_class_idx = torch.argmax(gnn_output[i]).item()
            #             type_label = node_type_labels.get(predicted_class_idx, "unknown type")
            #             interpreted_elements.append(f"{node_name} (as {type_label})")
            #     if interpreted_elements:
            #         gnn_insight += f"It specifically highlighted elements such as: {'; '.join(interpreted_elements)}. "

            # Scenario 3: If your GNN is designed to rank nodes by relevance or predict a single outcome from the graph
            # e.g., if gnn_output is [1, num_actions] predicting the best action
            # Or if it's an importance score for each node [num_nodes, 1]
            # if gnn_output.dim() == 2 and gnn_output.size(1) == 1: # Node importance scores
            #     # Pick the node with the highest score as the most relevant from GNN's perspective
            #     most_important_idx = torch.argmax(gnn_output.squeeze()).item()
            #     if most_important_idx < len(node_mapping):
            #         most_important_node = node_mapping.get(most_important_idx)
            #         if most_important_node:
            #             gnn_insight += f"The most relevant common sense element identified was '{most_important_node}'. "
        else:
            # Fallback if no relevant concepts were matched but GNN still ran
            gnn_insight = "The model processed information from common sense graphs. "
    else:
        # Fallback if GNN did not produce output (e.g., no subgraph)
        gnn_insight = "No specific graph-based reasoning insight was available. "


    prompt = (
        "You are a highly intelligent and helpful AI assistant designed to provide practical, common-sense advice.\n"
        "Your goal is to give actionable and empathetic responses to user's situations, drawing on provided context and reasoning.\n"
        "Always respond directly to the user in a helpful tone. Do not ask questions or make demands.\n"
        "**Ensure your responses are always advice for the user, starting with phrases like 'You should...', 'It's recommended to...', or similar. Do not use 'I will' or imply that you are taking action on the user's behalf.**\n\n" # <--- ADDED THIS NEW INSTRUCTION
        "Here's an example of how you should respond:\n"
        "User: 'I spilled water on my laptop.'\n"
        "Context: time = morning, weather = rainy, mood = worried, concepts = water, laptop, electronics\n"
        "Response: 'You should immediately turn off your laptop, unplug it, and remove the battery if possible. Let it dry completely for at least 24-48 hours before attempting to turn it on again.'\n\n" # <--- MODIFIED THE EXAMPLE RESPONSE
        f"User: '{query}'\n"
        f"Context: time = {time_of_day}, weather = {weather}, mood = {mood}, concepts = {relevant_concepts_str}\n"
        f"Reasoning Insight: {gnn_insight}\n"
        "Response: "
    )

    outputs = generator(
        prompt,
        max_new_tokens=80,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        return_full_text=False,
    )

    generated_text = outputs[0]["generated_text"].strip()
    cleaned = clean_response(generated_text)

    # Fallback/hardcoded responses for specific queries if LLM output is poor
    if not cleaned or len(cleaned.split()) < 3 or cleaned.lower() == query.lower() or "give it to me" in cleaned.lower() or "watery stain" in cleaned.lower():
        if "hungry" in query.lower():
            return "You should find something to eat. Perhaps prepare a snack or a meal."
        elif "tired" in query.lower():
            return "It sounds like you need some rest. Consider taking a break or getting some sleep."
        elif "spilled" in query.lower() and "laptop" in query.lower():
             return "Immediately turn off your laptop, unplug it, and remove the battery if possible. Let it dry completely before turning it on."
        else:
            return "I recommend looking for a practical solution to your situation."

    return cleaned