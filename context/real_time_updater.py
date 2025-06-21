import os
from datetime import datetime

# REMOVED: import openai

# REMOVED: openai.api_key assignment lines


class ContextManager: # This class is a placeholder from your original file, actual one is in context_manager.py
    def __init__(self):
        self.context = {
            "weather": "unknown",
            "user_mood": "neutral",
            "time_of_day": "unknown"
        }
        self.history = [] # Added for storing past interactions if needed

    def add_user_utterance(self, utterance):
        self.history.append({"role": "user", "content": utterance, "timestamp": datetime.now().isoformat()})

    def add_system_response(self, response):
        self.history.append({"role": "system", "content": response, "timestamp": datetime.now().isoformat()})

    def update_real_time_data(self, key, value):
        self.context[key] = value

    def get_all_context(self):
        return self.context

    def get_history(self, num_entries=5):
        return self.history[-num_entries:] # Return last N entries


class RealTimeContextUpdater:
    def __init__(self, context_manager_instance): # Accept an instance of ContextManager
        self.context_manager = context_manager_instance

    def update_from_query(self, query: str): # This function is not currently used by main.py
        query_lower = query.lower()

        # Determine weather from query
        weather = "rainy" if "rain" in query_lower else "sunny"

        # Determine mood from keywords
        if any(w in query_lower for w in ["tired", "sleepy", "exhausted", "fatigued"]):
            mood = "tired"
        elif any(w in query_lower for w in ["happy", "joyful", "excited"]):
            mood = "happy"
        elif any(w in query_lower for w in ["sad", "unhappy", "depressed"]):
            mood = "sad"
        else:
            mood = "neutral"

        # Determine time of day
        hour = datetime.now().hour
        if 5 <= hour < 12:
            time_of_day = "morning"
        elif 12 <= hour < 17:
            time_of_day = "afternoon"
        else:
            time_of_day = "evening"

        # Update context
        self.context_manager.update_real_time_data("weather", weather)
        self.context_manager.update_real_time_data("user_mood", mood)
        self.context_manager.update_real_time_data("time_of_day", time_of_day)

    def update_manual(self, weather: str, mood: str, time_of_day: str):
        # Only update if a meaningful value is provided
        if weather and weather.lower() != "no data":
            self.context_manager.update_real_time_data("weather", weather)
        else:
            # If 'No data' selected, explicitly set to 'unknown' or leave previous if desired
            self.context_manager.update_real_time_data("weather", "unknown")

        if mood and mood.lower() != "no data":
            self.context_manager.update_real_time_data("user_mood", mood)
        else:
            self.context_manager.update_real_time_data("user_mood", "neutral")

        if time_of_day: # Time of day has no 'No data' option in app.py, always has a value
            self.context_manager.update_real_time_data("time_of_day", time_of_day)

    def get_context(self):
        return self.context_manager.get_all_context()


def build_prompt(user_query, context, gnn_insight=""): # Added gnn_insight parameter
    prompt_parts = [
        "You are a highly intelligent and helpful AI assistant designed to provide practical, common-sense advice.",
        "Your goal is to give actionable and empathetic responses to user's situations, drawing on provided context and reasoning.",
        "Always respond directly to the user in a helpful tone. Do not ask questions or make demands.",
        "Ensure your responses are always advice for the user, starting with phrases like 'You should...', 'It's recommended to...', or similar. Do not use 'I will' or imply that you are taking action on the user's behalf.",
        "Based on the following dynamic context and reasoning insights, provide the best common-sense advice:"
    ]
    prompt_parts.append(f"- Weather: {context.get('weather', 'unknown')}")
    prompt_parts.append(f"- User Mood: {context.get('user_mood', 'neutral')}")
    prompt_parts.append(f"- Time of Day: {context.get('time_of_day', 'unknown')}")

    if gnn_insight: # Only add if GNN provided insight
        prompt_parts.append(f"- Knowledge Graph Insight: {gnn_insight}")
    
    prompt_parts.append(f"\nThe user said: \"{user_query}\"")
    prompt_parts.append("Respond with common-sense advice:")

    return "\n".join(prompt_parts)


def get_model_response(prompt):
    """
    This function now returns a static placeholder message instead of calling the OpenAI API.
    """
    print("WARNING: OpenAI API key is not set, returning placeholder response.")
    
    # Extract parts from the prompt for clarity and to avoid f-string nesting issues
    user_query_extracted = "your situation" # Default if not found
    weather_extracted = "unknown"
    mood_extracted = "neutral"
    time_extracted = "unknown"
    gnn_insight_extracted = "No specific knowledge graph insight available for this query." # More generic default

    try:
        # Improved parsing for context data
        def extract_value_from_prompt(label, default="N/A"):
            start_idx = prompt.find(label)
            if start_idx != -1:
                value_start = start_idx + len(label)
                end_idx = prompt.find('\n', value_start) # Look for newline as delimiter
                if end_idx == -1: # If it's the last line in the context block
                    end_idx = prompt.find('\n', prompt.find('The user said: "')) # Stop before user query
                    if end_idx == -1: # if no user query, take rest of string
                         end_idx = len(prompt)
                
                value = prompt[value_start:end_idx].strip()
                # Basic cleanup for context values
                if value.lower() in ["no data", "unknown", "neutral"]:
                    return default
                return value
            return default

        user_query_start_marker = 'The user said: "'
        if user_query_start_marker in prompt:
            query_start = prompt.find(user_query_start_marker) + len(user_query_start_marker)
            query_end = prompt.find('"', query_start) # Find the closing quote
            if query_start != -1 and query_end != -1:
                user_query_extracted = prompt[query_start:query_end]
        
        weather_extracted = extract_value_from_prompt("- Weather: ")
        mood_extracted = extract_value_from_prompt("- User Mood: ")
        time_extracted = extract_value_from_prompt("- Time of Day: ")
        
        gnn_insight_marker = "- Knowledge Graph Insight: "
        if gnn_insight_marker in prompt:
            insight_start = prompt.find(gnn_insight_marker) + len(gnn_insight_marker)
            insight_end = prompt.find('\n', insight_start)
            if insight_end == -1: # If it's the very last line
                insight_end = len(prompt)
            gnn_insight_extracted = prompt[insight_start:insight_end].strip()
            if not gnn_insight_extracted: # If it was empty
                 gnn_insight_extracted = "No specific knowledge graph insight available for this query."


    except Exception as e: # Catch any parsing errors
        print(f"Warning: Could not fully parse prompt for placeholder response due to error: {e}. Some details might be missing.")


    # Construct the placeholder response using the extracted values
    # REMOVED: "This is a placeholder response because the OpenAI API key is not configured..."
    placeholder_response = (
        "Based on your input and the common sense knowledge the system processed: "
        f"\n- **Your Query**: \"{user_query_extracted}\""
        f"\n- **Context**: Weather - {weather_extracted}, User Mood - {mood_extracted}, Time of Day - {time_extracted}"
        f"\n- **Knowledge Graph Insight**: {gnn_insight_extracted}"
        "\n\n**Advice**: To address this situation, it's generally helpful to gather more information and consider all relevant factors. Depending on the specifics, you might want to look into common solutions for similar issues. Taking your current context (weather, mood, and time of day) into account can also help you make a more effective plan."
    )
    return placeholder_response


def main(): # This main function is for testing real_time_updater.py in isolation, not used by app.py
    user_query = "I spilled water on my laptop."

    # This ContextManager instance is separate from the one in main.py
    context_manager_test = ContextManager() 
    context_updater = RealTimeContextUpdater(context_manager_test)

    context_updater.update_from_query(user_query)
    current_context = context_updater.get_context()
    prompt = build_prompt(user_query, current_context)
    response = get_model_response(prompt)

    print("Context:", current_context)
    print("\nPrompt Sent:\n", prompt)
    print("\nModel Response:\n", response)


if __name__ == "__main__":
    main()