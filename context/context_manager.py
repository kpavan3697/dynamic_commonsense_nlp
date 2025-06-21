class ContextManager:
    def __init__(self, max_history=10):
        # Keeps last `max_history` dialogue turns
        self.dialogue_history = []
        self.max_history = max_history
        
        # Store arbitrary real-time contextual info
        self.real_time_data = {}

    def add_user_utterance(self, utterance: str):
        self._add_to_history({"speaker": "user", "utterance": utterance})

    def add_system_response(self, response: str):
        self._add_to_history({"speaker": "system", "utterance": response})

    def _add_to_history(self, turn: dict):
        self.dialogue_history.append(turn)
        if len(self.dialogue_history) > self.max_history:
            self.dialogue_history.pop(0)

    def get_dialogue_history(self):
        return self.dialogue_history

    def update_real_time_data(self, key: str, value):
        self.real_time_data[key] = value

    def get_real_time_data(self, key: str):
        return self.real_time_data.get(key, None)

    def get_all_context(self): # Added for easier retrieval of all real-time data
        return self.real_time_data

    def clear_context(self):
        self.dialogue_history = []
        self.real_time_data = {}