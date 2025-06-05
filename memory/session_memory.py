from langchain_core.chat_history import InMemoryChatMessageHistory

_memory_store = {}

def get_memory(session_id: str) -> InMemoryChatMessageHistory:
    """
    Returns the InMemoryChatMessageHistory for a given session.
    If not exists, initializes a new one.
    """
    # Check if the session ID exists in the memory store
    if session_id not in _memory_store:
        _memory_store[session_id] = InMemoryChatMessageHistory() # Create a new memory store for the session
    return _memory_store[session_id] # Return the existing memory store for the session
