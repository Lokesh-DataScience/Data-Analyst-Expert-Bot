import streamlit as st
import requests
from langchain_core.messages import AIMessage, HumanMessage
import uuid

# Ensure the session ID is set for tracking
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

# Define the API URL for the chat service
API_URL = "http://localhost:8000/chat"  # Update if deployed

# Set up the Streamlit page configuration
st.set_page_config(page_title="Analyst Bot", layout="wide")

# Initialize streamlit memory
st.session_state.setdefault("chat_history", [])

st.title("💬 Data Analyst Expert")

user_input = st.chat_input("Ask me anything about data analysis...")

if user_input:
    # Append user's message to chat history
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    # Convert history to plain text list (for serialization)
    history_serialized = [
        {"type": "human", "content": m.content} if isinstance(m, HumanMessage) # HumanMessage
        else {"type": "ai", "content": m.content} # AIMessage
        for m in st.session_state.chat_history # Convert messages to a serializable format
    ]

    try:
        # Send the request to the chat API
        with st.spinner("Thinking..."):
            # Make the API call with the user's input and chat history
            res = requests.post(API_URL, json={
                "question": user_input,
                "chat_history": history_serialized,
                "session_id": st.session_state["session_id"]
            })
        # Check if the request was successful
        res.raise_for_status()
        # Extract the AI's response
        answer = res.json().get("response", "⚠️ No answer returned.")
        st.session_state.chat_history.append(AIMessage(content=answer)) # Append AI's response to chat history

    except Exception as e:
        st.session_state.chat_history.append(AIMessage(content=f"❌ Error: {e}"))

# Display conversation
for msg in st.session_state.chat_history:
    with st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant"): # HumanMessage or AIMessage
        st.markdown(msg.content)
