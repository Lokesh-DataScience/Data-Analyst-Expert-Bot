import streamlit as st
import requests
from langchain_core.messages import AIMessage, HumanMessage
import uuid
import base64

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

uploaded_image = st.file_uploader("Upload an image (optional)", type=["jpg", "jpeg", "png"])
user_input = st.chat_input("Ask me anything about data analysis...")

def encode_image_file(file):
    return base64.b64encode(file.read()).decode('utf-8')

if user_input:
    # Append user's message to chat history
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    # Convert history to plain text list (for serialization)
    history_serialized = [
        {"type": "human", "content": m.content} if isinstance(m, HumanMessage) # HumanMessage
        else {"type": "ai", "content": m.content} # AIMessage
        for m in st.session_state.chat_history # Convert messages to a serializable format
    ]
    
    image_b64 = encode_image_file(uploaded_image) if uploaded_image else None
    
    try:
        # Send the request to the chat API
        with st.spinner("Thinking..."):
            # Make the API call with the user's input and chat history
            payload = {
                "question": user_input,
                "chat_history": history_serialized,
                "session_id": st.session_state["session_id"],
            }
            if image_b64:
                payload["image_base64"] = image_b64
                payload["image_type"] = uploaded_image.type
            res = requests.post(API_URL, json=payload)
        res.raise_for_status()
        answer = res.json().get("response", "⚠️ No answer returned.")
        st.session_state.chat_history.append(AIMessage(content=answer))

    except Exception as e:
        st.session_state.chat_history.append(AIMessage(content=f"❌ Error: {e}"))

# Display conversation
for msg in st.session_state.chat_history:
    with st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant"): # HumanMessage or AIMessage
        st.markdown(msg.content)
