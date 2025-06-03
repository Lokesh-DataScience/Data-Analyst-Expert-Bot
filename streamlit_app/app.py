import streamlit as st
import requests
from langchain_core.messages import AIMessage, HumanMessage

API_URL = "http://localhost:8000/chat"  # Update if deployed

st.set_page_config(page_title="RAG Chatbot with Groq", layout="wide")

# Initialize memory
st.session_state.setdefault("chat_history", [])

st.title("💬 RAG Chatbot (Groq + FAISS + LangChain)")

user_input = st.chat_input("Ask me anything about the scraped data...")

if user_input:
    # Append user's message to chat history
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    # Convert history to plain text list (for serialization)
    history_serialized = [
        {"type": "human", "content": m.content} if isinstance(m, HumanMessage)
        else {"type": "ai", "content": m.content}
        for m in st.session_state.chat_history
    ]

    try:
        with st.spinner("Thinking..."):
            res = requests.post(API_URL, json={
                "question": user_input,
                "chat_history": history_serialized
            })

        res.raise_for_status()
        answer = res.json().get("response", "⚠️ No answer returned.")
        st.session_state.chat_history.append(AIMessage(content=answer))

    except Exception as e:
        st.session_state.chat_history.append(AIMessage(content=f"❌ Error: {e}"))

# Display conversation
for msg in st.session_state.chat_history:
    with st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant"):
        st.markdown(msg.content)
