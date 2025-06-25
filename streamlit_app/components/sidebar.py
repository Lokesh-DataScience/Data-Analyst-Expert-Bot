import streamlit as st
import uuid
import requests
def render_sidebar(get_recent_chat_titles, load_chat_history_from_backend):
    st.sidebar.header("🕓 Recent Chats")
    if st.sidebar.button("➕ New Chat", type="primary", use_container_width=True):
        session_id = st.session_state.get("session_id")
        chat_history = st.session_state.get("chat_history", [])
        if chat_history:
            requests.post(
                "http://localhost:8000/save-chat",
                json={"session_id": session_id, "chat_history": chat_history}
            )
        st.session_state["session_id"] = str(uuid.uuid4())
        st.session_state["chat_history"] = []
        st.session_state["last_uploaded_image_name"] = None
        st.rerun()
    st.sidebar.divider()
    session_titles = get_recent_chat_titles()
    for chat in session_titles.get("sessions", []):
        if st.sidebar.button(chat["title"][:40], key=chat["session_id"]):
            st.session_state["session_id"] = chat["session_id"]
            history_response = load_chat_history_from_backend(chat["session_id"])
            st.session_state["chat_history"] = history_response.get("chat_history", [])
            st.rerun()