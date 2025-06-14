import streamlit as st
import requests
from langchain_core.messages import AIMessage, HumanMessage
import uuid
import base64
import time
from datetime import datetime, timedelta
from PIL import Image
import io
import hashlib

# Set page config first, before any other Streamlit commands
st.set_page_config(page_title="Analyst Bot", layout="wide")

# Initialize session state variables
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "image_upload_attempts" not in st.session_state:
    st.session_state.image_upload_attempts = []

if "last_uploaded_image_name" not in st.session_state:
    st.session_state.last_uploaded_image_name = None

@st.cache_data(ttl=120)  # cache for 2 minutes
def get_chat_history_for_session(session_id):
    try:
        res = requests.get(f"http://localhost:8000/recent-chats/{session_id}")
        res.raise_for_status()
        return res.json().get("chat_history", [])
    except Exception as e:
        st.warning(f"⚠️ Could not load recent chat history: {e}")
        return []

@st.cache_data(ttl=60)  # cache for 60 seconds
def get_recent_chat_titles():
    try:
        titles_res = requests.get("http://localhost:8000/recent-chat-titles")
        titles_res.raise_for_status()
        return titles_res.json().get("sessions", [])
    except Exception as e:
        st.sidebar.error(f"Failed to load recent chats: {e}")
        return []

def encode_image_file(file):
    """Encodes an image file to base64."""
    return base64.b64encode(file.read()).decode('utf-8')

def load_chat_history_from_backend(session_id):
    """Load and format chat history from backend"""
    full_history = get_chat_history_for_session(session_id)
    formatted_history = []
    
    for msg in full_history:
        if msg["type"] == "ai":
            formatted_history.append({
                "type": "ai",
                "content": msg["content"]
            })
        elif msg["type"] == "human":
            formatted_history.append({
                "type": "human",
                "content": msg["content"],
                "image": msg.get("file", {}).get("base64") if "file" in msg else msg.get("image"),
                "image_type": msg.get("file", {}).get("format") if "file" in msg else msg.get("image_type")
            })
    return formatted_history

# Load recent chat history from the backend on first load
if not st.session_state["chat_history"]:
    st.session_state["chat_history"] = load_chat_history_from_backend(st.session_state["session_id"])

# Clear old image upload timestamps (older than 6 hours)
cutoff_time = datetime.now() - timedelta(hours=6)
st.session_state.image_upload_attempts = [
    ts for ts in st.session_state.image_upload_attempts if ts > cutoff_time
]

# Check if image upload limit is reached
image_upload_limit_reached = len(st.session_state.image_upload_attempts) >= 3

# Sidebar for recent chats
with st.sidebar:
    st.header("🕓 Recent Chats")
    
    # New Chat button
    if st.button("➕ New Chat", type="primary", use_container_width=True):
        st.session_state["session_id"] = str(uuid.uuid4())
        st.session_state["chat_history"] = []
        st.session_state["last_uploaded_image_name"] = None
        st.rerun()
    
    st.divider()
    
    session_titles = get_recent_chat_titles()
    
    for chat in session_titles:
        if st.button(chat["title"][:40], key=chat["session_id"]):
            st.session_state["session_id"] = chat["session_id"]
            st.session_state["chat_history"] = load_chat_history_from_backend(chat["session_id"])
            st.rerun()

# Main app title
st.title("💬 Data Analyst Expert")

# File upload sections
uploaded_image = None
if image_upload_limit_reached:
    st.warning("⏳ You've reached the image upload limit (3 per 6 hours). Try again later.")
else:
    uploaded_image = st.file_uploader("Upload an image (optional)", type=["jpg", "jpeg", "png"])
    
    # Track new image uploads
    if uploaded_image and uploaded_image.name != st.session_state.last_uploaded_image_name:
        st.session_state.image_upload_attempts.append(datetime.now())
        st.session_state.last_uploaded_image_name = uploaded_image.name

uploads_left = max(0, 3 - len(st.session_state.image_upload_attempts))
st.caption(f"🖼️ Uploads remaining in this 6-hour window: {uploads_left}")

# CSV and PDF upload
uploaded_csv = st.file_uploader("Upload a CSV file (optional)", type=["csv"])
uploaded_pdf = st.file_uploader("Upload a PDF file (optional)", type=["pdf"])

# Prepare file data
csv_b64 = None
csv_filename = None
if uploaded_csv:
    csv_b64 = base64.b64encode(uploaded_csv.read()).decode('utf-8')
    csv_filename = uploaded_csv.name

pdf_b64 = None
pdf_filename = None
if uploaded_pdf:
    pdf_b64 = base64.b64encode(uploaded_pdf.read()).decode('utf-8')
    pdf_filename = uploaded_pdf.name

# Chat input
user_input = st.chat_input("Ask me anything about data analysis...")

# Process user input
if user_input:
    # Encode image if uploaded
    image_b64 = encode_image_file(uploaded_image) if uploaded_image else None
    image_type = uploaded_image.type if uploaded_image else None

    # Add user message to chat history
    st.session_state.chat_history.append({
        "type": "human",
        "content": user_input,
        "image": image_b64,
        "image_type": image_type
    })

    # Prepare chat history for API (serialize properly)
    history_serialized = []
    for msg in st.session_state.chat_history:
        if isinstance(msg, dict):
            history_serialized.append({
                "type": msg["type"],
                "content": msg["content"]
            })
        elif isinstance(msg, AIMessage):
            history_serialized.append({
                "type": "ai",
                "content": msg.content
            })
        elif isinstance(msg, HumanMessage):
            history_serialized.append({
                "type": "human",
                "content": msg.content
            })

    try:
        with st.spinner("Thinking..."):
            # Prepare payload
            payload = {
                "question": user_input,
                "session_id": st.session_state["session_id"],
                "chat_history": history_serialized
            }
            
            # Determine API endpoint based on uploaded files
            if csv_b64 and csv_filename:
                payload["csv_base64"] = csv_b64
                payload["csv_filename"] = csv_filename
                api_url = "http://localhost:8000/csv-upload"
            elif image_b64:
                payload["image_base64"] = image_b64
                payload["image_type"] = image_type
                api_url = "http://localhost:8000/image-upload"
            elif pdf_b64 and pdf_filename:
                payload["pdf_base64"] = pdf_b64
                payload["pdf_filename"] = pdf_filename
                api_url = "http://localhost:8000/pdf-upload"
            else:
                api_url = "http://localhost:8000/chat"

            # Send request to API
            res = requests.post(api_url, json=payload)
            res.raise_for_status()
            
            answer = res.json().get("response", "⚠️ No answer returned.")
            
            # Add AI response to chat history
            st.session_state.chat_history.append({
                "type": "ai",
                "content": answer
            })

    except requests.exceptions.RequestException as e:
        error_msg = f"❌ API Error: {e}"
        st.session_state.chat_history.append({
            "type": "ai",
            "content": error_msg
        })
    except Exception as e:
        error_msg = f"❌ Unexpected Error: {e}"
        st.session_state.chat_history.append({
            "type": "ai",
            "content": error_msg
        })

# Display chat history
shown_images = set()

for msg in st.session_state.chat_history:
    if isinstance(msg, AIMessage):
        st.chat_message("ai").markdown(msg.content)
    elif isinstance(msg, HumanMessage):
        st.chat_message("user").markdown(msg.content)
    elif isinstance(msg, dict):
        if msg.get("type") == "ai":
            st.chat_message("ai").markdown(msg["content"])
        elif msg.get("type") == "human":
            with st.chat_message("user"):
                st.markdown(msg["content"])
                
                # Display image if present
                if msg.get("image"):
                    try:
                        image_data = base64.b64decode(msg["image"])
                        image_hash = hashlib.md5(image_data).hexdigest()
                        
                        # Only show each unique image once
                        if image_hash not in shown_images:
                            shown_images.add(image_hash)
                            image = Image.open(io.BytesIO(image_data))
                            st.image(image, caption="Uploaded Image", use_container_width=True)
                    except Exception as e:
                        st.error(f"🖼️ Could not display image: {e}")