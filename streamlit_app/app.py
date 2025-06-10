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

# Ensure the session ID is set for tracking
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

# Initialize session state for image upload attempts and last uploaded image
if "image_upload_attempts" not in st.session_state:
    st.session_state.image_upload_attempts = []

# Initialize session state for last uploaded image name
if "last_uploaded_image_name" not in st.session_state:
    st.session_state.last_uploaded_image_name = None

# Clear timestamps older than 6 hours
cutoff_time = datetime.now() - timedelta(hours=6) # 6-hour window for image uploads
st.session_state.image_upload_attempts = [
    ts for ts in st.session_state.image_upload_attempts if ts > cutoff_time # Only keep recent timestamps
]

image_upload_limit_reached = len(st.session_state.image_upload_attempts) >= 3 # Limit to 3 uploads per 6 hours

# Define the API URL for the chat service
API_URL = "http://localhost:8000/chat"  # Update if deployed

# Set up the Streamlit page configuration
st.set_page_config(page_title="Analyst Bot", layout="wide")

# Initialize streamlit memory
st.session_state.setdefault("chat_history", [])

st.title("💬 Data Analyst Expert")

uploaded_image = None
if image_upload_limit_reached:
    st.warning("⏳ You’ve reached the image upload limit (3 per 6 hours). Try again later.")
else:
    uploaded_image = st.file_uploader("Upload an image (optional)", type=["jpg", "jpeg", "png"])

    # Only count a new upload if it's a different file than before
    if uploaded_image and uploaded_image.name != st.session_state.last_uploaded_image_name: 
        st.session_state.image_upload_attempts.append(datetime.now()) # Record the upload time
        st.session_state.last_uploaded_image_name = uploaded_image.name # Update the last uploaded image name

uploads_left = max(0, 3 - len(st.session_state.image_upload_attempts)) # Calculate uploads left in the current 6-hour window
st.caption(f"🖼️ Uploads remaining in this 6-hour window: {uploads_left}")

user_input = st.chat_input("Ask me anything about data analysis...")

def encode_image_file(file):
    """Encodes an image file to base64."""
    return base64.b64encode(file.read()).decode('utf-8') 

if user_input:
    # Encode image only if newly uploaded
    image_b64 = encode_image_file(uploaded_image) if uploaded_image else None
    image_type = uploaded_image.type if uploaded_image else None

    # Append user message and image (if any) to chat history
    st.session_state.chat_history.append({
        "type": "human",
        "content": user_input,
        "image": image_b64,
        "image_type": image_type
    })

    # Serialize history for API (ignore images for now)
    history_serialized = [
        {"type": msg["type"], "content": msg["content"]} # Include image metadata
        #  if "image" in msg  # Only include image if it exists
        if isinstance(msg, dict)
        else {"type": "ai", "content": msg.content}
        for msg in st.session_state.chat_history # Convert AIMessage to dict
    ]

    try:
        # Send the request to the chat API
        with st.spinner("Thinking..."):
            # Prepare the payload for the API request
            payload = {
                "question": user_input,
                "chat_history": history_serialized,
                "session_id": st.session_state["session_id"],
            }
            # Include image data if available
            if image_b64:
                payload["image_base64"] = image_b64
                payload["image_type"] = image_type
            res = requests.post(API_URL, json=payload) # Send the request to the API
        res.raise_for_status() # Raise an error for bad responses
        answer = res.json().get("response", "⚠️ No answer returned.") # Get the response from the API
        st.session_state.chat_history.append(AIMessage(content=answer)) # Append AI response to chat history

    except Exception as e:
        st.session_state.chat_history.append(AIMessage(content=f"❌ Error: {e}")) # Append error message to chat history

# Display conversation
shown_images = set() # Set to track shown images by their hash

# Display chat history
for msg in st.session_state.chat_history:
    # Check if the message is a human message with an image
    if isinstance(msg, dict) and msg.get("type") == "human":
        with st.chat_message("user"): # Display user message
            st.markdown(msg["content"]) 
            
            if msg.get("image"):
                # Hash the image content to detect uniqueness
                image_hash = hashlib.md5(msg["image"].encode()).hexdigest() 

                if image_hash not in shown_images:
                    shown_images.add(image_hash) # Add to shown images to avoid duplicates

                    image_data = base64.b64decode(msg["image"]) # Decode the base64 image data
                    image = Image.open(io.BytesIO(image_data)) 
                    resized_image = image.resize((500, 300)) # Resize the image for better display
                    st.image(resized_image, caption="Uploaded Image") 
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"): # Display AI message
            st.markdown(msg.content) 