import streamlit as st
import base64
import io
import hashlib
from PIL import Image
from langchain_core.messages import AIMessage, HumanMessage
import requests

def render_chat_interface(
    user_input,
    uploaded_image,
    uploaded_csv,
    uploaded_pdf,
    encode_image_file,
    load_chat_history_from_backend,
    session_id,
    chat_history
):
    # Prepare file data
    csv_b64 = csv_filename = None
    if uploaded_csv:
        uploaded_csv.seek(0)
        csv_b64 = base64.b64encode(uploaded_csv.read()).decode('utf-8')
        csv_filename = uploaded_csv.name

    pdf_b64 = pdf_filename = None
    if uploaded_pdf:
        uploaded_pdf.seek(0)
        pdf_b64 = base64.b64encode(uploaded_pdf.read()).decode('utf-8')
        pdf_filename = uploaded_pdf.name

    image_b64 = image_type = None
    if uploaded_image:
        uploaded_image.seek(0)
        image_b64 = encode_image_file(uploaded_image)
        image_type = uploaded_image.type

    if user_input:
        # Add user message to chat history
        chat_history.append({
            "type": "human",
            "content": user_input,
            "image": image_b64,
            "image_type": image_type
        })

        # Prepare chat history for API
        history_serialized = []
        for msg in chat_history:
            if isinstance(msg, dict):
                history_serialized.append({
                    "type": msg["type"],
                    "content": msg["content"]
                })

        # Build payload for /multi-upload
        payload = {
            "question": user_input,
            "session_id": session_id,
            "chat_history": history_serialized
        }
        if image_b64 and image_type:
            payload["image_base64"] = image_b64
            payload["image_type"] = image_type
        if csv_b64 and csv_filename:
            payload["csv_base64"] = csv_b64
            payload["csv_filename"] = csv_filename
        if pdf_b64 and pdf_filename:
            payload["pdf_base64"] = pdf_b64
            payload["pdf_filename"] = pdf_filename

        # Always use /multi-upload if any file is present
        api_url = "http://localhost:8000/multi-upload"

        import requests
        try:
            with st.spinner("Thinking..."):
                res = requests.post(api_url, json=payload)
                res.raise_for_status()
                answer = res.json().get("response", "⚠️ No answer returned.")
                chat_history.append({"type": "ai", "content": answer})
        except Exception as e:
            chat_history.append({"type": "ai", "content": f"❌ API Error: {e}"})

    # Display chat history (unchanged)
    shown_images = set()
    for msg in chat_history:
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
                            if image_hash not in shown_images:
                                shown_images.add(image_hash)
                                image = Image.open(io.BytesIO(image_data))
                                st.image(image, caption="Uploaded Image", use_container_width=True)
                        except Exception as e:
                            st.error(f"🖼️ Could not display image: {e}")