from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from chains.rag_chain import build_chain, build_contextual_chain
from typing import List, Dict, Optional
from memory.session_memory import get_memory
from langchain_core.messages import HumanMessage, AIMessage
from groq import Groq
import base64
import tempfile
import os
from loaders.load_csv import load_csv
from loaders.load_pdf import PyPDFLoader, ingest_pdf
from diskcache import Cache
import hashlib
import pandas as pd
from utils.data_analyzer import DataAnalyzer

# Set up cache directory
cache = Cache(directory="./.cache")
# Store chat histories per session
chat_store = cache.get("chat_store", {})  # recover from cache if available

# Util: Create stable hash key
def hash_data(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


app = FastAPI()

# CORS for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input schema
class QueryRequest(BaseModel):
    question: str
    chat_history: Optional[List[Dict]] = None
    session_id: Optional[str] = None

class ImageQueryRequest(QueryRequest):
    image_base64: str
    image_type: str

class CSVQueryRequest(QueryRequest):
    csv_base64: str
    csv_filename: str

class PdfQueryRequest(QueryRequest):
    pdf_base64: str
    pdf_filename: str

def update_memory_and_history(memory, chat_history, session_id: str):
    session_key = session_id or "default"
    memory.messages.clear()

    # Initialize or fetch session history
    existing_history = chat_store.get(session_key, [])

    updated_history = []

    for msg in chat_history or []:
        # Update memory (langchain) messages
        if msg["type"] == "human":
            memory.add_message(HumanMessage(content=msg["content"]))
        elif msg["type"] == "ai":
            memory.add_message(AIMessage(content=msg["content"]))

        # Track file info if provided
        entry = {
            "type": msg["type"],
            "content": msg["content"],
        }
        if "file" in msg:
            entry["file"] = msg["file"]  # Include file metadata
        updated_history.append(entry)

    # Persist updated chat history
    chat_store[session_key] = existing_history + updated_history
    cache["chat_store"] = chat_store

    # Langchain-style formatted string
    chat_history_str = "\n".join([f"{m.type}: {m.content}" for m in memory.messages])
    return chat_history_str

# Initialize retrieval chain once
rag_chain = build_chain()

@app.post("/chat")
def chat_endpoint(request: QueryRequest):
    memory = get_memory(request.session_id or "default")
    # Pass memory to the chain
    session_key = request.session_id or "default"
    # Update memory + store human message
    chat_history_str = update_memory_and_history(memory, request.chat_history, session_key)
    # Invoke model
    response = rag_chain.invoke({
        "input": request.question,
        "chat_history": chat_history_str
    })
    # Append AI response to history
    chat_store.setdefault(session_key, [])
    chat_store[session_key].append({
        "type": "ai",
        "content": response.get("answer", "No response")
    })
    cache["chat_store"] = chat_store
    return {"response": response.get("answer", "No response")}

@app.post("/image-upload")
def image_upload_endpoint(request: ImageQueryRequest):
    if request.chat_history:
        for msg in request.chat_history:
            if msg["type"] == "human":
                msg["file"] = {
                    "type": "image",
                    "format": request.image_type,
                    "base64": request.image_base64
                }

    image_key = hash_data(request.image_base64)
    if image_key in cache:
        image_context = cache[image_key]
    else:
        client = Groq()
        model = "meta-llama/llama-4-maverick-17b-128e-instruct"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image for data analysis:"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{request.image_type};base64,{request.image_base64}",
                        },
                    },
                ],
            }
        ]
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model
        )
        image_context = chat_completion.choices[0].message.content
        cache[image_key] = image_context
    # Step 2: Use build_contextual_chain with image context
    contextual_chain = build_contextual_chain()

    memory = get_memory(request.session_id or "default")

    session_key = request.session_id or "default"

    # Update memory + store human message
    chat_history_str = update_memory_and_history(memory, request.chat_history, session_key)

    # Invoke model
    response = contextual_chain.invoke({
        "input": request.question,
        "chat_history": chat_history_str,
        "context": image_context
    })

    # Append AI response to history
    chat_store.setdefault(session_key, [])
    chat_store[session_key].append({
        "type": "ai",
        "content": response.content if hasattr(response, "content") else str(response)
    })
    cache["chat_store"] = chat_store  # persist

    # After getting image_context (for image-upload)
    return {"response": response.content if hasattr(response, "content") else str(response)}

@app.post("/csv-upload")
def csv_upload_endpoint(request: CSVQueryRequest):
    
    if request.chat_history:
        for msg in request.chat_history:
            if msg["type"] == "human":
                msg["file"] = {
                    "type": "csv",
                    "name": request.csv_filename,
                    "base64": request.csv_base64
                }

    csv_key = hash_data(request.csv_base64 + request.question)
    if csv_key in cache:
        csv_context = cache[csv_key]
    else:
        csv_bytes = base64.b64decode(request.csv_base64)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_csv:
            tmp_csv.write(csv_bytes)
            tmp_csv_path = tmp_csv.name
        csv_docs = load_csv(tmp_csv_path)
        os.unlink(tmp_csv_path)
        csv_context = "\n".join([doc.page_content for doc in csv_docs])
        cache[csv_key] = csv_context

    contextual_chain = build_contextual_chain()
    memory = get_memory(request.session_id or "default")
    session_key = request.session_id or "default"

    # Update memory + store human message
    chat_history_str = update_memory_and_history(memory, request.chat_history, session_key)

    # Invoke model
    response = contextual_chain.invoke({
        "input": request.question,
        "chat_history": chat_history_str,
        "context": csv_context
    })

    # Append AI response to history
    chat_store.setdefault(session_key, [])
    chat_store[session_key].append({
        "type": "ai",
        "content": response.content if hasattr(response, "content") else str(response)
    })
    cache["chat_store"] = chat_store  # persist

    return {"response": response.content if hasattr(response, "content") else str(response)}

@app.post("/pdf-upload")
def pdf_upload_endpoint(request: PdfQueryRequest):
    if request.chat_history:
        for msg in request.chat_history:
            if msg["type"] == "human":
                msg["file"] = {
                    "type": "pdf",
                    "name": request.pdf_filename,
                    "base64": request.pdf_base64
                }

    pdf_key = hash_data(request.pdf_base64 + request.question)
    if pdf_key in cache:
        pdf_context = cache[pdf_key]
    else:
        pdf_bytes = base64.b64decode(request.pdf_base64)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
            tmp_pdf.write(pdf_bytes)
            tmp_pdf_path = tmp_pdf.name
        loader = PyPDFLoader(tmp_pdf_path)
        pdf_docs = loader.load()
        os.unlink(tmp_pdf_path)
        pdf_context = "\n".join([doc.page_content for doc in pdf_docs])
        cache[pdf_key] = pdf_context


    # Prepare chat history
    memory = get_memory(request.session_id or "default")
    session_key = request.session_id or "default"

    # Update memory + store human message
    chat_history_str = update_memory_and_history(memory, request.chat_history, session_key)

    # Invoke model
    contextual_chain = build_contextual_chain()
    response = contextual_chain.invoke({
        "input": request.question,
        "chat_history": chat_history_str,
        "context": pdf_context
    })

    # Append AI response to history
    chat_store.setdefault(session_key, [])
    chat_store[session_key].append({
        "type": "ai",
        "content": response.content if hasattr(response, "content") else str(response)
    })
    cache["chat_store"] = chat_store  # persist

    return {"response": response.content if hasattr(response, "content") else str(response)}

@app.get("/recent-chats/{session_id}")
def get_recent_chats(session_id: str):
    history = chat_store.get(session_id, [])
    return {"chat_history": chat_store.get(session_id, [])}

@app.get("/recent-chat-titles")
def get_recent_chat_titles():
    titles = []
    for session_id, history in chat_store.items():
        for msg in history:
            if msg["type"] == "human":
                titles.append({
                    "session_id": session_id,
                    "title": msg["content"]
                })
                break  # Only take the first human message
    return {"sessions": titles}