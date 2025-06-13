from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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

app = FastAPI()

# CORS for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ["http://localhost:8501"]
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

def update_memory_and_history(memory, chat_history):
    if chat_history:
        memory.messages.clear()
        for msg in chat_history:
            if msg["type"] == "human":
                memory.add_message(HumanMessage(content=msg["content"]))
            elif msg["type"] == "ai":
                memory.add_message(AIMessage(content=msg["content"]))
    chat_history_str = "\n".join([f"{m.type}: {m.content}" for m in memory.messages])
    return chat_history_str

# Initialize retrieval chain once
rag_chain = build_chain()

@app.post("/chat")
def chat_endpoint(request: QueryRequest):
    memory = get_memory(request.session_id or "default")
    # Pass memory to the chain
    chat_history_str = update_memory_and_history(memory, request.chat_history)
    response = rag_chain.invoke({
        "input": request.question,
        "chat_history": chat_history_str
    })
    return {"response": response.get("answer", "No response")}

@app.post("/image-upload")
def image_upload_endpoint(request: ImageQueryRequest):
    # Step 1: Get image description from Groq multimodal LLM
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

    # Step 2: Use build_contextual_chain with image context
    contextual_chain = build_contextual_chain()

    memory = get_memory(request.session_id or "default")

    chat_history_str = update_memory_and_history(memory, request.chat_history)

    response = contextual_chain.invoke({
        "input": request.question,
        "chat_history": chat_history_str,
        "context": image_context
    })
    # After getting image_context (for image-upload)
    return {"response": response.content if hasattr(response, "content") else str(response)}

@app.post("/csv-upload")
def csv_upload_endpoint(request: CSVQueryRequest):
    # Decode and save the CSV file temporarily
    csv_bytes = base64.b64decode(request.csv_base64)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_csv:
        tmp_csv.write(csv_bytes)
        tmp_csv_path = tmp_csv.name

    # Load CSV content using your loader
    csv_docs = load_csv(tmp_csv_path)
    os.unlink(tmp_csv_path)  # Clean up temp file

    # Prepare context from CSV content
    csv_context = "\n".join([doc.page_content for doc in csv_docs])

    contextual_chain = build_contextual_chain()

    memory = get_memory(request.session_id or "default")
    chat_history_str = update_memory_and_history(memory, request.chat_history)

    response = contextual_chain.invoke({
        "input": request.question,
        "chat_history": chat_history_str,
        "context": csv_context
    })
    # After getting csv_context (for csv-upload)
    return {"response": response.content if hasattr(response, "content") else str(response)}