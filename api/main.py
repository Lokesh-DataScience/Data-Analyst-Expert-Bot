from fastapi import FastAPI
from api.chat import router as chat_router

app = FastAPI(title="RAG + Groq Chatbot API")

app.include_router(chat_router, prefix="/api", tags=["chat"])

@app.get("/")
def read_root():
    return {"message": "Welcome to the RAG + LLM Chatbot API"}
