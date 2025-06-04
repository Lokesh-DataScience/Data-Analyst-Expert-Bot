import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from loaders.load_data import load_jsonl
from langchain_core.documents import Document

data = load_jsonl("data/data.jsonl")

def create_faiss_vectorstore(
    output_dir="vectorstore_data",
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
):
    try:
        print("Loading model...")
        model_kwargs = {'device': device}
        encode_kwargs = {'normalize_embeddings': False}
        hf = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        print("Model loaded successfully.")
        print("Creating FAISS vectorstore...")
        documents = [
            Document(
                page_content=item["content"],
                metadata={
                    "title": item.get("title", ""),
                    "chunk_id": item.get("chunk_id", ""),
                    "source": item.get("source", ""),
                }
            )
            for item in data
        ]

        vectorstore = FAISS.from_documents(
            documents,
            embedding=hf
        )
        print("FAISS vectorstore created successfully.")
        vectorstore.save_local(output_dir)
    except Exception as e:
        print(f"Error creating FAISS vectorstore: {e}")

if __name__ == "__main__":
    create_faiss_vectorstore()