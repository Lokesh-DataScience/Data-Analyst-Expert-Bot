import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from loaders.load_data import load_jsonl
from langchain_core.documents import Document

# vector_db/faiss_db.py
data = load_jsonl("data/data.jsonl")

EMBEDDING = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": False}
)

def create_faiss_vectorstore(output_dir="vectorstore_data"):
    """
    Create a FAISS vectorstore from JSONL data using HuggingFace embeddings.
    """
    try:
        print("Loading model...")
        print("Model loaded successfully.")
        print("Creating FAISS vectorstore...")
        
        documents = [
            Document(
                page_content=item["content"], # Content of the document
                metadata={
                    "title": item.get("title", ""),
                    "chunk_id": item.get("chunk_id", ""),
                    "source": item.get("source", ""),
                }
            )
            for item in data # Load data from JSONL
        ]
        # Ensure documents are properly formatted
        vectorstore = FAISS.from_documents(
            documents,
            embedding=EMBEDDING
        )
        print("FAISS vectorstore created successfully.")
        # Save the vectorstore to disk
        vectorstore.save_local(output_dir)
    except Exception as e:
        print(f"Error creating FAISS vectorstore: {e}")

if __name__ == "__main__":
    # Create the FAISS vectorstore
    create_faiss_vectorstore()