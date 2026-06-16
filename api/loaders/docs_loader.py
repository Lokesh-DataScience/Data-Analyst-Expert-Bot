import json
from typing import List, Dict, Any

from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


class DocumentLoader:
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu"
    ):
        self.embedding = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": False}
        )

    def load_pdf(self, file_path: str) -> List[Document]:
        """
        Load a PDF file and return pages as LangChain Documents.
        """
        loader = PyPDFLoader(file_path)
        return loader.load()

    def ingest_pdf(
        self,
        file_path: str,
        vectorstore_dir: str = "vectorstore_data"
    ) -> None:
        """
        Load PDF, create embeddings, and save FAISS vector store.
        (Writes to a single shared store — kept for backwards compatibility.
        For per-user ingestion, use rag_chain.add_documents_to_user_store
        with the documents returned by load_pdf/load_csv instead.)
        """
        documents = self.load_pdf(file_path)

        vectorstore = FAISS.from_documents(
            documents,
            self.embedding
        )

        vectorstore.save_local(vectorstore_dir)

        print(f"✅ Ingested PDF and saved to {vectorstore_dir}")

    def load_csv(self, file_path: str) -> List[Document]:
        """
        Load a CSV file and return LangChain Documents.
        """
        loader = CSVLoader(file_path=file_path)
        docs = loader.load()
        return docs

    def load_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load a JSONL file and return a list of dictionaries.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f]
        except Exception as e:
            print(f"Error loading JSONL file: {e}")
            return []

    def create_vectorstore(
        self,
        documents: List[Document],
        vectorstore_dir: str = "vectorstore_data"
    ) -> None:
        """
        Create and save a FAISS vector store from documents.
        """
        vectorstore = FAISS.from_documents(
            documents,
            self.embedding
        )

        vectorstore.save_local(vectorstore_dir)

        print(f"✅ Vector store saved to {vectorstore_dir}")