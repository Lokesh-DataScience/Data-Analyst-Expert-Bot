# pdf_ingest.py
from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def ingest_pdf(file_path: str, vectorstore_dir: str = "vectorstore_data"):
    # Load and split PDF into pages
    loader = PyPDFLoader(file_path)
    pages = loader.load()  # returns List[Document]

    # Embed pages
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": False}
    )
    vectorstore = FAISS.from_documents(pages, embedding)
    vectorstore.save_local(vectorstore_dir)

    print(f"✅ Ingested PDF and saved to {vectorstore_dir}")
