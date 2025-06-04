# vectorstore_create.py

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from splitters import recursive_split
from loaders.load_data import load_jsonl

def create_and_save_vectorstore(
    data_path,
    output_dir,
    model_name,
    device
):
    print("Loading and splitting data...")
    data = load_jsonl(data_path)
    docs = recursive_split.split_data_to_documents(data)

    print("Loading embedding model...")
    hf = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": False}
    )

    print("Creating FAISS vectorstore...")
    vectorstore = FAISS.from_documents(docs, embedding=hf)
    vectorstore.save_local(output_dir)
    print(f"Vectorstore saved at: {output_dir}")

if __name__ == "__main__":
    create_and_save_vectorstore(
        data_path="data/data.jsonl",
        output_dir="vectorstore_data",
        model_name="sentence-transformers/all-mpnet-base-v2",
        device="cpu"
    )