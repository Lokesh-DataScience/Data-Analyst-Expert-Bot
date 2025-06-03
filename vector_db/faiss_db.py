from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from splitters import recursive_split
from loaders.load_data import load_jsonl

data = load_jsonl("data/gfg_data.jsonl")

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
        vectorstore = FAISS.from_documents(
            recursive_split.split_data_to_documents(data),
            embedding=hf
        )
        print("FAISS vectorstore created successfully.")
        vectorstore.save_local(output_dir)
    except Exception as e:
        print(f"Error creating FAISS vectorstore: {e}")
    return vectorstore