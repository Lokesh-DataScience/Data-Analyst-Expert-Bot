
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq
from vector_db.faiss_db import create_faiss_vectorstore
from langchain import hub
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def build_chain():
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": False}
    )
    vectorstore = FAISS.load_local("vectorstore_data", embedding, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={'k': 6, 'lambda_mult': 0.25}
    )

    # Load prompt for retrieval QA chat
    retrieval_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    # Initialize Groq LLM
    llm = ChatGroq(
        model_name="llama3-70b-8192",  # Use correct model name
        temperature=0.1,
        max_tokens=1024
    )

    # Create a combine_docs_chain that prepares the prompt for the LLM
    combine_docs_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=retrieval_prompt
    )

    # Create the retrieval chain
    chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=combine_docs_chain
    )

    return chain
