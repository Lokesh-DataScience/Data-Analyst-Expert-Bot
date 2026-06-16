import sys
import os
import shutil
import hashlib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.runnables.base import RunnableSequence
from data.vector_db.faiss_db import EMBEDDING

LLM = ChatGroq(
    model_name="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.1,
    max_tokens=1024
)

TEMPLATE = """
        You are a helpful AI assistant who specializes in data analysis. Your primary goal is to assist with topics related to data analysis, including (but not limited to): data cleaning, visualization, statistical analysis, machine learning for analytics, tools like Python, SQL, Excel, and business intelligence.

        If the user's input is clearly unrelated to data analysis (e.g., topics like cooking, history, movies, etc.), politely respond with:
        "I specialize in data analysis. Feel free to ask me anything related to that!"

        If the input is vague or general (e.g., “What do you know?” or “Tell me something interesting”), you can steer the conversation by briefly responding and guiding it toward data analysis, like:
        "I know quite a bit about data analysis! Would you like to explore a topic like data cleaning, visualization, or tools like Python and SQL?"

        Do not attempt to answer unrelated questions in detail.
        Chat History:
        {chat_history}  
        
        Context:
        {context}

        Input:
        {input}

        Answer:"""

INPUT_VARIABLES = ["context", "input", "chat_history"]

# ============================================================
# PER-USER VECTOR STORE PATHS
# ============================================================
BASE_VECTORSTORE_DIR = os.path.join("data", "vectorstore_data")
USER_VECTORSTORE_ROOT = os.path.join("data", "user_vectorstores")

os.makedirs(USER_VECTORSTORE_ROOT, exist_ok=True)


def _user_dir(user_email: str) -> str:
    """
    Maps a user's email to a filesystem-safe directory name.
    Uses a short hash to avoid issues with special characters in emails.
    """
    safe_id = hashlib.sha256(user_email.encode("utf-8")).hexdigest()[:24]
    return os.path.join(USER_VECTORSTORE_ROOT, safe_id)


def get_user_vectorstore_path(user_email: str) -> str:
    """
    Returns the path to a user's personal vector store, creating it
    (seeded from the shared base knowledge base) on first access.
    """
    user_path = _user_dir(user_email)

    if not os.path.exists(os.path.join(user_path, "index.faiss")):
        os.makedirs(user_path, exist_ok=True)
        if os.path.exists(os.path.join(BASE_VECTORSTORE_DIR, "index.faiss")):
            # Seed the user's personal store with a copy of the shared
            # knowledge base, so they start with the same baseline
            # expertise before their own uploads are added on top.
            for fname in os.listdir(BASE_VECTORSTORE_DIR):
                shutil.copy2(
                    os.path.join(BASE_VECTORSTORE_DIR, fname),
                    os.path.join(user_path, fname),
                )
        else:
            # No base index available — create an empty store lazily
            # the first time the user adds a document.
            pass

    return user_path


def load_user_vectorstore(user_email: str) -> FAISS:
    """
    Loads (or seeds) the FAISS vector store for a specific user.
    Falls back to the shared base store for anonymous/demo users.
    """
    if not user_email or user_email == "anonymous":
        return FAISS.load_local(
            BASE_VECTORSTORE_DIR, EMBEDDING, allow_dangerous_deserialization=True
        )

    user_path = get_user_vectorstore_path(user_email)
    if not os.path.exists(os.path.join(user_path, "index.faiss")):
        # Still empty (no base index existed and nothing ingested yet)
        return FAISS.load_local(
            BASE_VECTORSTORE_DIR, EMBEDDING, allow_dangerous_deserialization=True
        )

    return FAISS.load_local(
        user_path, EMBEDDING, allow_dangerous_deserialization=True
    )


def add_documents_to_user_store(user_email: str, documents) -> None:
    """
    Adds new documents (e.g. from an uploaded PDF/CSV) into a user's
    personal vector store, growing their private knowledge base over time.
    """
    if not documents:
        return
    if not user_email or user_email == "anonymous":
        return  # don't pollute the shared base store with anonymous uploads

    user_path = get_user_vectorstore_path(user_email)

    if os.path.exists(os.path.join(user_path, "index.faiss")):
        store = FAISS.load_local(
            user_path, EMBEDDING, allow_dangerous_deserialization=True
        )
        store.add_documents(documents)
    else:
        store = FAISS.from_documents(documents, EMBEDDING)

    store.save_local(user_path)


# ============================================================
# CHAIN BUILDERS
# ============================================================
def build_chain(user_email: str = None):
    """
    Builds the retrieval chain. If user_email is provided, retrieval
    uses that user's personal vector store (their own uploaded docs +
    the seeded base knowledge base). Otherwise falls back to the
    shared base store.
    """
    vectorstore = load_user_vectorstore(user_email)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={'k': 6, 'lambda_mult': 0.25}
    )
    retrieval_prompt = PromptTemplate(
        input_variables=INPUT_VARIABLES,
        template=TEMPLATE
    )
    llm = LLM
    combine_docs_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=retrieval_prompt
    )
    chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=combine_docs_chain
    )
    return chain


def build_contextual_chain():
    contextual_prompt = PromptTemplate(
        input_variables=INPUT_VARIABLES,
        template=TEMPLATE
    )
    llm = LLM
    return RunnableSequence(contextual_prompt, llm)