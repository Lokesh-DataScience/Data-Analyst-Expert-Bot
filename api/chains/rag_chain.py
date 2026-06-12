import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.runnables.base import RunnableSequence
from vector_db.faiss_db import EMBEDDING

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

def build_chain():
    vectorstore = FAISS.load_local("vectorstore_data", EMBEDDING, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={'k': 6, 'lambda_mult': 0.25}
    )
    # prompt for retrieval QA chat
    retrieval_prompt = PromptTemplate(
        input_variables=INPUT_VARIABLES,
        template=TEMPLATE
    )
    # Initialize Groq LLM
    llm = LLM
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

def build_contextual_chain():
    contextual_prompt = PromptTemplate(
        input_variables=INPUT_VARIABLES,
        template=TEMPLATE
    )
    llm = LLM
    return RunnableSequence(contextual_prompt, llm)

