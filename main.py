from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from vector_db.faiss_db import create_faiss_vectorstore
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain import hub
from langchain_core.runnables import RunnableWithMessageHistory, RunnableLambda
from memory.session_memory import get_memory
load_dotenv()

def main():
    try:
        # Load vectorstore and create retriever
        vectorstore = create_faiss_vectorstore()
        print("Creating retriever...")
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={'k': 6, 'lambda_mult': 0.25}
        )
        print("Retriever created successfully.")

        # Initialize Groq LLM
        llm = ChatGroq(
            model_name="llama3-70b-8192",  # Use correct model name
            temperature=0.1,
            max_tokens=1024
        )

        # Create combine_docs_chain
        qa_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
        combine_docs_chain = create_stuff_documents_chain(llm, qa_prompt)        
        # Create retrieval chain
        print("Creating retrieval chain...")
        base_chain = create_retrieval_chain(
            retriever, 
            combine_docs_chain
        )
        wrapped_chain = base_chain | RunnableLambda(lambda x: {"output": x["answer"]})
        chain_with_memory = RunnableWithMessageHistory(
            wrapped_chain,
            lambda session_id: get_memory(session_id),
            input_messages_key="input",
            history_messages_key="chat_history"
        )
        print("Retrieval chain created successfully.")
        return chain_with_memory

    except Exception as e:
        print(f"Error in main function: {e}")
        return None

if __name__ == "__main__":
    chain = main()
    if chain:
        while True:
            query = input("\nUser: ")
            if query.lower() in ["exit", "quit"]:
                break
            response = chain.invoke({"input": query}, config={"configurable": {"session_id": "user-123"}})
            print("\nBot:", response["output"])

