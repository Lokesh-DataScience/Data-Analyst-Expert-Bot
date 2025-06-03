from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from vector_db.faiss_db import create_faiss_vectorstore
from langchain_groq import ChatGroq
from dotenv import load_dotenv

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
        prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant. Use the following context to answer the user's question.
        
        Context:
        {context}
        
        Question: {input}
        """)
        combine_docs_chain = create_stuff_documents_chain(llm, prompt)

        # Create retrieval chain
        print("Creating retrieval chain...")
        retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
        print("Retrieval chain created successfully.")
        return retrieval_chain

    except Exception as e:
        print(f"Error in main function: {e}")
        return None

if __name__ == "__main__":
    chain = main()
    if chain:
        while True:
            query = input("You: ")
            if query == "exit":
                print("Exiting...")
                break
            elif query.strip() == "":
                print("Please enter a valid query.")
                continue
            else:
              response = chain.invoke({"input": query})
              print("\nAnswer:\n", response["answer"])


