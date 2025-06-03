from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

def split_data_to_documents(
    data: list[dict],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    source: str = "geeksforgeeks.org"
) -> list[Document]:
    """
    Splits input data into Document chunks using RecursiveCharacterTextSplitter.

    Args:
        data: List of dicts with 'content', 'title', and 'link' keys.
        chunk_size: Size of each text chunk.
        chunk_overlap: Overlap between chunks.
        source: Source metadata for each document.

    Returns:
        List of Document objects.
    """
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    documents = []

    try:
        for item in data:
            chunks = text_splitter.split_text(item['content'])
            for i, chunk in enumerate(chunks):
                documents.append(Document(
                    page_content=chunk,
                    metadata={
                        "title": item['title'],

                        "chunk_id": i + 1,
                        "source": source
                    }
                ))
        return documents
    except Exception as e:
        print(f"Error splitting data into documents: {e}")
        return []
    