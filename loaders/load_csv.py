from langchain_community.document_loaders.csv_loader import CSVLoader

def load_csv(file_path: str):
    """Load a CSV file and return its content."""
    loader = CSVLoader(file_path=file_path)
    documents = loader.load()
    return documents