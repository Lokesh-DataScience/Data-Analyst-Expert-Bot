import json

def load_jsonl(path):
    """
    Load a JSONL file and return a list of dictionaries.
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]
    except Exception as e:
        print(f"Error loading JSONL file: {e}")
        return []
