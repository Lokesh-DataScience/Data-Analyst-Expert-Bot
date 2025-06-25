import base64

def encode_image_file(file):
    """Encodes an image file to base64."""
    file.seek(0)
    return base64.b64encode(file.read()).decode('utf-8')

def encode_csv_file(file):
    """Encodes a CSV file to base64 and returns (b64_string, filename)."""
    file.seek(0)
    b64 = base64.b64encode(file.read()).decode('utf-8')
    return b64, file.name

def encode_pdf_file(file):
    """Encodes a PDF file to base64 and returns (b64_string, filename)."""
    file.seek(0)
    b64 = base64.b64encode(file.read()).decode('utf-8')
    return b64, file.name

def get_file_details(file):
    """Returns a dict with file details for display."""
    return {
        "Filename": file.name,
        "File size": f"{file.size / 1024:.2f} KB",
        "File type": file.type
    }