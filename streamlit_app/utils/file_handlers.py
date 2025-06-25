import base64
import hashlib
import io
from PIL import Image
from typing import Optional, Tuple
import streamlit as st

class FileHandler:
    """Handles file operations and encoding"""
    
    @staticmethod
    def encode_file_to_base64(file) -> str:
        """Encode any file to base64"""
        if hasattr(file, 'read'):
            return base64.b64encode(file.read()).decode('utf-8')
        else:
            return base64.b64encode(file).decode('utf-8')
    
    @staticmethod
    def decode_base64_to_image(base64_str: str) -> Optional[Image.Image]:
        """Decode base64 string to PIL Image"""
        try:
            image_data = base64.b64decode(base64_str)
            return Image.open(io.BytesIO(image_data))
        except Exception as e:
            st.error(f"Could not decode image: {e}")
            return None
    
    @staticmethod
    def get_file_hash(file_data: bytes) -> str:
        """Generate hash for file data"""
        return hashlib.md5(file_data).hexdigest()
    
    @staticmethod
    def validate_file_size(file, max_size_mb: int) -> bool:
        """Validate file size"""
        if hasattr(file, 'size'):
            return file.size <= (max_size_mb * 1024 * 1024)
        return True
    
    @staticmethod
    def get_file_info(file) -> dict:
        """Get file information"""
        return {
            "name": getattr(file, 'name', 'unknown'),
            "size": getattr(file, 'size', 0),
            "type": getattr(file, 'type', 'unknown')
        }

