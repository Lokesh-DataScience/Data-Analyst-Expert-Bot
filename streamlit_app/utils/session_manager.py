import streamlit as st
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any
import requests

class SessionManager:
    """Manages Streamlit session state and user sessions"""
    
    def __init__(self):
        self.session_id_key = "session_id"
        self.chat_history_key = "chat_history"
        self.image_attempts_key = "image_upload_attempts"
        self.last_image_key = "last_uploaded_image_name"
    
    def initialize_session(self):
        """Initialize all session state variables"""
        if self.session_id_key not in st.session_state:
            st.session_state[self.session_id_key] = str(uuid.uuid4())
        
        if self.chat_history_key not in st.session_state:
            st.session_state[self.chat_history_key] = []
        
        if self.image_attempts_key not in st.session_state:
            st.session_state[self.image_attempts_key] = []
        
        if self.last_image_key not in st.session_state:
            st.session_state[self.last_image_key] = None
        
        # Clean old image upload attempts
        self._clean_old_image_attempts()
    
    def get_session_id(self) -> str:
        """Get current session ID"""
        return st.session_state[self.session_id_key]
    
    def get_chat_history(self) -> List[Dict[str, Any]]:
        """Get current chat history"""
        return st.session_state[self.chat_history_key]
    
    def add_message(self, message_type: str, content: str, **kwargs):
        """Add a message to chat history"""
        message = {
            "type": message_type,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        st.session_state[self.chat_history_key].append(message)
    
    def clear_chat_history(self):
        """Clear current chat history"""
        st.session_state[self.chat_history_key] = []
    
    def start_new_session(self):
        """Start a new chat session"""
        st.session_state[self.session_id_key] = str(uuid.uuid4())
        self.clear_chat_history()
        st.session_state[self.last_image_key] = None
    
    def load_session(self, session_id: str, chat_history: List[Dict]):
        """Load an existing session"""
        st.session_state[self.session_id_key] = session_id
        st.session_state[self.chat_history_key] = chat_history
    
    def track_image_upload(self, image_name: str):
        """Track image upload for rate limiting"""
        current_time = datetime.now()
        st.session_state[self.image_attempts_key].append(current_time)
        st.session_state[self.last_image_key] = image_name
    
    def get_remaining_image_uploads(self) -> int:
        """Get remaining image uploads in current window"""
        from config.settings import AppConfig
        return max(0, AppConfig.IMAGE_UPLOAD_LIMIT - len(st.session_state[self.image_attempts_key]))
    
    def is_image_upload_allowed(self) -> bool:
        """Check if image upload is allowed"""
        from config.settings import AppConfig
        return len(st.session_state[self.image_attempts_key]) < AppConfig.IMAGE_UPLOAD_LIMIT
    
    def _clean_old_image_attempts(self):
        """Remove old image upload attempts outside the time window"""
        from config.settings import AppConfig
        cutoff_time = datetime.now() - timedelta(hours=AppConfig.IMAGE_UPLOAD_WINDOW_HOURS)
        st.session_state[self.image_attempts_key] = [
            ts for ts in st.session_state[self.image_attempts_key] 
            if ts > cutoff_time
        ]