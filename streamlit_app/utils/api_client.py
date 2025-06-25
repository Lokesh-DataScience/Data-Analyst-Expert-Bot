import requests
import streamlit as st
from typing import Dict, Any, Optional
from config.settings import AppConfig

class APIClient:
    """Handles all API communications"""
    
    def __init__(self):
        self.base_url = AppConfig.BASE_API_URL
        self.timeout = AppConfig.REQUEST_TIMEOUT
    
    @st.cache_data(ttl=AppConfig.CHAT_HISTORY_CACHE_TTL)
    def get_chat_history(_self, session_id: str) -> Dict[str, Any]:
        """Get chat history for a session"""
        try:
            response = requests.get(
                f"{_self.base_url}/recent-chats/{session_id}",
                timeout=_self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.warning(f"⚠️ Could not load recent chat history: {e}")
            return {"chat_history": []}
    
    @st.cache_data(ttl=AppConfig.RECENT_CHATS_CACHE_TTL)
    def get_recent_chat_titles(_self) -> Dict[str, Any]:
        """Get recent chat titles"""
        try:
            response = requests.get(
                f"{_self.base_url}/recent-chat-titles",
                timeout=_self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Failed to load recent chats: {e}")
            return {"sessions": []}
    
    def send_chat_message(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send a chat message"""
        try:
            response = requests.post(
                f"{self.base_url}/chat",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise Exception(f"Chat API error: {e}")
    
    def upload_image(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Upload image for analysis"""
        try:
            response = requests.post(
                f"{self.base_url}/image-upload",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise Exception(f"Image upload API error: {e}")
    
    def upload_csv(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Upload CSV for analysis"""
        try:
            response = requests.post(
                f"{self.base_url}/csv-upload",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise Exception(f"CSV upload API error: {e}")
    
    def upload_pdf(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Upload PDF for analysis"""
        try:
            response = requests.post(
                f"{self.base_url}/pdf-upload",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise Exception(f"PDF upload API error: {e}")
    
    def analyze_data(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze uploaded data"""
        try:
            response = requests.post(
                f"{self.base_url}/analyze-data",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            # Fallback to basic cleaning
            try:
                response = requests.post(
                    f"{self.base_url}/clean-data",
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()
            except Exception as fallback_e:
                raise Exception(f"Data analysis failed: {e}, Fallback failed: {fallback_e}")

