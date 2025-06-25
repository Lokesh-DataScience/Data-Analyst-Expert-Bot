class AppConfig:
    """Application configuration settings"""
    
    # App Settings
    PAGE_TITLE = "Analyst Bot"
    APP_TITLE = "🤖 AI-Powered Data Analysis Platform"
    APP_DESCRIPTION = "Upload your dataset and let AI handle the complete analysis pipeline"
    
    # API Settings
    BASE_API_URL = "http://localhost:8000"
    REQUEST_TIMEOUT = 300
    
    # File Upload Settings
    MAX_FILE_SIZE_MB = 200
    ALLOWED_CSV_TYPES = ["csv", "xlsx", "xls"]
    ALLOWED_IMAGE_TYPES = ["jpg", "jpeg", "png"]
    ALLOWED_PDF_TYPES = ["pdf"]
    
    # Image Upload Limits
    IMAGE_UPLOAD_LIMIT = 3
    IMAGE_UPLOAD_WINDOW_HOURS = 6
    
    # Cache Settings
    CHAT_HISTORY_CACHE_TTL = 120  # 2 minutes
    RECENT_CHATS_CACHE_TTL = 60   # 1 minute
    
    # Display Settings
    DEFAULT_MAX_ROWS_DISPLAY = 10
    MAX_CHAT_HISTORY_DISPLAY = 50
