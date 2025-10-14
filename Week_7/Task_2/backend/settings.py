import os

class Settings:
    google_api_key: str = os.environ.get("GOOGLE_API_KEY", "")
    openai_api_key: str = os.environ.get("OPENAI_API_KEY", "")
    app_api_token: str  = os.environ.get("APP_API_TOKEN", "")  # bearer auth for UI→API

settings = Settings()
