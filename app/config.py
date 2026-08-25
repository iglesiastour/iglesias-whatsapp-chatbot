from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    service_name: str = "iglesias-whatsapp-chatbot"
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    n8n_webhook_url: str = ""
    n8n_timeout_seconds: float = 20.0
    whatsapp_verify_token: str = ""
    whatsapp_access_token: str = ""
    whatsapp_phone_number_id: str = ""
    openai_api_key: str = ""
    google_sheets_id: str = ""
    
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "openai/gpt-4.1-mini"

    database_url: str = ""
    conversation_repository_backend: str = "memory"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
