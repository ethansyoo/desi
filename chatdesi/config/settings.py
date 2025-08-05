"""Configuration settings for chatDESI application."""

import os
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    cluster_url: str = "cluster89780.vxuht.mongodb.net"
    app_name: str = "mongosh+2.3.3"
    pdf_db_name: str = "pdf_database"
    pdf_collection_name: str = "pdf_documents"
    adql_db_name: str = "pdf_database"
    adql_collection_name: str = "adql_feedback"


@dataclass
class ModelConfig:
    """AI model configuration settings."""
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    default_temperature: float = 0.7
    default_token_limit: int = 1500
    
    # Available models for user selection
    available_models: Dict[str, Dict[str, str]] = None
    
    def __post_init__(self):
        if self.available_models is None:
            self.available_models = {
                "OpenAI GPT-4": {
                    "provider": "openai",
                    "model_name": "gpt-4o",
                    "display_name": "OpenAI GPT-4 Omni",
                    "description": "Most capable OpenAI model"
                },
                "OpenAI GPT-3.5": {
                    "provider": "openai", 
                    "model_name": "gpt-3.5-turbo",
                    "display_name": "OpenAI GPT-3.5 Turbo",
                    "description": "Fast and efficient OpenAI model"
                },
                "Claude 3.5 Sonnet": {
                    "provider": "anthropic",
                    "model_name": "claude-3-5-sonnet-20241022",
                    "display_name": "Claude 3.5 Sonnet",
                    "description": "Anthropic's most capable model"
                },
                "Claude 3 Haiku": {
                    "provider": "anthropic",
                    "model_name": "claude-3-haiku-20240307", 
                    "display_name": "Claude 3 Haiku",
                    "description": "Fast and efficient Anthropic model"
                }
            }


@dataclass
class UIConfig:
    """UI configuration settings."""
    default_max_records: int = 500
    max_max_records: int = 50000
    chunk_size: int = 150
    chunk_overlap: int = 50
    default_top_k: int = 3


@dataclass
class ExternalConfig:
    """External service configuration."""
    github_csv_url: str = "https://raw.githubusercontent.com/ethansyoo/DESI_Chatbot/main/columns.csv"
    tap_service_url: str = "https://datalab.noirlab.edu/tap/sync"
    feedback_form_url: str = "https://forms.gle/pVoAzEgFwKZ4zmXNA"


class Settings:
    """Main settings class that combines all configuration."""
    
    def __init__(self):
        self.database = DatabaseConfig()
        self.model = ModelConfig()
        self.ui = UIConfig()
        self.external = ExternalConfig()
    
    def get_mongodb_connection_string(self, username: str, password: str) -> str:
        """Generate MongoDB connection string."""
        return (
            f"mongodb+srv://{username}:{password}@{self.database.cluster_url}/"
            f"?appName={self.database.app_name}&tls=true"
        )
    
    def get_tap_query_url(self, query: str, max_records: int) -> str:
        """Generate TAP service query URL."""
        formatted_query = query.replace(' ', '+')
        return (
            f"{self.external.tap_service_url}?REQUEST=doQuery&LANG=ADQL&"
            f"FORMAT=csv&QUERY={formatted_query}&MAXREC={max_records}"
        )


# Global settings instance
settings = Settings()