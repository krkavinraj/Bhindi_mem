"""
Configuration management for the Bhindi Knowledge Graph system.
"""
import os
from typing import Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class Neo4jConfig:
    """Neo4j database configuration."""
    uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    username: str = os.getenv("NEO4J_USERNAME", "neo4j")
    password: str = os.getenv("NEO4J_PASSWORD", "password")
    database: str = os.getenv("NEO4J_DATABASE", "neo4j")

@dataclass
class AzureOpenAIConfig:
    """Azure OpenAI API configuration."""
    api_key: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    deployment_name: str = os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT4O", "gpt-4o")
    model: str = os.getenv("AZURE_OPENAI_MODEL", "gpt-4o")
    max_tokens: int = int(os.getenv("AZURE_OPENAI_MAX_TOKENS", "2000"))
    temperature: float = float(os.getenv("AZURE_OPENAI_TEMPERATURE", "0.7"))

@dataclass
class AppConfig:
    """Application configuration."""
    debug: bool = os.getenv("DEBUG", "False").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    graph_layout: str = os.getenv("GRAPH_LAYOUT", "force")
    auto_refresh: bool = os.getenv("AUTO_REFRESH", "True").lower() == "true"
    max_nodes_display: int = int(os.getenv("MAX_NODES_DISPLAY", "100"))

class Config:
    """Main configuration class."""
    
    def __init__(self):
        self.neo4j = Neo4jConfig()
        self.azure_openai = AzureOpenAIConfig()
        self.app = AppConfig()
    
    def validate(self) -> bool:
        """Validate configuration settings."""
        if not self.azure_openai.api_key:
            raise ValueError("Azure OpenAI API key is required")
        
        if not self.azure_openai.endpoint:
            raise ValueError("Azure OpenAI endpoint is required")
        
        if not self.neo4j.uri:
            raise ValueError("Neo4j URI is required")
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "neo4j": {
                "uri": self.neo4j.uri,
                "username": self.neo4j.username,
                "database": self.neo4j.database
            },
            "azure_openai": {
                "endpoint": self.azure_openai.endpoint,
                "api_version": self.azure_openai.api_version,
                "deployment_name": self.azure_openai.deployment_name,
                "model": self.azure_openai.model,
                "max_tokens": self.azure_openai.max_tokens,
                "temperature": self.azure_openai.temperature
            },
            "app": {
                "debug": self.app.debug,
                "log_level": self.app.log_level,
                "graph_layout": self.app.graph_layout,
                "auto_refresh": self.app.auto_refresh,
                "max_nodes_display": self.app.max_nodes_display
            }
        }

# Global configuration instance
config = Config()

# Node type colors for visualization
NODE_COLORS = {
    "Person": "#FF6B6B",      # Red
    "Concept": "#4ECDC4",     # Teal
    "Event": "#45B7D1",       # Blue
    "Preference": "#96CEB4",  # Green
    "Location": "#FFEAA7",    # Yellow
    "Organization": "#DDA0DD", # Plum
    "Skill": "#98D8C8",       # Mint
    "Goal": "#F7DC6F",        # Light Yellow
    "Memory": "#BB8FCE",      # Light Purple
    "Default": "#BDC3C7"      # Gray
}

# Relationship types
RELATIONSHIP_TYPES = [
    "KNOWS", "LIKES", "DISLIKES", "WORKS_AT", "LIVES_IN",
    "ATTENDED", "SKILLED_IN", "WANTS_TO", "REMEMBERS",
    "RELATED_TO", "PART_OF", "CREATED", "LEARNED"
]
