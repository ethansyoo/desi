"""
Authentication module for chatDESI.
"""

from .credentials import (
    Credentials,
    CredentialManager, 
    StreamlitAuth,
    create_auth_system,
    require_authentication
)

# Import new practical API key manager
try:
    from .api_key_manager import APIKeyManager, SimpleModelClient
    __all__ = [
        "Credentials", "CredentialManager", "StreamlitAuth", 
        "create_auth_system", "require_authentication", 
        "APIKeyManager", "SimpleModelClient"
    ]
except ImportError:
    __all__ = [
        "Credentials", "CredentialManager", "StreamlitAuth", 
        "create_auth_system", "require_authentication"
    ]