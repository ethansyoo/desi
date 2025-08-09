"""
Authentication module for chatDESI.
"""

# Import only from the new, existing API key manager
from .api_key_manager import APIKeyManager, SimpleModelClient

# Expose only the components that are actually in use
__all__ = [
    "APIKeyManager",
    "SimpleModelClient"
]