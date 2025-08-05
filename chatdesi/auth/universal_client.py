"""
Universal AI client wrapper for different providers.
"""

from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod


class UniversalAIClient(ABC):
    """Abstract base class for AI clients."""
    
    @abstractmethod
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_tokens: int,
        temperature: float
    ) -> str:
        """Generate chat completion."""
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Get provider name."""
        pass


class OpenAIClient(UniversalAIClient):
    """OpenAI client wrapper."""
    
    def __init__(self, client):
        self.client = client
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_tokens: int,
        temperature: float
    ) -> str:
        """Generate OpenAI chat completion."""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")
    
    def get_provider_name(self) -> str:
        return "openai"


class AnthropicClient(UniversalAIClient):
    """Anthropic client wrapper."""
    
    def __init__(self, client):
        self.client = client
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_tokens: int,
        temperature: float
    ) -> str:
        """Generate Anthropic chat completion."""
        try:
            # Convert messages format for Anthropic
            anthropic_messages = []
            system_message = None
            
            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    anthropic_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
            
            # Create the request
            kwargs = {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": anthropic_messages
            }
            
            if system_message:
                kwargs["system"] = system_message
            
            response = self.client.messages.create(**kwargs)
            return response.content[0].text
            
        except Exception as e:
            raise Exception(f"Anthropic API error: {str(e)}")
    
    def get_provider_name(self) -> str:
        return "anthropic"


class AIClientFactory:
    """Factory for creating universal AI clients."""
    
    @staticmethod
    def create_universal_client(provider: str, raw_client) -> UniversalAIClient:
        """Create universal client wrapper."""
        if provider == "openai":
            return OpenAIClient(raw_client)
        elif provider == "anthropic":
            return AnthropicClient(raw_client)
        else:
            raise ValueError(f"Unsupported provider: {provider}")


class ModelSelector:
    """Handles model selection and client creation."""
    
    def __init__(self):
        self.selected_model_info = None
        self.universal_client = None
    
    def set_model(self, model_key: str, model_info: Dict[str, str], api_keys: Dict[str, str]):
        """Set the active model and create client."""
        from .api_key_manager import ModelClientFactory
        
        provider = model_info["provider"]
        api_key = api_keys.get(provider)
        
        if not api_key:
            raise ValueError(f"No API key provided for {provider}")
        
        # Create raw client
        raw_client = ModelClientFactory.create_client(provider, api_key)
        
        # Create universal client
        self.universal_client = AIClientFactory.create_universal_client(provider, raw_client)
        self.selected_model_info = model_info
    
    def get_client(self) -> Optional[UniversalAIClient]:
        """Get the universal client."""
        return self.universal_client
    
    def get_model_name(self) -> Optional[str]:
        """Get the current model name."""
        return self.selected_model_info["model_name"] if self.selected_model_info else None
    
    def is_ready(self) -> bool:
        """Check if model is ready to use."""
        return self.universal_client is not None and self.selected_model_info is not None