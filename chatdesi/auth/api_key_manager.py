"""
Practical API key management with live testing.
"""

from typing import Dict, Optional, Iterable
import time

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False


class APIKeyManager:
    """API key management with live validation."""
    
    def __init__(self):
        # Use session state to cache test results to avoid re-testing keys
        if 'tested_keys' not in st.session_state:
            st.session_state.tested_keys = {}
    
    def test_openai_key(self, api_key: str) -> bool:
        """Test OpenAI API key with a minimal request."""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=1
            )
            return True
        except Exception:
            return False
    
    def test_anthropic_key(self, api_key: str) -> bool:
        """Test Anthropic API key with a minimal request."""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1,
                messages=[{"role": "user", "content": "Hi"}]
            )
            return True
        except Exception:
            return False
    
    def get_user_api_keys(self) -> Dict[str, str]:
        """Get and test API keys from user input."""
        st.write("### 🔑 AI Model Configuration")
        api_keys = {}
        
        openai_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
        if openai_key:
            # Check cache before making an API call
            if st.session_state.tested_keys.get(openai_key) or self.test_openai_key(openai_key):
                api_keys["openai"] = openai_key
                st.session_state.tested_keys[openai_key] = True
                st.success("✅ OpenAI API key is valid.")
            else:
                st.error("❌ Invalid OpenAI API key.")

        anthropic_key = st.text_input("Anthropic API Key", type="password", placeholder="sk-ant-...")
        if anthropic_key:
            if st.session_state.tested_keys.get(anthropic_key) or self.test_anthropic_key(anthropic_key):
                api_keys["anthropic"] = anthropic_key
                st.session_state.tested_keys[anthropic_key] = True
                st.success("✅ Anthropic API key is valid.")
            else:
                st.error("❌ Invalid Anthropic API key.")
                
        return api_keys
    
    def get_available_models(self, api_keys: Dict[str, str]) -> Dict[str, Dict[str, str]]:
        """Get available models based on working API keys."""
        from ..config import settings
        available_models = {}
        all_models = settings.model.available_models
        
        if "openai" in api_keys:
            for k, v in all_models.items():
                if v["provider"] == "openai":
                    available_models[k] = v
        
        if "anthropic" in api_keys:
            for k, v in all_models.items():
                if v["provider"] == "anthropic":
                    available_models[k] = v
                    
        return available_models
    
    def render_model_selector(self, available_models: Dict[str, Dict[str, str]]) -> Optional[str]:
        """Render model selection interface."""
        if not available_models:
            st.warning("⚠️ No working API keys found. Please enter valid API keys above.")
            return None
        
        st.write("### 🎯 Model Selection")
        
        model_options = {f"{info['display_name']} ({info['provider']})": key 
                         for key, info in available_models.items()}
        
        selected_display = st.selectbox("Choose AI Model:", options=list(model_options.keys()))
        return model_options[selected_display]


class SimpleModelClient:
    """Simple client wrapper that works with any provider."""
    
    def __init__(self, provider: str, api_key: str, model_name: str):
        self.provider = provider
        self.model_name = model_name
        
        if provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
        elif provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def _stream_completion(self, messages, max_tokens, temperature) -> Iterable[str]:
        """Private generator for handling streaming responses."""
        if self.provider == "openai":
            stream = self.client.chat.completions.create(
                model=self.model_name, messages=messages, max_tokens=max_tokens,
                temperature=temperature, stream=True,
            )
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        
        elif self.provider == "anthropic":
            system_message = next((msg["content"] for msg in messages if msg["role"] == "system"), None)
            user_messages = [msg for msg in messages if msg["role"] != "system"]
            with self.client.messages.stream(
                model=self.model_name, max_tokens=max_tokens, temperature=temperature,
                messages=user_messages, system=system_message
            ) as stream:
                for text in stream.text_stream:
                    yield text

    def chat_completion(self, messages, max_tokens, temperature, stream=False):
        """Generate chat completion, supporting both streaming and non-streaming."""
        if stream:
            return self._stream_completion(messages, max_tokens, temperature)
        
        # Fallback for non-streaming requests
        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model_name, messages=messages, max_tokens=max_tokens, temperature=temperature
            )
            return response.choices[0].message.content
        elif self.provider == "anthropic":
            system_message = next((msg["content"] for msg in messages if msg["role"] == "system"), None)
            user_messages = [msg for msg in messages if msg["role"] != "system"]
            response = self.client.messages.create(
                model=self.model_name, max_tokens=max_tokens, temperature=temperature,
                messages=user_messages, system=system_message
            )
            return response.content[0].text

    def get_provider_name(self) -> str:
        return self.provider