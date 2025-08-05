"""
Practical API key management with live testing.
"""

from typing import Dict, Optional
import time

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False


class APIKeyManager:
    """API key management with live validation."""
    
    def __init__(self):
        self.tested_keys = {}  # Cache test results
    
    def test_openai_key(self, api_key: str) -> bool:
        """Test OpenAI API key with a minimal request."""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            
            # Make a minimal test request
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=1
            )
            return True
            
        except Exception as e:
            error_msg = str(e).lower()
            if "authentication" in error_msg or "api key" in error_msg:
                return False
            elif "rate limit" in error_msg:
                # Rate limited but key is valid
                return True
            else:
                # Other error - assume key is bad
                return False
    
    def test_anthropic_key(self, api_key: str) -> bool:
        """Test Anthropic API key with a minimal request."""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            
            # Make a minimal test request
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1,
                messages=[{"role": "user", "content": "Hi"}]
            )
            return True
            
        except Exception as e:
            error_msg = str(e).lower()
            if "authentication" in error_msg or "api key" in error_msg:
                return False
            elif "rate limit" in error_msg:
                # Rate limited but key is valid
                return True
            else:
                return False
    
    def get_user_api_keys(self) -> Dict[str, str]:
        """Get and test API keys from user input."""
        if not STREAMLIT_AVAILABLE:
            return {}
        
        st.write("### 🔑 AI Model Configuration")
        st.write("Enter your API keys to enable additional models:")
        
        api_keys = {}
        
        # OpenAI API Key
        with st.expander("🤖 OpenAI API Key", expanded=True):
            openai_key = st.text_input(
                "OpenAI API Key",
                type="password",
                placeholder="sk-... or sk-proj-...",
                help="Get your key from https://platform.openai.com/api-keys",
                key="openai_api_key"
            )
            
            if openai_key and openai_key.strip():
                key = openai_key.strip()
                
                # Check if we've already tested this key
                if key in self.tested_keys:
                    if self.tested_keys[key]["openai"]:
                        api_keys["openai"] = key
                        st.success("✅ OpenAI API key verified")
                    else:
                        st.error("❌ OpenAI API key failed previous test")
                else:
                    # Test the key
                    with st.spinner("Testing OpenAI API key..."):
                        is_valid = self.test_openai_key(key)
                        self.tested_keys[key] = {"openai": is_valid}
                        
                        if is_valid:
                            api_keys["openai"] = key
                            st.success("✅ OpenAI API key works!")
                        else:
                            st.error("❌ OpenAI API key test failed. Please check your key.")
        
        # Anthropic API Key
        with st.expander("🧠 Anthropic API Key", expanded=False):
            anthropic_key = st.text_input(
                "Anthropic API Key",
                type="password",
                placeholder="sk-ant-...",
                help="Get your key from https://console.anthropic.com/",
                key="anthropic_api_key"
            )
            
            if anthropic_key and anthropic_key.strip():
                key = anthropic_key.strip()
                
                # Check if we've already tested this key
                if key in self.tested_keys:
                    if self.tested_keys[key].get("anthropic", False):
                        api_keys["anthropic"] = key
                        st.success("✅ Anthropic API key verified")
                    else:
                        st.error("❌ Anthropic API key failed previous test")
                else:
                    # Test the key
                    with st.spinner("Testing Anthropic API key..."):
                        is_valid = self.test_anthropic_key(key)
                        if key not in self.tested_keys:
                            self.tested_keys[key] = {}
                        self.tested_keys[key]["anthropic"] = is_valid
                        
                        if is_valid:
                            api_keys["anthropic"] = key
                            st.success("✅ Anthropic API key works!")
                        else:
                            st.error("❌ Anthropic API key test failed. Please check your key.")
        
        return api_keys
    
    def get_available_models(self, api_keys: Dict[str, str]) -> Dict[str, Dict[str, str]]:
        """Get available models based on working API keys."""
        available_models = {}
        
        if "openai" in api_keys:
            available_models.update({
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
                }
            })
        
        if "anthropic" in api_keys:
            available_models.update({
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
            })
        
        return available_models
    
    def render_model_selector(self, available_models: Dict[str, Dict[str, str]]) -> Optional[str]:
        """Render model selection interface."""
        if not STREAMLIT_AVAILABLE:
            return None
        
        if not available_models:
            st.warning("⚠️ No working API keys found. Please enter valid API keys above.")
            return None
        
        st.write("### 🎯 Model Selection")
        
        # Create options for selectbox
        model_options = {}
        for key, info in available_models.items():
            display_text = f"{info['display_name']} - {info['description']}"
            model_options[display_text] = key
        
        selected_display = st.selectbox(
            "Choose AI Model:",
            options=list(model_options.keys()),
            index=0,
            help="Select the AI model to use for chat and ADQL generation"
        )
        
        selected_key = model_options[selected_display]
        selected_model = available_models[selected_key]
        
        # Show model details
        st.info(f"**Provider:** {selected_model['provider'].title()}\n"
                f"**Model:** {selected_model['model_name']}\n"
                f"**Description:** {selected_model['description']}")
        
        return selected_key


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
    
    def chat_completion(self, messages, max_tokens, temperature):
        """Generate chat completion regardless of provider."""
        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return response.choices[0].message.content
            
            elif self.provider == "anthropic":
                # Convert messages for Anthropic
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
                
                kwargs = {
                    "model": self.model_name,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "messages": anthropic_messages
                }
                
                if system_message:
                    kwargs["system"] = system_message
                
                response = self.client.messages.create(**kwargs)
                return response.content[0].text
            
        except Exception as e:
            raise Exception(f"{self.provider} API error: {str(e)}")
    
    def get_provider_name(self) -> str:
        return self.provider