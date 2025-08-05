"""
Performance monitoring and caching improvements.
"""

import time
import hashlib
from typing import Dict, Any, Optional, Callable
from functools import wraps

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False


class PerformanceMonitor:
    """Monitor and display performance metrics."""
    
    @staticmethod
    def time_function(show_in_sidebar: bool = True):
        """Decorator to time function execution."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                
                execution_time = end_time - start_time
                
                # Show timing information
                if STREAMLIT_AVAILABLE and show_in_sidebar:
                    try:
                        if execution_time > 2.0:
                            st.sidebar.warning(f"⏱️ {func.__name__}: {execution_time:.2f}s")
                        else:
                            st.sidebar.success(f"⏱️ {func.__name__}: {execution_time:.2f}s")
                    except:
                        pass
                
                return result
            return wrapper
        return decorator
    
    @staticmethod
    def show_performance_metrics():
        """Show overall performance metrics in sidebar."""
        if not STREAMLIT_AVAILABLE:
            return
        
        with st.sidebar.expander("📊 Performance Metrics", expanded=False):
            # Show session metrics if available
            if "performance_metrics" in st.session_state:
                metrics = st.session_state["performance_metrics"]
                
                for metric_name, metric_value in metrics.items():
                    st.metric(metric_name, f"{metric_value:.2f}s")
            else:
                st.info("Performance metrics will appear here as you use the app.")


class CacheManager:
    """Enhanced caching for embeddings and API responses."""
    
    @staticmethod
    def get_cache_key(*args, **kwargs) -> str:
        """Generate cache key from arguments."""
        import json
        cache_input = json.dumps([str(arg) for arg in args] + [f"{k}:{v}" for k, v in sorted(kwargs.items())])
        return hashlib.md5(cache_input.encode()).hexdigest()
    
    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)  # Cache for 1 hour
    def cached_embedding_generation(text: str, model_name: str) -> list:
        """Cache embeddings to avoid recomputation."""
        from ..data.pdf_manager import EmbeddingModel
        
        model = EmbeddingModel(model_name)
        return model.embed_text(text).tolist()
    
    @staticmethod
    @st.cache_data(ttl=86400, show_spinner=False)  # Cache for 24 hours
    def cached_reference_data_load(url: str):
        """Cache reference data loading."""
        import pandas as pd
        return pd.read_csv(url, sep=",", encoding="utf-8", on_bad_lines="skip")
    
    @staticmethod
    @st.cache_data(ttl=1800, show_spinner=False)  # Cache for 30 minutes
    def cached_api_response(messages_hash: str, model: str, temperature: float, max_tokens: int):
        """Cache API responses for identical requests."""
        # This is a placeholder - actual implementation would need the API call
        # The real implementation would be in the AI client classes
        return None
    
    @staticmethod
    def clear_cache():
        """Clear all Streamlit caches."""
        if STREAMLIT_AVAILABLE:
            st.cache_data.clear()
            st.success("🗑️ Cache cleared successfully!")


class ConnectionPool:
    """Simple connection pooling for database operations."""
    _connections = {}
    _connection_times = {}
    
    @classmethod
    def get_connection(cls, connection_string: str):
        """Get or create database connection with pooling."""
        if connection_string not in cls._connections:
            from pymongo import MongoClient
            
            start_time = time.time()
            cls._connections[connection_string] = MongoClient(
                connection_string,
                maxPoolSize=10,
                minPoolSize=1,
                maxIdleTimeMS=30000,
                serverSelectionTimeoutMS=5000
            )
            cls._connection_times[connection_string] = time.time() - start_time
            
            if STREAMLIT_AVAILABLE:
                st.sidebar.success(f"🔗 DB Connected: {cls._connection_times[connection_string]:.2f}s")
        
        return cls._connections[connection_string]
    
    @classmethod
    def close_all_connections(cls):
        """Close all pooled connections."""
        for client in cls._connections.values():
            client.close()
        cls._connections.clear()
        cls._connection_times.clear()
        
        if STREAMLIT_AVAILABLE:
            st.sidebar.info("🔗 All connections closed")


class ResourceMonitor:
    """Monitor system resources and API usage."""
    
    @staticmethod
    def track_api_usage(provider: str, model: str, input_tokens: int = 0, output_tokens: int = 0):
        """Track API usage for cost monitoring."""
        if not STREAMLIT_AVAILABLE:
            return
        
        # Initialize session state for tracking
        if "api_usage" not in st.session_state:
            st.session_state["api_usage"] = {}
        
        usage_key = f"{provider}_{model}"
        if usage_key not in st.session_state["api_usage"]:
            st.session_state["api_usage"][usage_key] = {
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0
            }
        
        # Update usage
        st.session_state["api_usage"][usage_key]["calls"] += 1
        st.session_state["api_usage"][usage_key]["input_tokens"] += input_tokens
        st.session_state["api_usage"][usage_key]["output_tokens"] += output_tokens
    
    @staticmethod
    def show_api_usage_sidebar():
        """Show API usage in sidebar."""
        if not STREAMLIT_AVAILABLE:
            return
        
        if "api_usage" not in st.session_state or not st.session_state["api_usage"]:
            return
        
        with st.sidebar.expander("💰 API Usage", expanded=False):
            for usage_key, usage_data in st.session_state["api_usage"].items():
                provider, model = usage_key.split("_", 1)
                
                st.write(f"**{provider.title()} - {model}**")
                st.write(f"Calls: {usage_data['calls']}")
                st.write(f"Input tokens: {usage_data['input_tokens']:,}")
                st.write(f"Output tokens: {usage_data['output_tokens']:,}")
                st.write("---")
    
    @staticmethod
    def estimate_cost(provider: str, model: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate API cost (simplified pricing)."""
        # Simplified pricing - you'd want to update these with current rates
        pricing = {
            "openai": {
                "gpt-4o": {"input": 0.00025, "output": 0.001},
                "gpt-3.5-turbo": {"input": 0.0000015, "output": 0.000002}
            },
            "anthropic": {
                "claude-3-5-sonnet-20241022": {"input": 0.000003, "output": 0.000015},
                "claude-3-haiku-20240307": {"input": 0.00000025, "output": 0.00000125}
            }
        }
        
        if provider in pricing and model in pricing[provider]:
            rates = pricing[provider][model]
            cost = (input_tokens * rates["input"]) + (output_tokens * rates["output"])
            return cost
        
        return 0.0