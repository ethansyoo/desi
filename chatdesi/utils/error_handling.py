"""
Improved error handling and user feedback.
"""

import traceback
import time
from typing import Callable, Any, Optional
from functools import wraps

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False


class ErrorHandler:
    """Centralized error handling for better UX."""
    
    @staticmethod
    def handle_api_errors(func: Callable) -> Callable:
        """Decorator for handling API-related errors."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                ErrorHandler._display_user_friendly_error(e)
                return None
        return wrapper
    
    @staticmethod
    def _display_user_friendly_error(error: Exception):
        """Display user-friendly error messages."""
        if not STREAMLIT_AVAILABLE:
            print(f"Error: {error}")
            return
        
        error_msg = str(error).lower()
        
        if "rate limit" in error_msg or "quota" in error_msg:
            st.error("⏱️ **Rate Limit Reached**\n\n"
                    "The API rate limit has been exceeded. Please wait a moment and try again.")
            
        elif "authentication" in error_msg or "api key" in error_msg or "unauthorized" in error_msg:
            st.error("🔑 **Authentication Error**\n\n"
                    "There's an issue with your API key. Please check that it's correct and has sufficient credits.")
            
        elif "connection" in error_msg or "network" in error_msg or "timeout" in error_msg:
            st.error("🌐 **Connection Error**\n\n"
                    "Unable to connect to the AI service. Please check your internet connection and try again.")
            
        elif "model" in error_msg and "not found" in error_msg:
            st.error("🤖 **Model Error**\n\n"
                    "The selected AI model is not available. Please try a different model.")
            
        elif "token" in error_msg and ("limit" in error_msg or "maximum" in error_msg):
            st.error("📝 **Input Too Long**\n\n"
                    "Your input is too long for the selected model. Please try a shorter message or increase the token limit.")
            
        else:
            st.error(f"❌ **Unexpected Error**\n\n{str(error)}")
        
        # Show debug details in expander
        with st.expander("🔍 Debug Details", expanded=False):
            st.code(traceback.format_exc())
    
    @staticmethod
    def show_connection_status(provider: str, is_connected: bool):
        """Show connection status for a provider."""
        if not STREAMLIT_AVAILABLE:
            return
        
        if is_connected:
            st.success(f"✅ Connected to {provider.title()}")
        else:
            st.error(f"❌ Unable to connect to {provider.title()}")


class RobustExecutor:
    """Execute functions with retry logic and progress indicators."""
    
    @staticmethod
    def execute_with_retry(
        func: Callable,
        max_retries: int = 3,
        delay: float = 1.0,
        progress_message: str = "Processing...",
        *args,
        **kwargs
    ) -> Any:
        """Execute function with retry logic and progress indicator."""
        
        if STREAMLIT_AVAILABLE:
            progress_placeholder = st.empty()
            status_placeholder = st.empty()
        
        for attempt in range(max_retries):
            try:
                if STREAMLIT_AVAILABLE:
                    with progress_placeholder:
                        if attempt > 0:
                            st.info(f"🔄 Retry attempt {attempt + 1}/{max_retries}")
                        with st.spinner(progress_message):
                            result = func(*args, **kwargs)
                    
                    # Clear progress indicators on success
                    progress_placeholder.empty()
                    status_placeholder.empty()
                    return result
                else:
                    return func(*args, **kwargs)
                    
            except Exception as e:
                if attempt == max_retries - 1:
                    # Last attempt failed
                    if STREAMLIT_AVAILABLE:
                        progress_placeholder.empty()
                        status_placeholder.empty()
                    ErrorHandler._display_user_friendly_error(e)
                    return None
                else:
                    # Wait before retry
                    if STREAMLIT_AVAILABLE:
                        status_placeholder.warning(f"⚠️ Attempt {attempt + 1} failed. Retrying in {delay}s...")
                    time.sleep(delay)
                    delay *= 1.5  # Exponential backoff
        
        return None
    
    @staticmethod
    def execute_with_timeout(
        func: Callable,
        timeout_seconds: float = 30.0,
        progress_message: str = "Processing...",
        *args,
        **kwargs
    ) -> Any:
        """Execute function with timeout and cancellation."""
        import threading
        import queue
        
        result_queue = queue.Queue()
        exception_queue = queue.Queue()
        
        def target():
            try:
                result = func(*args, **kwargs)
                result_queue.put(result)
            except Exception as e:
                exception_queue.put(e)
        
        if STREAMLIT_AVAILABLE:
            progress_placeholder = st.empty()
            cancel_placeholder = st.empty()
            
            with progress_placeholder:
                with st.spinner(progress_message):
                    thread = threading.Thread(target=target)
                    thread.daemon = True
                    thread.start()
                    
                    # Wait for completion or timeout
                    thread.join(timeout=timeout_seconds)
                    
                    if thread.is_alive():
                        progress_placeholder.empty()
                        st.error(f"⏱️ Operation timed out after {timeout_seconds} seconds")
                        return None
                    
                    # Check for results
                    if not result_queue.empty():
                        progress_placeholder.empty()
                        return result_queue.get()
                    elif not exception_queue.empty():
                        progress_placeholder.empty()
                        ErrorHandler._display_user_friendly_error(exception_queue.get())
                        return None
        else:
            # Non-Streamlit fallback
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout=timeout_seconds)
            
            if not result_queue.empty():
                return result_queue.get()
            elif not exception_queue.empty():
                raise exception_queue.get()
            else:
                raise TimeoutError(f"Operation timed out after {timeout_seconds} seconds")
        
        return None