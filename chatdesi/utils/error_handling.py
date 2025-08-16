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
        """
        Decorator for handling API-related errors.
        If an error occurs, it yields an error message string, making it
        compatible with st.write_stream.
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # This will now correctly yield from the generator
                # returned by the decorated function (e.g., _generate_chat_response)
                yield from func(*args, **kwargs)
            except Exception as e:
                # If an exception occurs in the stream, display the error
                # in the sidebar and yield a friendly message to the chat window.
                ErrorHandler._display_user_friendly_error(e)
                yield "Sorry, I encountered an error. Please check the sidebar for details."
        return wrapper
    
    @staticmethod
    def _display_user_friendly_error(error: Exception):
        """Display user-friendly error messages in the sidebar."""
        if not STREAMLIT_AVAILABLE:
            print(f"Error: {error}")
            return
        
        error_msg = str(error).lower()
        
        # Display errors in the sidebar to keep the chat clean
        if "rate limit" in error_msg or "quota" in error_msg:
            st.sidebar.error("⏱️ **Rate Limit Reached**\n\nThe API rate limit has been exceeded. Please wait a moment and try again.")
        elif "authentication" in error_msg or "api key" in error_msg or "unauthorized" in error_msg:
            st.sidebar.error("🔑 **Authentication Error**\n\nThere's an issue with your API key. Please check that it's correct and has sufficient credits.")
        elif "connection" in error_msg or "network" in error_msg or "timeout" in error_msg:
            st.sidebar.error("🌐 **Connection Error**\n\nUnable to connect to the AI service. Please check your internet connection and try again.")
        else:
            st.sidebar.error(f"❌ **Unexpected Error**\n\n{str(error)}")
        
        # Show debug details in an expander
        with st.sidebar.expander("🔍 Debug Details", expanded=False):
            st.code(traceback.format_exc())


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
                    
                    progress_placeholder.empty()
                    status_placeholder.empty()
                    return result
                else:
                    return func(*args, **kwargs)
                    
            except Exception as e:
                if attempt == max_retries - 1:
                    if STREAMLIT_AVAILABLE:
                        progress_placeholder.empty()
                        status_placeholder.empty()
                    ErrorHandler._display_user_friendly_error(e)
                    return None
                else:
                    if STREAMLIT_AVAILABLE:
                        status_placeholder.warning(f"⚠️ Attempt {attempt + 1} failed. Retrying in {delay}s...")
                    time.sleep(delay)
                    delay *= 1.5
        
        return None