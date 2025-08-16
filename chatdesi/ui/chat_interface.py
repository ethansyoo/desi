"""
Chat interface for chatDESI.
"""

from typing import List, Dict, Any, Optional
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

from .components import UIComponents, MathRenderer, SessionManager
from ..data import PDFManager
from ..config import settings
from ..utils import ErrorHandler, PerformanceMonitor


class ChatInterface:
    """Manages the chat mode interface."""
    
    def __init__(self, pdf_manager: PDFManager, ai_client):
        self.pdf_manager = pdf_manager
        self.ai_client = ai_client
        self.ui = UIComponents()
        self.renderer = MathRenderer()
        self.session = SessionManager()
    
    def render(self, reference_toggle: bool, token_limit: int, temp_val: float):
        """Render the complete chat interface."""
        self.session.initialize_chat_session()
        st.write("### chatDESI")

        # Display chat history
        for msg in st.session_state.history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # User input
        if prompt := st.chat_input("Enter your message"):
            self._handle_new_message(prompt, reference_toggle, token_limit, temp_val)

        # Sidebar actions
        with st.sidebar:
            if st.button("Clear Chat History"):
                self.session.clear_chat_history()
                st.rerun()

    def _handle_new_message(self, user_input: str, reference_toggle: bool, token_limit: int, temp_val: float):
        """Handle a new user message."""
        st.session_state.history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        relevant_docs = []
        if reference_toggle:
            with st.spinner("Finding relevant documents..."):
                relevant_docs = self.pdf_manager.find_relevant_docs(user_input, top_k=3)
            st.session_state["relevant_docs"] = relevant_docs

        with st.chat_message("assistant"):
            # Use st.write_stream to render the streaming response
            response_generator = self._generate_chat_response(
                user_input, relevant_docs, token_limit, temp_val
            )
            full_response = st.write_stream(response_generator)

        st.session_state.history.append({"role": "assistant", "content": full_response})
        # No st.rerun() is needed here; st.write_stream handles the display updates.

    def _generate_chat_response(self, user_input, relevant_docs, token_limit, temp_val):
        """Placeholder for the actual response generation logic."""
        raise NotImplementedError

# --- Implementation Class ---

class PracticalChatInterface(ChatInterface):
    """Chat interface using the practical AI client."""

    @ErrorHandler.handle_api_errors
    @PerformanceMonitor.time_function(show_in_sidebar=False)
    def _generate_chat_response(self, user_input, relevant_docs, token_limit, temp_val):
        """Generates a chat response, now supporting streaming."""
        context = ""
        if relevant_docs:
            context_snippets = "\n\n".join([doc["text"] for doc in relevant_docs[:3]])
            context = f"Relevant document context:\n\n{context_snippets}"
        
        messages = []
        if context:
            messages.append({"role": "system", "content": context})
        
        # Include recent history
        for msg in st.session_state.history[-6:]:
            messages.append(msg)
        
        return self.ai_client.chat_completion(
            messages=messages,
            max_tokens=token_limit,
            temperature=temp_val,
            stream=True  # Enable streaming
        )