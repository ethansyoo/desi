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
from ..data import PDFManager, ADQLManager
from ..config import settings


class ChatInterface:
    """Manages the chat mode interface."""
    
    def __init__(self, pdf_manager: PDFManager, openai_client):
        self.pdf_manager = pdf_manager
        self.openai_client = openai_client
        self.ui = UIComponents()
        self.renderer = MathRenderer()
        self.session = SessionManager()
    
    def render(self, reference_toggle: bool, token_limit: int, temp_val: float):
        """Render the complete chat interface."""
        if not STREAMLIT_AVAILABLE:
            return
        
        self.session.initialize_chat_session()
        
        st.write("### chatDESI")
        
        # User input
        user_input = self.ui.render_chat_input()
        
        # Action buttons
        buttons = self.ui.render_action_buttons()
        
        # Handle user interactions
        if buttons["send"] and user_input:
            self._handle_new_message(user_input, reference_toggle, token_limit, temp_val)
        
        if buttons["retry"]:
            self._handle_retry_message(reference_toggle, token_limit, temp_val)
        
        if buttons["clear"]:
            self.session.clear_chat_history()
            st.success("Chat history cleared!")
        
        # Display last response
        self._display_last_response()
        
        # Display chat history
        self.ui.render_chat_history(st.session_state["history"])
        
        # Display relevant documents in sidebar
        if "relevant_docs" in st.session_state:
            self.ui.render_relevant_documents_sidebar(st.session_state["relevant_docs"])
    
    def _handle_new_message(self, user_input: str, reference_toggle: bool, token_limit: int, temp_val: float):
        """Handle a new user message."""
        st.session_state["last_query"] = user_input
        st.session_state["history"].append({"role": "user", "content": user_input})
        
        # Retrieve relevant documents if enabled
        relevant_docs = []
        if reference_toggle:
            relevant_docs = self.pdf_manager.find_relevant_docs(user_input, top_k=3)
            st.session_state["relevant_docs"] = relevant_docs
        
        # Generate response
        try:
            response = self._generate_chat_response(
                user_input, relevant_docs, token_limit, temp_val
            )
            
            if response:
                st.session_state["history"].append({"role": "assistant", "content": response})
                st.session_state["last_response"] = response
                st.rerun()
            
        except Exception as e:
            st.error(f"Error generating response: {e}")
    
    def _handle_retry_message(self, reference_toggle: bool, token_limit: int, temp_val: float):
        """Handle retry of last message."""
        if not st.session_state.get("last_query"):
            st.warning("No previous query to retry.")
            return
        
        retry_message = (
            f"Previous query: {st.session_state['last_query']}. "
            f"Retry with improvements. Here was the response: {st.session_state['last_response']}."
        )
        
        st.session_state["history"].append({"role": "user", "content": retry_message})
        
        # Reuse previous relevant docs if available
        relevant_docs = st.session_state.get("relevant_docs", [])
        
        try:
            response = self._generate_retry_response(retry_message, relevant_docs, token_limit, temp_val)
            
            if response:
                st.session_state["history"].append({"role": "assistant", "content": response})
                st.session_state["last_response"] = response
                st.rerun()
            
        except Exception as e:
            st.error(f"Error generating retry response: {e}")
    
    def _generate_chat_response(
        self, 
        user_input: str, 
        relevant_docs: List[Dict[str, Any]], 
        token_limit: int, 
        temp_val: float
    ) -> Optional[str]:
        """Generate chat response using OpenAI."""
        
        # Build context from relevant documents
        context = ""
        if relevant_docs:
            context_snippets = "\n\n".join([doc["text"] for doc in relevant_docs[:3]])
            context = f"Relevant document context:\n\n{context_snippets}"
        
        # Build messages with token-safe history
        messages = []
        if context:
            messages.append({"role": "system", "content": context})
        
        # Add conversation history with token limiting
        max_history_tokens = 800
        token_count = len(context) // 4
        
        chat_history = st.session_state["history"][::-1]  # Reverse to get latest first
        history_trimmed = []
        
        for entry in chat_history:
            est_tokens = len(entry["content"]) // 4
            if token_count + est_tokens > max_history_tokens:
                break
            history_trimmed.insert(0, entry)  # Maintain original order
            token_count += est_tokens
        
        messages.extend(history_trimmed)
        
        # Generate response
        response = self.openai_client.chat.completions.create(
            messages=messages,
            model=settings.model.openai_model,
            max_tokens=token_limit,
            temperature=temp_val,
        )
        
        return response.choices[0].message.content
    
    def _generate_retry_response(
        self, 
        retry_message: str, 
        relevant_docs: List[Dict[str, Any]], 
        token_limit: int, 
        temp_val: float
    ) -> Optional[str]:
        """Generate retry response with improvement instruction."""
        
        # Build context
        reference_context = ""
        if relevant_docs:
            context_snippets = "\n\n".join([doc["text"] for doc in relevant_docs[:3]])
            reference_context = f"Relevant document context:\n\n{context_snippets}"
        
        messages = []
        if reference_context:
            messages.append({"role": "system", "content": reference_context})
        
        messages.append({
            "role": "system", 
            "content": "You are a helpful assistant. Improve the previous response, as it is not sufficient."
        })
        messages.append({"role": "user", "content": retry_message})
        
        response = self.openai_client.chat.completions.create(
            messages=messages,
            model=settings.model.openai_model,
            max_tokens=token_limit,
            temperature=temp_val,
        )
        
        return response.choices[0].message.content
    
    def _display_last_response(self):
        """Display the last assistant response."""
        if st.session_state.get("last_response"):
            st.write("### chatDESI")
            self.renderer.render_latex_from_response(st.session_state["last_response"])