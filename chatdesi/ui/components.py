"""
Reusable UI components for chatDESI Streamlit interface.
"""

import re
from typing import List, Dict, Any, Optional, Callable
import pandas as pd
import requests

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    st = None

from ..config import settings


class UIComponents:
    """Collection of reusable UI components."""
    
    @staticmethod
    def render_sidebar_settings() -> Dict[str, Any]:
        """
        Render the settings sidebar and return user selections.
        
        Returns:
            Dict containing user settings
        """
        if not STREAMLIT_AVAILABLE:
            return {}
        
        st.sidebar.write('### Settings')
        
        mode = st.sidebar.radio("Select Mode", ["Chat Mode", "ADQL Mode"])
        
        token_limit = st.sidebar.number_input(
            label="Token Limit", 
            min_value=500, 
            max_value=3000, 
            step=100, 
            value=settings.model.default_token_limit
        )
        
        temp_val = st.sidebar.slider(
            label="Temperature", 
            min_value=0.0, 
            max_value=1.5, 
            value=settings.model.default_temperature, 
            step=0.1
        )
        
        reference_toggle = st.sidebar.checkbox('Reference Papers', value=True)
        
        # ADQL-specific settings
        max_records = None
        if mode == "ADQL Mode":
            max_records = st.sidebar.number_input(
                "Set Max Rows (MAXREC)", 
                min_value=100, 
                max_value=settings.ui.max_max_records, 
                step=100, 
                value=settings.ui.default_max_records
            )
        
        return {
            "mode": mode,
            "token_limit": token_limit,
            "temperature": temp_val,
            "reference_toggle": reference_toggle,
            "max_records": max_records
        }
    
    @staticmethod
    def render_relevant_documents_sidebar(relevant_docs: List[Dict[str, Any]]):
        """Render relevant documents in sidebar."""
        if not STREAMLIT_AVAILABLE or not relevant_docs:
            return
        
        st.sidebar.write("## Relevant Documents")
        
        for doc in relevant_docs:
            filename = doc["metadata"].get("filename", "Unnamed")
            similarity = doc.get("similarity", None)
            score_str = f" — Similarity: {similarity:.2f}" if similarity is not None else ""
            
            text_chunk = doc["text"]
            
            with st.sidebar.expander(f"{filename}{score_str}", expanded=False):
                st.markdown(text_chunk, unsafe_allow_html=True)
    
    @staticmethod
    def render_chat_input() -> Optional[str]:
        """Render chat input field and return user input."""
        if not STREAMLIT_AVAILABLE:
            return None
        
        return st.text_input("Enter your message:", key="chat_input")
    
    @staticmethod
    def render_action_buttons() -> Dict[str, bool]:
        """
        Render action buttons and return which ones were clicked.
        
        Returns:
            Dict with button states
        """
        if not STREAMLIT_AVAILABLE:
            return {}
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            send_query = st.button("Send")
        with col2:
            clear_query = st.button("Clear Chat History")
        with col3:
            retry_query = st.button("Retry")
        
        return {
            "send": send_query,
            "clear": clear_query,
            "retry": retry_query
        }
    
    @staticmethod
    def render_chat_history(history: List[Dict[str, str]], expanded: bool = False):
        """Render expandable chat history."""
        if not STREAMLIT_AVAILABLE or not history:
            return
        
        with st.expander("View Full Chat History", expanded=expanded):
            for chat in history:
                if chat["role"] == "user":
                    st.markdown(f"**User:** {chat['content']}")
                else:
                    st.code(chat['content'], language="markdown")
    
    @staticmethod
    def render_feedback_buttons(on_positive: Callable = None, on_negative: Callable = None) -> Dict[str, bool]:
        """
        Render feedback buttons.
        
        Args:
            on_positive: Callback for positive feedback
            on_negative: Callback for negative feedback
            
        Returns:
            Dict with button states
        """
        if not STREAMLIT_AVAILABLE:
            return {}
        
        st.write("#### Was this query helpful?")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            positive_clicked = st.button("👍 Mark as Helpful")
            if positive_clicked and on_positive:
                on_positive()
        
        with col2:
            negative_clicked = st.button("👎 Not Helpful")
            if negative_clicked and on_negative:
                on_negative()
        
        return {
            "positive": positive_clicked,
            "negative": negative_clicked
        }
    
    @staticmethod
    def render_footer():
        """Render application footer."""
        if not STREAMLIT_AVAILABLE:
            return
        
        footer = f"""
        <style>
        .feedback-footer {{
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            text-align: center;
            opacity: 0.6;
            font-size: 14px;
            padding: 5px;
        }}
        </style>
        <div class="feedback-footer">
            Beta testing feedback: <a href="{settings.external.feedback_form_url}" target="_blank">Click here</a>
        </div>
        """
        st.markdown(footer, unsafe_allow_html=True)


class DataComponents:
    """Components for handling data display and processing."""
    
    @staticmethod 
    def load_reference_data() -> Optional[pd.DataFrame]:
        """Load reference CSV data with caching."""
        if not STREAMLIT_AVAILABLE:
            return None
        
        @st.cache_data
        def _load_data():
            try:
                df = pd.read_csv(
                    settings.external.github_csv_url,
                    sep=",",
                    encoding="utf-8",
                    on_bad_lines="skip"
                )
                return df
            except Exception as e:
                if st:
                    st.error(f"Error loading reference data: {e}")
                return None
        
        return _load_data()
    
    @staticmethod
    def render_reference_data_expander(df: pd.DataFrame):
        """Render expandable reference data view."""
        if not STREAMLIT_AVAILABLE or df is None:
            return
        
        with st.expander("📖 Reference Data for ADQL Queries", expanded=False):
            st.dataframe(df)
    
    @staticmethod
    def download_tap_data(query_url: str) -> Optional[pd.DataFrame]:
        """Download and validate TAP query results."""
        try:
            response = requests.get(query_url, timeout=60)
            response.raise_for_status()
            
            if response.text.strip().startswith("<VOTABLE"):
                if st:
                    st.error("TAP service returned an XML response instead of CSV. Check the query format.")
                return None
            
            if not response.text.strip():
                if st:
                    st.error("The TAP service returned an empty response. The query might be incorrect.")
                return None
            
            if "ERROR" in response.text[:100].upper():
                if st:
                    st.error(f"The TAP service returned an error: {response.text[:300]}")
                return None
            
            # Save and load as CSV
            with open("tap_query_result.csv", "w", encoding="utf-8") as file:
                file.write(response.text)
            
            df = pd.read_csv("tap_query_result.csv", sep=",", on_bad_lines="skip")
            
            if df.empty:
                if st:
                    st.error("The downloaded CSV is empty. Please check the query.")
                return None
            
            if len(df.columns) < 1:
                if st:
                    st.error("The CSV file has an unexpected format. Check the TAP service output.")
                return None
            
            return df
            
        except requests.exceptions.RequestException as e:
            if st:
                st.error(f"Network error while fetching TAP query result: {e}")
            return None
        except pd.errors.ParserError as e:
            if st:
                st.error(f"Failed to parse CSV response: {e}")
            return None
        except Exception as e:
            if st:
                st.error(f"Unexpected error: {e}")
            return None


class MathRenderer:
    """Utilities for rendering mathematical content."""
    
    @staticmethod
    def render_latex_from_response(response_text: str):
        """
        Automatically detects and renders LaTeX-style math and markdown from OpenAI responses.
        Supports $$...$$ blocks, $...$ inline, ```math blocks, and fallback markdown.
        """
        if not STREAMLIT_AVAILABLE:
            print(response_text)
            return
        
        # Handle ```math ... ``` blocks first
        response_text = re.sub(r"```math(.*?)```", r"$$\1$$", response_text, flags=re.DOTALL)
        
        # Split line by line for precise rendering
        for line in response_text.split("\n"):
            stripped = line.strip()
            
            if stripped.startswith("$$") and stripped.endswith("$$"):
                # Block LaTeX
                st.latex(stripped.strip("$$"))
            elif re.search(r"\$(.+?)\$", stripped):
                # Inline LaTeX → convert $...$ to \( ... \) for Streamlit markdown
                converted = re.sub(r"\$(.+?)\$", r"\\(\1\\)", line)
                st.markdown(converted)
            elif stripped.startswith("```") and stripped.endswith("```"):
                # Handle code blocks as code
                st.code(stripped.strip("`"), language="python")
            else:
                st.markdown(line)
    
    @staticmethod
    def parse_and_render_content(label: str, content: str):
        """Parse and render content with code blocks."""
        if not STREAMLIT_AVAILABLE:
            print(f"{label}: {content}")
            return
        
        st.markdown(f"**{label}:**")
        parts = re.split(r'```', content)
        
        for i, part in enumerate(parts):
            if i % 2 == 0:
                if part.strip():
                    st.markdown(part.strip())
            else:
                st.code(part.strip(), language='python')


class SessionManager:
    """Manages Streamlit session state."""
    
    @staticmethod
    def initialize_chat_session():
        """Initialize chat-related session state."""
        if not STREAMLIT_AVAILABLE:
            return
        
        defaults = {
            "history": [],
            "last_response": "",
            "last_query": "",
            "relevant_docs": []
        }
        
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    @staticmethod
    def initialize_adql_session():
        """Initialize ADQL-related session state."""
        if not STREAMLIT_AVAILABLE:
            return
        
        defaults = {
            "adql_query": "",
            "adql_history": [],
            "last_adql_query": "",
            "tap_data": None,
            "tap_data_updated": False,
            "last_adql_doc_id": None,
            "last_adql_user_query": "",
            "last_adql_generated_query": "",
            "show_feedback_buttons": False
        }
        
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    @staticmethod
    def clear_chat_history():
        """Clear chat history from session state."""
        if not STREAMLIT_AVAILABLE:
            return
        
        st.session_state["history"] = []
        st.session_state["last_query"] = ""
        st.session_state["last_response"] = ""
        st.session_state["relevant_docs"] = []
    
    @staticmethod
    def clear_adql_history():
        """Clear ADQL history from session state."""
        if not STREAMLIT_AVAILABLE:
            return
        
        st.session_state["adql_history"] = []
        st.session_state["adql_query"] = ""
        st.session_state["last_adql_query"] = ""