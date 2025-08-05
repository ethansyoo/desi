"""
ADQL interface for chatDESI.
"""

from typing import Optional
import pandas as pd

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

from .components import UIComponents, DataComponents, SessionManager
from ..data import ADQLManager, ADQLGenerator
from ..config import settings


class ADQLInterface:
    """Manages the ADQL mode interface."""
    
    def __init__(self, adql_manager: ADQLManager, openai_client):
        self.adql_manager = adql_manager
        self.openai_client = openai_client
        self.adql_generator = ADQLGenerator(openai_client, adql_manager)
        self.ui = UIComponents()
        self.data_components = DataComponents()
        self.session = SessionManager()
    
    def render(self, token_limit: int, temp_val: float, max_records: int):
        """Render the complete ADQL interface."""
        if not STREAMLIT_AVAILABLE:
            return
        
        self.session.initialize_adql_session()
        
        st.write("### ADQL Query Builder")
        
        # Load and display reference data
        df_reference = self.data_components.load_reference_data()
        if df_reference is not None:
            self.data_components.render_reference_data_expander(df_reference)
        
        # Natural language input
        user_query_nl = st.text_area(
            "Describe your ADQL query in natural language:",
            height=100,
            key="adql_nl_input"
        )
        
        # ADQL query display/edit box
        sql_query_input = st.text_area(
            "ADQL Query",
            value=st.session_state["adql_query"],
            height=100,
            key="adql_query_box"
        )
        
        # Action buttons
        self._render_action_buttons(user_query_nl, df_reference, token_limit, temp_val)
        
        # Query execution button
        self._render_execution_section(sql_query_input, user_query_nl, max_records)
        
        # Feedback buttons
        self._render_feedback_section()
        
        # History section
        self._render_history_section()
    
    def _render_action_buttons(self, user_query_nl: str, df_reference: pd.DataFrame, token_limit: int, temp_val: float):
        """Render ADQL generation and retry buttons."""
        col1, col2 = st.columns([1, 1])
        
        with col1:
            generate_query = st.button("Generate ADQL Query")
        
        with col2:
            retry_query = st.button("Retry Last Query")
        
        # Handle generate query
        if generate_query:
            if user_query_nl:
                self._handle_generate_query(user_query_nl, df_reference, token_limit, temp_val)
            else:
                st.warning("Please enter a natural language query.")
        
        # Handle retry query
        if retry_query:
            self._handle_retry_query(token_limit, temp_val)
    
    def _handle_generate_query(self, user_query_nl: str, df_reference: pd.DataFrame, token_limit: int, temp_val: float):
        """Handle ADQL query generation."""
        if df_reference is None:
            st.error("Reference data is not available. Cannot generate ADQL query.")
            return
        
        available_columns = ", ".join(df_reference.columns)
        conversation_history = st.session_state.get("adql_history", [])
        
        generated_query = self.adql_generator.generate_adql_query(
            user_query_nl,
            available_columns,
            conversation_history,
            temp_val,
            token_limit
        )
        
        if generated_query:
            st.session_state["adql_query"] = generated_query
            st.session_state["last_adql_query"] = generated_query
            st.session_state["adql_history"].append({"role": "user", "content": user_query_nl})
            st.session_state["adql_history"].append({"role": "assistant", "content": generated_query})
            st.rerun()
    
    def _handle_retry_query(self, token_limit: int, temp_val: float):
        """Handle retry of last ADQL query."""
        if not st.session_state.get("last_adql_query"):
            st.warning("No previous query to retry.")
            return
        
        retry_message = f"Retrying last ADQL query: {st.session_state['last_adql_query']}"
        st.session_state["adql_history"].append({"role": "user", "content": retry_message})
        
        try:
            response = self.openai_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Improve the last ADQL query."},
                    {"role": "user", "content": retry_message}
                ],
                model=settings.model.openai_model,
                max_tokens=token_limit,
                temperature=temp_val,
            )
            
            improved_query = response.choices[0].message.content
            st.session_state["adql_query"] = improved_query
            st.session_state["adql_history"].append({"role": "assistant", "content": improved_query})
            st.rerun()
            
        except Exception as e:
            st.error(f"Error improving query: {e}")
    
    def _render_execution_section(self, sql_query_input: str, user_query_nl: str, max_records: int):
        """Render query execution section."""
        if st.button("Run Query and Graph Data"):
            self._execute_adql_query(sql_query_input, user_query_nl, max_records)
    
    def _execute_adql_query(self, query: str, user_query_nl: str, max_records: int):
        """Execute ADQL query against TAP service."""
        if not query:
            st.warning("Please generate or enter an ADQL query first.")
            return
        
        # Update session state with current query
        st.session_state["adql_query"] = query
        
        # Build TAP query URL
        tap_query_url = settings.get_tap_query_url(query, max_records)
        
        with st.spinner(f"Fetching up to {max_records} rows from TAP service..."):
            df = self.data_components.download_tap_data(tap_query_url)
        
        if df is not None:
            # Log successful query
            entry_id = self.adql_manager.log_adql_query(
                user_query_nl,
                query,
                execution_success=True,
                tap_result_rows=len(df)
            )
            
            # Update session state
            st.session_state["tap_data"] = df
            st.session_state["tap_data_updated"] = True
            st.session_state["last_adql_doc_id"] = entry_id
            st.session_state["last_adql_user_query"] = user_query_nl
            st.session_state["last_adql_generated_query"] = query
            st.session_state["show_feedback_buttons"] = True
            
            st.success(f"Data successfully retrieved! Showing up to {max_records} results.")
            st.write("### TAP Query Result Data:")
            st.dataframe(df)
            
        else:
            # Log failed query
            self.adql_manager.log_adql_query(
                user_query_nl,
                query,
                execution_success=False,
                tap_result_rows=0
            )
            st.error("Failed to retrieve data. Please check the query or try again.")
    
    def _render_feedback_section(self):
        """Render feedback buttons section."""
        if not st.session_state.get("show_feedback_buttons"):
            return
        
        if not all([
            st.session_state.get("last_adql_doc_id"),
            st.session_state.get("last_adql_user_query"),
            st.session_state.get("last_adql_generated_query")
        ]):
            return
        
        def on_positive_feedback():
            success = self.adql_manager.update_feedback(
                st.session_state["last_adql_doc_id"],
                "positive"
            )
            if success:
                st.toast("Thanks for the feedback!")
                st.session_state["show_feedback_buttons"] = False
            else:
                st.error("Failed to save feedback.")
        
        def on_negative_feedback():
            success = self.adql_manager.update_feedback(
                st.session_state["last_adql_doc_id"],
                "negative"
            )
            if success:
                st.toast("Got it — we'll use that to improve.")
                st.session_state["show_feedback_buttons"] = False
            else:
                st.error("Failed to save feedback.")
        
        self.ui.render_feedback_buttons(on_positive_feedback, on_negative_feedback)
    
    def _render_history_section(self):
        """Render ADQL history section."""
        # Display ADQL history
        with st.expander("View ADQL Query History", expanded=False):
            for entry in st.session_state.get("adql_history", []):
                if entry["role"] == "user":
                    st.markdown(f"**User:** {entry['content']}")
                else:
                    st.code(entry["content"], language="sql")
        
        # Clear history button
        if st.button("Clear ADQL History"):
            self.session.clear_adql_history()
            st.success("ADQL history cleared!")