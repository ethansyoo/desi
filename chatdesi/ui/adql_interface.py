"""
ADQL interface for chatDESI.
"""

from typing import Optional
import pandas as pd
import re # Added for parsing SQL from response

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

from .components import UIComponents, DataComponents, SessionManager
from ..data import ADQLManager, ADQLGenerator
from ..config import settings
from ..utils import ErrorHandler, PerformanceMonitor # Added for decorators


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
        
        df_reference = self.data_components.load_reference_data()
        if df_reference is not None:
            self.data_components.render_reference_data_expander(df_reference)
        
        user_query_nl = st.text_area(
            "Describe your ADQL query in natural language:",
            height=100,
            key="adql_nl_input"
        )
        
        sql_query_input = st.text_area(
            "ADQL Query",
            value=st.session_state["adql_query"],
            height=100,
            key="adql_query_box"
        )
        
        self._render_action_buttons(user_query_nl, df_reference, token_limit, temp_val)
        self._render_execution_section(sql_query_input, user_query_nl, max_records)
        self._render_feedback_section()
        self._render_history_section()
    
    def _render_action_buttons(self, user_query_nl, df_reference, token_limit, temp_val):
        """Renders action buttons for the ADQL interface."""
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Generate ADQL Query"):
                if user_query_nl:
                    self._handle_generate_query(user_query_nl, df_reference, token_limit, temp_val)
                else:
                    st.warning("Please enter a natural language query.")
        # Retry functionality can be added here if needed

    def _handle_generate_query(self, user_query_nl, df_reference, token_limit, temp_val):
        """Handles the logic for generating an ADQL query."""
        raise NotImplementedError

    def _render_execution_section(self, sql_query_input, user_query_nl, max_records):
        """Renders the query execution section."""
        if st.button("Run Query and Graph Data"):
            self._execute_adql_query(sql_query_input, user_query_nl, max_records)
    
    def _execute_adql_query(self, query, user_query_nl, max_records):
        """Executes the ADQL query."""
        # Logic for executing the query...
        pass
    
    def _render_feedback_section(self):
        """Renders feedback buttons."""
        # Feedback logic...
        pass
        
    def _render_history_section(self):
        """Renders the query history."""
        # History display logic...
        pass

# --- Refactored Implementation Classes ---

class PracticalADQLGenerator:
    """ADQL generator using universal client."""
    def __init__(self, ai_client, adql_manager):
        self.ai_client = ai_client
        self.adql_manager = adql_manager
    
    @st.cache_data(show_spinner=False)
    @ErrorHandler.handle_api_errors
    @PerformanceMonitor.time_function(show_in_sidebar=False)
    def generate_adql_query(self, user_input, available_columns, conversation_history=None, temperature=None, max_tokens=None):
        system_prompt = (
            "You are a helpful assistant that converts natural language queries into ADQL. "
            "Return only the SQL query inside a code block (```sql ... ```) and nothing else. "
            "Follow ADQL format strictly.\n\n"
            "Important rules:\n"
            "- ADQL does NOT support the `LIMIT` clause.\n\n"
            f"Available columns: {available_columns}"
        )
        
        messages = [{"role": "system", "content": system_prompt}]
        
        rl_context = self.adql_manager.find_similar_adql_queries(user_input, top_k=3)
        if rl_context["positive"]:
            pos_examples = "\n\n".join([f"NL: {doc['user_query']}\nADQL:\n{doc['generated_adql']}" for doc in rl_context["positive"]])
            messages.append({"role": "system", "content": f"Here are good ADQL examples:\n\n{pos_examples}"})
        
        if conversation_history: messages.extend(conversation_history[-4:])
        
        messages.append({"role": "user", "content": user_input})
        
        response = self.ai_client.chat_completion(
            messages=messages,
            max_tokens=max_tokens or settings.model.default_token_limit,
            temperature=temperature or settings.model.default_temperature
        )
        
        match = re.search(r"```sql\s*(.*?)\s*```", response, re.DOTALL)
        return match.group(1).strip() if match else response.strip()

class PracticalADQLInterface(ADQLInterface):
    """ADQL interface using the practical AI client."""
    def __init__(self, adql_manager, ai_client):
        super().__init__(adql_manager, ai_client)
        self.adql_generator = PracticalADQLGenerator(ai_client, adql_manager)
    
    def _handle_generate_query(self, user_query_nl, df_reference, token_limit, temp_val):
        if df_reference is None:
            st.error("Reference data is not available.")
            return
        
        available_columns = ", ".join(df_reference.columns)
        conversation_history = st.session_state.get("adql_history", [])
        
        with st.spinner("Generating ADQL query..."):
            generated_query = self.adql_generator.generate_adql_query(
                user_query_nl, available_columns, conversation_history, temp_val, token_limit
            )
        
        if generated_query:
            st.session_state["adql_query"] = generated_query
            st.session_state["last_adql_query"] = generated_query
            st.session_state["adql_history"].append({"role": "user", "content": user_query_nl})
            st.session_state["adql_history"].append({"role": "assistant", "content": generated_query})
            st.rerun()