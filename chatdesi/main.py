"""
Main application entry point for chatDESI with practical API key testing.
"""

import sys
import os

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

# Handle both relative and absolute imports
try:
    # We no longer need the old authentication system
    from .auth.api_key_manager import APIKeyManager, SimpleModelClient
    from .data import DatabaseFactory, PDFManager, ADQLManager
    from .ui import ChatInterface, ADQLInterface, UIComponents
    from .utils import ErrorHandler, PerformanceMonitor
    from .config import settings
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from chatdesi.auth.api_key_manager import APIKeyManager, SimpleModelClient
    from chatdesi.data import DatabaseFactory, PDFManager, ADQLManager
    from chatdesi.ui import ChatInterface, ADQLInterface, UIComponents
    from chatdesi.utils import ErrorHandler, PerformanceMonitor
    from chatdesi.config import settings


def main():
    """Main application function."""
    if not STREAMLIT_AVAILABLE:
        print("Error: Streamlit is required to run chatDESI.")
        return
    
    st.set_page_config(
        page_title="chatDESI - Multi-Model",
        page_icon="🔭",
        layout="wide"
    )
    
    # --- NEW: Step 1: Get MongoDB credentials from Streamlit Secrets ---
    try:
        mongo_user = st.secrets["mongo_username"]
        mongo_pass = st.secrets["mongo_password"]
    except KeyError:
        st.error("MongoDB credentials not found in Streamlit Secrets.")
        st.info("Please create a `.streamlit/secrets.toml` file with your `mongo_username` and `mongo_password`.")
        st.stop()
    
    # Step 2: API Key Input and Validation
    api_key_manager = APIKeyManager()
    api_keys = api_key_manager.get_user_api_keys()
    
    if not api_keys:
        st.warning("Please enter at least one valid API key to proceed.")
        st.stop()
    
    # Get available models and let user choose
    available_models = api_key_manager.get_available_models(api_keys)
    selected_model_key = api_key_manager.render_model_selector(available_models)
    
    if not selected_model_key:
        st.stop()
    
    # Step 3: Initialize Services
    try:
        # --- UPDATED: Database connection using credentials from Secrets ---
        with st.spinner("Connecting to database..."):
            db_manager = DatabaseFactory.create_from_credentials(mongo_user, mongo_pass)
            if not db_manager.test_connection():
                st.error("❌ Unable to connect to MongoDB. Check credentials in secrets.toml.")
                return
            st.success("✅ Connected to MongoDB")
        
        # AI Model setup
        selected_model_info = available_models[selected_model_key]
        provider = selected_model_info["provider"]
        model_name = selected_model_info["model_name"] 
        api_key = api_keys[provider]
        
        with st.spinner(f"Initializing {selected_model_info['display_name']}..."):
            ai_client = SimpleModelClient(provider, api_key, model_name)
            st.success(f"✅ Connected to {selected_model_info['display_name']}")
        
        # Initialize data managers
        pdf_manager = PDFManager(db_manager)
        adql_manager = ADQLManager(db_manager)
        
        # --- UPDATED: Render main interface without auth_system ---
        render_main_interface(pdf_manager, adql_manager, ai_client)
        
    except Exception as e:
        st.error(f"❌ Application error: {e}")
        with st.expander("Debug Details"):
            import traceback
            st.code(traceback.format_exc())


@PerformanceMonitor.time_function()
def render_main_interface(pdf_manager, adql_manager, ai_client):
    """Render the main interface with performance monitoring."""
    
    col_left, col_right = st.columns([4, 1])
    
    with col_right:
        ui_components = UIComponents()
        user_settings = ui_components.render_sidebar_settings()
        
        # Show current model
        st.sidebar.write("### 🤖 Current Model")
        st.sidebar.info(f"**{ai_client.get_provider_name().title()}**\n{ai_client.model_name}")
        
        # Performance section
        st.sidebar.write("### ⚡ Performance")
        st.sidebar.caption("Function timing appears below")
        
        # --- REMOVED: Logout button is no longer needed ---
    
    with col_left:
        # Create enhanced interfaces
        chat_interface = PracticalChatInterface(pdf_manager, ai_client)
        adql_interface = PracticalADQLInterface(adql_manager, ai_client)
        
        if user_settings["mode"] == "Chat Mode":
            chat_interface.render(
                reference_toggle=user_settings["reference_toggle"],
                token_limit=user_settings["token_limit"],
                temp_val=user_settings["temperature"]
            )
        elif user_settings["mode"] == "ADQL Mode":
            adql_interface.render(
                token_limit=user_settings["token_limit"],
                temp_val=user_settings["temperature"],
                max_records=user_settings["max_records"]
            )
    
    ui_components.render_footer()


class PracticalChatInterface(ChatInterface):
    """Chat interface using the practical AI client."""
    
    def __init__(self, pdf_manager, ai_client):
        self.pdf_manager = pdf_manager
        self.ai_client = ai_client
        from .ui.components import UIComponents, MathRenderer, SessionManager
        self.ui = UIComponents()
        self.renderer = MathRenderer()
        self.session = SessionManager()
    
    @ErrorHandler.handle_api_errors
    @PerformanceMonitor.time_function(show_in_sidebar=False)
    def _generate_chat_response(self, user_input, relevant_docs, token_limit, temp_val):
        """Generate chat response using universal client."""
        
        context = ""
        if relevant_docs:
            context_snippets = "\n\n".join([doc["text"] for doc in relevant_docs[:3]])
            context = f"Relevant document context:\n\n{context_snippets}"
        
        messages = []
        if context:
            messages.append({"role": "system", "content": context})
        
        max_history_tokens = 800
        token_count = len(context) // 4
        
        chat_history = st.session_state["history"][::-1]
        history_trimmed = []
        
        for entry in chat_history:
            est_tokens = len(entry["content"]) // 4
            if token_count + est_tokens > max_history_tokens:
                break
            history_trimmed.insert(0, entry)
            token_count += est_tokens
        
        messages.extend(history_trimmed)
        
        return self.ai_client.chat_completion(
            messages=messages,
            max_tokens=token_limit,
            temperature=temp_val
        )


class PracticalADQLInterface(ADQLInterface):
    """ADQL interface using the practical AI client."""
    
    def __init__(self, adql_manager, ai_client):
        self.adql_manager = adql_manager
        self.ai_client = ai_client
        self.adql_generator = PracticalADQLGenerator(ai_client, adql_manager)
        from .ui.components import UIComponents, DataComponents, SessionManager
        self.ui = UIComponents()
        self.data_components = DataComponents()
        self.session = SessionManager()
    
    def _handle_generate_query(self, user_query_nl, df_reference, token_limit, temp_val):
        """Generate ADQL query using universal client."""
        if df_reference is None:
            st.error("Reference data is not available.")
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


class PracticalADQLGenerator:
    """ADQL generator using universal client."""
    
    def __init__(self, ai_client, adql_manager):
        self.ai_client = ai_client
        self.adql_manager = adql_manager
    
    @ErrorHandler.handle_api_errors
    @PerformanceMonitor.time_function(show_in_sidebar=False)
    def generate_adql_query(self, user_input, available_columns, conversation_history=None, temperature=None, max_tokens=None):
        """Generate ADQL query using universal client."""
        
        system_prompt = (
            "You are a helpful assistant that converts natural language queries into ADQL "
            "(Astronomical Data Query Language). Return only the SQL query inside a code block "
            "(```sql ... ```) and nothing else. Avoid explanations, prefaces, or post-processing text. "
            "Follow ADQL format strictly.\n\n"
            "Important rules:\n"
            "- ADQL does NOT support the `LIMIT` clause.\n"
            "- Use BETWEEN or JOIN clauses appropriately.\n"
            "- Ensure the query is executable in a TAP service.\n\n"
            f"Available columns: {available_columns}"
        )
        
        messages = [{"role": "system", "content": system_prompt}]
        
        rl_context = self.adql_manager.find_similar_adql_queries(user_input, top_k=3)
        if rl_context["positive"]:
            pos_examples = "\n\n".join([
                f"NL: {doc['user_query']}\nADQL:\n{doc['generated_adql']}" 
                for doc in rl_context["positive"]
            ])
            messages.append({
                "role": "system",
                "content": f"Here are good ADQL examples:\n\n{pos_examples}"
            })
        
        if conversation_history:
            messages.extend(conversation_history[-4:])
        
        messages.append({"role": "user", "content": user_input})
        
        response = self.ai_client.chat_completion(
            messages=messages,
            max_tokens=max_tokens or settings.model.default_token_limit,
            temperature=temperature or settings.model.default_temperature
        )
        
        import re
        match = re.search(r"```sql\s*(.*?)\s*```", response, re.DOTALL)
        return match.group(1).strip() if match else response.strip()


def run_streamlit_app():
    """Entry point for running the Streamlit app."""
    main()


if __name__ == "__main__":
    run_streamlit_app()