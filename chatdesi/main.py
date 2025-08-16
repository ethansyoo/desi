"""
Main application entry point for chatDESI with role-based access.
"""
import sys
import os
import streamlit as st

# Handle both relative and absolute imports
try:
    from .auth.api_key_manager import APIKeyManager, SimpleModelClient
    from .data import DatabaseFactory, PDFManager, ADQLManager, PDFProcessor
    from .ui import ChatInterface, ADQLInterface, UIComponents
    from .utils import ErrorHandler, PerformanceMonitor
    from .config import settings
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from chatdesi.auth.api_key_manager import APIKeyManager, SimpleModelClient
    from chatdesi.data import DatabaseFactory, PDFManager, ADQLManager, PDFProcessor
    from chatdesi.ui import ChatInterface, ADQLInterface, UIComponents
    from chatdesi.utils import ErrorHandler, PerformanceMonitor
    from chatdesi.config import settings

def main():
    """Main application function."""
    st.set_page_config(
        page_title="chatDESI - Multi-Model",
        page_icon="🔭",
        layout="wide"
    )

    # Initialize session state for admin login
    if 'admin_logged_in' not in st.session_state:
        st.session_state['admin_logged_in'] = False

    # Admin Login UI in the sidebar
    render_admin_login()

    # Determine which database connection string to use
    try:
        if st.session_state['admin_logged_in']:
            st.sidebar.success("👑 Admin Mode Activated")
            connection_string = st.secrets["mongo_admin_connection_string"]
        else:
            connection_string = st.secrets["mongo_general_connection_string"]
    except KeyError:
        st.error("Database connection string not found in secrets.toml. Please check your configuration.")
        st.stop()

    # Step 2: API Key Input and Validation
    api_key_manager = APIKeyManager()
    api_keys = api_key_manager.get_user_api_keys()

    if not api_keys:
        st.warning("Please enter at least one valid API key to proceed.")
        st.stop()
    
    available_models = api_key_manager.get_available_models(api_keys)
    selected_model_key = api_key_manager.render_model_selector(available_models)
    
    if not selected_model_key:
        st.stop()

    # Step 3: Initialize Services
    try:
        with st.spinner("Connecting to database..."):
            db_manager = DatabaseFactory.create_from_connection_string(connection_string)
            if not db_manager.test_connection():
                st.error("❌ Unable to connect to MongoDB. Check credentials and IP Access List.")
                return
            st.toast("✅ Connected to MongoDB") # <-- CHANGE HERE

        selected_model_info = available_models[selected_model_key]
        provider = selected_model_info["provider"]
        model_name = selected_model_info["model_name"]
        api_key = api_keys[provider]

        with st.spinner(f"Initializing {selected_model_info['display_name']}..."):
            ai_client = SimpleModelClient(provider, api_key, model_name)
            st.toast(f"✅ Connected to {selected_model_info['display_name']}") # <-- AND CHANGE HERE

        pdf_manager = PDFManager(db_manager)
        adql_manager = ADQLManager(db_manager)
        
        render_main_interface(pdf_manager, adql_manager, ai_client)

    except Exception as e:
        st.error(f"❌ Application error: {e}")
        with st.expander("Debug Details"):
            import traceback
            st.code(traceback.format_exc())

def render_admin_login():
    """Renders the admin login form in the sidebar."""
    if st.session_state.get('admin_logged_in'):
        if st.sidebar.button("Logout Admin"):
            st.session_state['admin_logged_in'] = False
            st.rerun()
    else:
        with st.sidebar.expander("Admin Login"):
            password = st.text_input("Enter Admin Password", type="password", key="admin_password_input")
            if st.button("Login"):
                if password == st.secrets.get("app_admin_password"):
                    st.session_state['admin_logged_in'] = True
                    st.rerun()
                else:
                    st.error("Incorrect password.")

def render_admin_panel(pdf_manager: PDFManager):
    """Renders the PDF management panel in the sidebar."""
    st.sidebar.write("---")
    st.sidebar.header("👑 Admin Panel")

    # PDF Upload
    st.sidebar.subheader("Upload New Document")
    uploaded_file = st.sidebar.file_uploader("Select a PDF", type="pdf", key="pdf_uploader")
    if uploaded_file:
        with st.spinner("Processing PDF..."):
            text = PDFProcessor.extract_text_from_pdf(uploaded_file.getvalue())
            result = pdf_manager.add_pdf_to_db(text, uploaded_file.name)
            st.sidebar.success(f"Status: {result['message']}")

    # PDF Deletion
    st.sidebar.subheader("Manage Existing Documents")
    try:
        doc_stats = pdf_manager.get_document_stats()
        st.sidebar.caption(f"Found **{doc_stats['unique_documents']}** unique documents.")
        if doc_stats["filenames"]:
            for filename in sorted(doc_stats["filenames"]):
                col1, col2 = st.sidebar.columns([3, 1])
                col1.write(filename)
                if col2.button("Delete", key=f"del_{filename}"):
                    pdf_manager.delete_document(filename)
                    st.rerun()
    except Exception as e:
        st.sidebar.error(f"Could not retrieve document list: {e}")

@PerformanceMonitor.time_function()
def render_main_interface(pdf_manager, adql_manager, ai_client):
    """Render the main interface with performance monitoring."""
    col_left, col_right = st.columns([4, 1])
    
    with col_right:
        ui_components = UIComponents()
        user_settings = ui_components.render_sidebar_settings()
        
        st.sidebar.write("### 🤖 Current Model")
        st.sidebar.info(f"**{ai_client.get_provider_name().title()}**\n{ai_client.model_name}")
        
        st.sidebar.write("### ⚡ Performance")
        st.sidebar.caption("Function timing appears below")
        
        if st.session_state.get('admin_logged_in'):
            render_admin_panel(pdf_manager)
    
    with col_left:
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
        context = ""
        if relevant_docs:
            context_snippets = "\n\n".join([doc["text"] for doc in relevant_docs[:3]])
            context = f"Relevant document context:\n\n{context_snippets}"
        
        messages = []
        if context: messages.append({"role": "system", "content": context})
        
        max_history_tokens = 800
        token_count = len(context) // 4
        
        chat_history = st.session_state["history"][::-1]
        history_trimmed = []
        
        for entry in chat_history:
            est_tokens = len(entry["content"]) // 4
            if token_count + est_tokens > max_history_tokens: break
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
        if df_reference is None:
            st.error("Reference data is not available.")
            return
        
        available_columns = ", ".join(df_reference.columns)
        conversation_history = st.session_state.get("adql_history", [])
        
        generated_query = self.adql_generator.generate_adql_query(
            user_query_nl, available_columns, conversation_history, temp_val, token_limit
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
        
        import re
        match = re.search(r"```sql\s*(.*?)\s*```", response, re.DOTALL)
        return match.group(1).strip() if match else response.strip()

def run_streamlit_app():
    """Entry point for running the Streamlit app."""
    main()

if __name__ == "__main__":
    run_streamlit_app()