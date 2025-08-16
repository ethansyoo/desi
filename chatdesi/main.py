"""
Main application entry point for chatDESI with role-based access.
"""
import sys
import os
import streamlit as st
from pymongo.errors import ConnectionFailure, ConfigurationError

# Handle both relative and absolute imports
try:
    from .auth.api_key_manager import APIKeyManager, SimpleModelClient
    from .data import DatabaseFactory, PDFManager, ADQLManager, PDFProcessor, DatabaseManager
    from .ui import UIComponents
    from .ui.chat_interface import PracticalChatInterface
    from .ui.adql_interface import PracticalADQLInterface
    from .utils import PerformanceMonitor
    from .config import settings
except ImportError:
    # Fallback for running the script directly
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from chatdesi.auth.api_key_manager import APIKeyManager, SimpleModelClient
    from chatdesi.data import DatabaseFactory, PDFManager, ADQLManager, PDFProcessor, DatabaseManager
    from chatdesi.ui import UIComponents
    from chatdesi.ui.chat_interface import PracticalChatInterface
    from chatdesi.ui.adql_interface import PracticalADQLInterface
    from chatdesi.utils import PerformanceMonitor
    from chatdesi.config import settings

# --- Caching Functions for Performance ---

@st.cache_resource
def get_db_manager(connection_string: str) -> DatabaseManager:
    """Initializes and caches the database manager. No UI elements inside."""
    db_manager = DatabaseFactory.create_from_connection_string(connection_string)
    return db_manager

@st.cache_resource
def get_ai_client(provider: str, api_key: str, model_name: str) -> SimpleModelClient:
    """Initializes and caches the AI client. No UI elements inside."""
    client = SimpleModelClient(provider, api_key, model_name)
    return client

# --- Main Application Logic ---

def main():
    """Main application function."""
    st.set_page_config(
        page_title="chatDESI - Multi-Model",
        page_icon="🔭",
        layout="wide"
    )

    if 'admin_logged_in' not in st.session_state:
        st.session_state['admin_logged_in'] = False

    render_admin_login()

    try:
        connection_string = (st.secrets["mongo_admin_connection_string"] 
                             if st.session_state['admin_logged_in'] 
                             else st.secrets["mongo_general_connection_string"])
    except KeyError:
        st.error("Database connection string not found in secrets.toml.")
        st.stop()

    api_key_manager = APIKeyManager()
    api_keys = api_key_manager.get_user_api_keys()
    if not api_keys:
        st.warning("Please enter at least one valid API key to proceed.")
        st.stop()
    
    available_models = api_key_manager.get_available_models(api_keys)
    selected_model_key = api_key_manager.render_model_selector(available_models)
    if not selected_model_key:
        st.stop()

    # --- CORRECTED INITIALIZATION LOGIC ---
    
    # 1. Get the cached database manager
    db_manager = get_db_manager(connection_string)
    
    # 2. Test connection and display UI feedback OUTSIDE the cached function
    #    Use session state to prevent re-testing on every script rerun.
    if 'db_connection_ok' not in st.session_state:
        st.session_state.db_connection_ok = False

    if not st.session_state.db_connection_ok:
        try:
            with st.spinner("Connecting to database..."):
                db_manager.test_connection()
            st.session_state.db_connection_ok = True
            st.toast("✅ Connected to MongoDB")
        except (ConnectionFailure, ConfigurationError) as e:
            st.error("❌ MongoDB Connection Failed. Check credentials and IP Access List.")
            with st.expander("See error details"):
                st.exception(e)
            st.stop()

    # 3. Get the AI client
    selected_model_info = available_models[selected_model_key]
    provider, model_name, api_key = (selected_model_info["provider"], 
                                     selected_model_info["model_name"], 
                                     api_keys[selected_model_info["provider"]])
    
    ai_client = get_ai_client(provider, api_key, model_name)
    
    # Display toast only once per client
    client_key = f"{provider}_{model_name}_connected"
    if client_key not in st.session_state:
        st.toast(f"✅ Connected to {provider.title()}")
        st.session_state[client_key] = True
    
    # --- END OF CORRECTIONS ---

    # Initialize managers
    pdf_manager = PDFManager(db_manager)
    adql_manager = ADQLManager(db_manager)
    
    render_main_interface(pdf_manager, adql_manager, ai_client)

def render_admin_login():
    """Renders the admin login form in the sidebar."""
    if st.session_state.get('admin_logged_in'):
        if st.sidebar.button("Logout Admin"):
            st.session_state.admin_logged_in = False
            # Reset connection status on logout
            st.session_state.db_connection_ok = False
            st.cache_resource.clear()
            st.rerun()
    else:
        with st.sidebar.expander("Admin Login"):
            password = st.text_input("Enter Admin Password", type="password", key="admin_password_input")
            if st.button("Login"):
                if password == st.secrets.get("app_admin_password"):
                    st.session_state.admin_logged_in = True
                    # Reset connection status on login to force reconnection with admin user
                    st.session_state.db_connection_ok = False
                    st.cache_resource.clear()
                    st.rerun()
                else:
                    st.error("Incorrect password.")

def render_admin_panel(pdf_manager: PDFManager):
    """Renders the PDF management panel in the sidebar."""
    st.sidebar.write("---")
    st.sidebar.header("👑 Admin Panel")

    st.sidebar.subheader("Upload New Document")
    uploaded_file = st.sidebar.file_uploader("Select a PDF", type="pdf", key="pdf_uploader")
    if uploaded_file:
        with st.spinner("Processing PDF..."):
            text = PDFProcessor.extract_text_from_pdf(uploaded_file.getvalue())
            result = pdf_manager.add_pdf_to_db(text, uploaded_file.name)
            st.sidebar.success(f"Status: {result['message']}")

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

def run_streamlit_app():
    """Entry point for running the Streamlit app."""
    main()

if __name__ == "__main__":
    run_streamlit_app()