#!/usr/bin/env python3
"""
Test script for UI modules.
"""

def test_ui_components():
    """Test UI components module."""
    print("🧪 Testing UI components...")
    
    try:
        from chatdesi.ui.components import UIComponents, DataComponents, MathRenderer, SessionManager
        
        # Test component classes can be instantiated
        ui = UIComponents()
        data = DataComponents()
        math = MathRenderer()
        session = SessionManager()
        
        assert ui is not None
        assert data is not None
        assert math is not None
        assert session is not None
        
        print("✅ UI component classes created")
        
        # Test static methods exist
        assert hasattr(UIComponents, 'render_sidebar_settings')
        assert hasattr(DataComponents, 'load_reference_data')
        assert hasattr(MathRenderer, 'render_latex_from_response')
        assert hasattr(SessionManager, 'initialize_chat_session')
        
        print("✅ UI component methods available")
        print("✅ UI components test passed!")
        return True
        
    except Exception as e:
        print(f"❌ UI components test failed: {e}")
        return False


def test_chat_interface():
    """Test chat interface module."""
    print("\n🧪 Testing chat interface...")
    
    try:
        from chatdesi.ui.chat_interface import ChatInterface
        
        # Test that class can be imported
        assert ChatInterface is not None
        print("✅ ChatInterface imported")
        
        # Test interface has required methods
        required_methods = ['render', '_handle_new_message', '_handle_retry_message']
        for method in required_methods:
            assert hasattr(ChatInterface, method), f"Missing method: {method}"
        
        print("✅ ChatInterface methods available")
        print("✅ Chat interface test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Chat interface test failed: {e}")
        return False


def test_adql_interface():
    """Test ADQL interface module."""
    print("\n🧪 Testing ADQL interface...")
    
    try:
        from chatdesi.ui.adql_interface import ADQLInterface
        
        # Test that class can be imported
        assert ADQLInterface is not None
        print("✅ ADQLInterface imported")
        
        # Test interface has required methods
        required_methods = ['render', '_handle_generate_query', '_execute_adql_query']
        for method in required_methods:
            assert hasattr(ADQLInterface, method), f"Missing method: {method}"
        
        print("✅ ADQLInterface methods available")
        print("✅ ADQL interface test passed!")
        return True
        
    except Exception as e:
        print(f"❌ ADQL interface test failed: {e}")
        return False


def test_main_app():
    """Test main application module."""
    print("\n🧪 Testing main application...")
    
    try:
        from chatdesi.main import main, render_main_interface, run_streamlit_app
        
        # Test functions exist
        assert callable(main)
        assert callable(render_main_interface)
        assert callable(run_streamlit_app)
        
        print("✅ Main application functions available")
        print("✅ Main app test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Main app test failed: {e}")
        return False


def test_ui_imports():
    """Test UI module imports."""
    print("\n🧪 Testing UI module imports...")
    
    try:
        # Test package-level imports
        from chatdesi.ui import (
            UIComponents, DataComponents, MathRenderer, SessionManager,
            ChatInterface, ADQLInterface
        )
        
        print("✅ All UI classes imported successfully")
        
        # Test main app import
        from chatdesi.main import main
        print("✅ Main app imported")
        
        return True
        
    except Exception as e:
        print(f"❌ UI import test failed: {e}")
        return False


def test_full_integration():
    """Test integration between all modules."""
    print("\n🧪 Testing full module integration...")
    
    try:
        # Test that all major components can be imported together
        from chatdesi.config import settings
        from chatdesi.auth import create_auth_system
        from chatdesi.data import DatabaseFactory, PDFManager, ADQLManager
        from chatdesi.ui import ChatInterface, ADQLInterface
        from chatdesi.main import main
        
        print("✅ All modules imported together")
        
        # Test that settings work across modules
        assert settings.model.openai_model == "gpt-4o"
        assert settings.database.pdf_db_name == "pdf_database"
        
        print("✅ Configuration integration works")
        print("✅ Full integration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Full integration test failed: {e}")
        return False


def main():
    """Run all UI module tests."""
    print("🚀 Testing chatDESI UI modules...\n")
    
    tests = [
        test_ui_imports,
        test_ui_components,
        test_chat_interface,
        test_adql_interface,
        test_main_app,
        test_full_integration
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All UI tests passed! Ready to run the application!")
        print("\n🚀 Next steps:")
        print("1. Copy your encrypted_credentials.txt to the project root")
        print("2. Run: streamlit run chatdesi/main.py")
        print("3. Test both Chat Mode and ADQL Mode")
        return True
    else:
        print("❌ Some tests failed. Check the errors above.")
        return False


if __name__ == "__main__":
    main()