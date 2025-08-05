#!/usr/bin/env python3
"""
Test script to verify our new modules work correctly.
"""

def test_config_module():
    """Test the config module."""
    print("🧪 Testing config module...")
    
    try:
        from chatdesi.config import settings
        
        # Test database config
        assert settings.database.pdf_db_name == "pdf_database"
        print("✅ Database config loaded")
        
        # Test model config
        assert settings.model.openai_model == "gpt-4o"
        print("✅ Model config loaded")
        
        # Test connection string generation
        conn_str = settings.get_mongodb_connection_string("user", "pass")
        assert "mongodb+srv://user:pass@" in conn_str
        print("✅ MongoDB connection string generation works")
        
        print("✅ Config module test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Config module test failed: {e}")
        return False


def test_auth_module():
    """Test the auth module."""
    print("\n🧪 Testing auth module...")
    
    try:
        from chatdesi.auth import CredentialManager, Credentials
        
        # Test credential manager creation
        manager = CredentialManager("nonexistent_file.txt")
        assert manager is not None
        print("✅ CredentialManager created")
        
        # Test credentials dataclass
        creds = Credentials("api_key", "user", "pass")
        assert creds.openai_api_key == "api_key"
        print("✅ Credentials dataclass works")
        
        print("✅ Auth module test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Auth module test failed: {e}")
        return False


def test_imports():
    """Test that all imports work."""
    print("\n🧪 Testing imports...")
    
    try:
        # Test individual imports
        from chatdesi.config.settings import settings
        from chatdesi.auth.credentials import CredentialManager
        
        print("✅ Individual imports work")
        
        # Test package imports
        from chatdesi.config import settings as config_settings
        from chatdesi.auth import create_auth_system
        
        print("✅ Package imports work")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Testing chatDESI modular components...\n")
    
    tests = [
        test_imports,
        test_config_module, 
        test_auth_module
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! Ready for next step.")
        return True
    else:
        print("❌ Some tests failed. Check the errors above.")
        return False


if __name__ == "__main__":
    main()