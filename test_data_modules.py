#!/usr/bin/env python3
"""
Test script for data modules.
"""

def test_database_module():
    """Test database management module."""
    print("🧪 Testing database module...")
    
    try:
        from chatdesi.data.database import DatabaseManager, DatabaseFactory
        
        # Test database manager creation
        db_manager = DatabaseManager("test_user", "test_pass")
        assert db_manager.username == "test_user"
        print("✅ DatabaseManager created")
        
        # Test factory
        db_manager2 = DatabaseFactory.create_from_credentials("user", "pass")
        assert db_manager2 is not None
        print("✅ DatabaseFactory works")
        
        print("✅ Database module test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Database module test failed: {e}")
        return False


def test_pdf_module():
    """Test PDF management module.""" 
    print("\n🧪 Testing PDF module...")
    
    try:
        from chatdesi.data.pdf_manager import EmbeddingModel, PDFProcessor
        
        # Test embedding model (without actually loading it)
        print("✅ EmbeddingModel import works")
        
        # Test PDF processor
        processor = PDFProcessor()
        assert processor is not None
        print("✅ PDFProcessor created")
        
        print("✅ PDF module test passed!")
        return True
        
    except Exception as e:
        print(f"❌ PDF module test failed: {e}")
        return False


def test_adql_module():
    """Test ADQL management module."""
    print("\n🧪 Testing ADQL module...")
    
    try:
        from chatdesi.data.adql_manager import RenderUtilities
        
        # Test render utilities
        renderer = RenderUtilities()
        assert renderer is not None
        print("✅ RenderUtilities works")
        
        print("✅ ADQL module test passed!")
        return True
        
    except Exception as e:
        print(f"❌ ADQL module test failed: {e}")
        return False


def test_data_imports():
    """Test that data module imports work."""
    print("\n🧪 Testing data module imports...")
    
    try:
        # Test package-level imports
        from chatdesi.data import DatabaseManager, PDFManager, ADQLManager
        print("✅ Main data classes imported")
        
        # Test factory imports
        from chatdesi.data import DatabaseFactory, PDFProcessor
        print("✅ Utility classes imported")
        
        return True
        
    except Exception as e:
        print(f"❌ Data import test failed: {e}")
        return False


def test_integration():
    """Test basic integration between modules."""
    print("\n🧪 Testing module integration...")
    
    try:
        from chatdesi.config import settings
        from chatdesi.data import DatabaseFactory
        
        # Test that config settings work with database
        connection_str = settings.get_mongodb_connection_string("user", "pass")
        assert "mongodb+srv://" in connection_str
        print("✅ Config-Database integration works")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


def main():
    """Run all data module tests."""
    print("🚀 Testing chatDESI data modules...\n")
    
    tests = [
        test_data_imports,
        test_database_module,
        test_pdf_module, 
        test_adql_module,
        test_integration
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All data module tests passed! Ready for Step 4: UI Components.")
        return True
    else:
        print("❌ Some tests failed. Check the errors above.")
        return False


if __name__ == "__main__":
    main()