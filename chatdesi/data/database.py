"""
Database connection and management for chatDESI.
"""

import certifi
from typing import Optional
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import ConnectionFailure

from ..config import settings


class DatabaseManager:
    """Manages MongoDB connections for chatDESI."""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self._client: Optional[MongoClient] = None
        self._pdf_db: Optional[Database] = None
        self._adql_db: Optional[Database] = None
    
    def _get_client(self) -> MongoClient:
        """Get or create MongoDB client."""
        if self._client is None:
            # NEW: Add tlsCAFile=certifi.where() to use the certifi library
            self._client = MongoClient(
                self.connection_string,
                serverSelectionTimeoutMS=60000,
                tlsCAFile=certifi.where()
            )
        return self._client
    
    def get_pdf_collection(self) -> Collection:
        """Get PDF documents collection."""
        if self._pdf_db is None:
            client = self._get_client()
            self._pdf_db = client[settings.database.pdf_db_name]
        
        return self._pdf_db[settings.database.pdf_collection_name]
    
    def get_adql_collection(self) -> Collection:
        """Get ADQL feedback collection."""
        if self._adql_db is None:
            client = self._get_client()
            self._adql_db = client[settings.database.adql_db_name]
        
        return self._adql_db[settings.database.adql_collection_name]
    
    def test_connection(self) -> bool:
        """Test if database connection is working with detailed error logging."""
        try:
            client = self._get_client()
            client.admin.command('ping')
            return True
        except ConnectionFailure as e:
            import streamlit as st
            st.error("MongoDB ConnectionFailure: A detailed error occurred.")
            st.exception(e)
            return False
        except Exception as e:
            import streamlit as st
            st.error("An unexpected error occurred during the database connection test.")
            st.exception(e)
            return False
    
    def close_connection(self):
        """Close database connections."""
        if self._client:
            self._client.close()
            self._client = None
            self._pdf_db = None
            self._adql_db = None


class DatabaseFactory:
    """Factory for creating database managers."""
    
    @staticmethod
    def create_from_connection_string(connection_string: str) -> DatabaseManager:
        """Create database manager from a full connection string."""
        return DatabaseManager(connection_string)