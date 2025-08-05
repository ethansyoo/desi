"""
Database connection and management for chatDESI.
"""

from typing import Optional
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from ..config import settings


class DatabaseManager:
    """Manages MongoDB connections for chatDESI."""
    
    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password
        self._client: Optional[MongoClient] = None
        self._pdf_db: Optional[Database] = None
        self._adql_db: Optional[Database] = None
    
    def _get_client(self) -> MongoClient:
        """Get or create MongoDB client."""
        if self._client is None:
            connection_string = settings.get_mongodb_connection_string(
                self.username, self.password
            )
            self._client = MongoClient(connection_string)
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
        """Test if database connection is working."""
        try:
            client = self._get_client()
            # Ping the database
            client.admin.command('ping')
            return True
        except Exception:
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
    def create_from_credentials(username: str, password: str) -> DatabaseManager:
        """Create database manager from credentials."""
        return DatabaseManager(username, password)
    
    @staticmethod
    def create_from_auth(credentials) -> DatabaseManager:
        """Create database manager from auth credentials object."""
        return DatabaseManager(
            credentials.mongo_username,
            credentials.mongo_password
        )