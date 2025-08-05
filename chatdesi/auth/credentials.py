"""
Authentication and credential management for chatDESI.
"""

import hashlib
import os
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass

import streamlit as st
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


@dataclass
class Credentials:
    """Container for decrypted credentials."""
    openai_api_key: str
    mongo_username: str
    mongo_password: str


class CredentialManager:
    """Manages encrypted credentials for the application."""
    
    def __init__(self, credentials_file: str = "encrypted_credentials.txt"):
        self.credentials_file = Path(credentials_file)
        self._cached_credentials: Optional[Credentials] = None
    
    def decrypt_data(self, encrypted_data: bytes, key: bytes) -> str:
        """Decrypt data using AES-GCM."""
        nonce = encrypted_data[:12]
        aesgcm = AESGCM(key)
        decrypted = aesgcm.decrypt(nonce, encrypted_data[12:], None)
        return decrypted.decode()
    
    def derive_key_from_password(self, password: str, salt: bytes) -> bytes:
        """Derive encryption key from password using PBKDF2."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return kdf.derive(password.encode())
    
    def verify_password(self, password: str, stored_hash: bytes) -> bool:
        """Verify password against stored hash."""
        entered_hash = hashlib.sha256(password.encode()).hexdigest().encode()
        return entered_hash == stored_hash
    
    def load_credentials(self, password: str) -> Optional[Credentials]:
        """Load and decrypt credentials from file."""
        if not self.credentials_file.exists():
            return None
        
        try:
            with open(self.credentials_file, "rb") as f:
                lines = f.readlines()
                if len(lines) < 5:
                    raise ValueError("Invalid credentials file format")
                
                salt = lines[0].strip()
                stored_password_hash = lines[1].strip()
                encrypted_openai_key = lines[2].strip()
                encrypted_mongo_username = lines[3].strip()
                encrypted_mongo_password = lines[4].strip()
            
            # Verify password
            if not self.verify_password(password, stored_password_hash):
                return None
            
            # Derive key and decrypt
            key = self.derive_key_from_password(password, salt)
            
            credentials = Credentials(
                openai_api_key=self.decrypt_data(encrypted_openai_key, key),
                mongo_username=self.decrypt_data(encrypted_mongo_username, key),
                mongo_password=self.decrypt_data(encrypted_mongo_password, key)
            )
            
            self._cached_credentials = credentials
            return credentials
            
        except Exception as e:
            st.error(f"Error loading credentials: {e}")
            return None
    
    def get_cached_credentials(self) -> Optional[Credentials]:
        """Get cached credentials if available."""
        return self._cached_credentials
    
    def clear_cache(self):
        """Clear cached credentials."""
        self._cached_credentials = None


class StreamlitAuth:
    """Streamlit-specific authentication interface."""
    
    def __init__(self, credential_manager: CredentialManager):
        self.credential_manager = credential_manager
    
    def initialize_session_state(self):
        """Initialize authentication-related session state."""
        if "authenticated" not in st.session_state:
            st.session_state["authenticated"] = False
        if "credentials" not in st.session_state:
            st.session_state["credentials"] = None
    
    def show_login_form(self) -> bool:
        """
        Show login form and handle authentication.
        Returns True if user is authenticated, False otherwise.
        """
        self.initialize_session_state()
        
        if st.session_state["authenticated"]:
            return True
        
        st.title("chatDESI")
        st.markdown("### Please enter your credentials to continue")
        
        with st.form("login_form"):
            password = st.text_input("Enter your password:", type="password")
            submitted = st.form_submit_button("Login")
            
            if submitted and password:
                credentials = self.credential_manager.load_credentials(password)
                
                if credentials:
                    st.session_state["authenticated"] = True
                    st.session_state["credentials"] = credentials
                    st.success("Credentials verified successfully!")
                    st.rerun()
                else:
                    st.error("Invalid password or corrupted credentials file.")
                    return False
            elif submitted:
                st.warning("Please enter a password.")
        
        return st.session_state["authenticated"]
    
    def get_credentials(self) -> Optional[Credentials]:
        """Get current user credentials."""
        return st.session_state.get("credentials")
    
    def logout(self):
        """Logout user and clear session."""
        st.session_state["authenticated"] = False
        st.session_state["credentials"] = None
        self.credential_manager.clear_cache()
        st.rerun()


# Convenience functions for easy import
def create_auth_system(credentials_file: str = "encrypted_credentials.txt") -> StreamlitAuth:
    """Create and return a configured authentication system."""
    credential_manager = CredentialManager(credentials_file)
    return StreamlitAuth(credential_manager)


def require_authentication(auth_system: StreamlitAuth) -> Optional[Credentials]:
    """
    Require authentication and return credentials if successful.
    This function should be called at the beginning of your Streamlit app.
    """
    if not auth_system.show_login_form():
        st.stop()  # Stop execution if not authenticated
    
    return auth_system.get_credentials()