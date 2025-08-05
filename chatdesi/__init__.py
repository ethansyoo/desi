"""
chatDESI - Modular astronomical data analysis chatbot.
"""

__version__ = "2.0.0"
__author__ = "Ethan Yoo"

from .config.settings import settings
from .main import main, run_streamlit_app

__all__ = ["settings", "main", "run_streamlit_app"]