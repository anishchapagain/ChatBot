"""
Main application entry point for the Data Analysis Chatbot.
"""
import streamlit as st
import yaml
import os
from modules.data_loader import DataLoader
from modules.llm_manager import LLMManager
from modules.query_converter import QueryConverter
from modules.error_handler import ErrorHandler
from ui.chatbot import ChatbotUI
from ui.dashboard import Dashboard

def load_config():
    """Load application configuration from YAML file."""
    try:
        with open("config/config.yaml", "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        st.error(f"Error loading configuration: {str(e)}")
        return {}

def main():
    """Main application."""
    # Load configuration
    config = load_config()
    
    # Initialize error handler
    error_handler = ErrorHandler(config.get("errors", {}))
    
    try:
        # Set page configuration
        st.set_page_config(
            page_title=config.get("ui", {}).get("title", "Data Analysis Chatbot"),
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Add app title
        st.title(config.get("ui", {}).get("title", "Data Analysis Chatbot"))
        
        # Initialize core components
        data_loader = DataLoader(config.get("data", {}), error_handler)
        llm_manager = LLMManager(config.get("llm", {}), error_handler)
        query_converter = QueryConverter(llm_manager, error_handler)
        
        # Create tabs for chatbot and dashboard
        tab1, tab2 = st.tabs(["Chat Interface", "Analysis Dashboard"])
        
        # Initialize UI components
        with tab1:
            chatbot_ui = ChatbotUI(data_loader, query_converter, error_handler, config)
            chatbot_ui.render()
            
        with tab2:
            dashboard = Dashboard(data_loader, error_handler, config)
            dashboard.render()
            
    except Exception as e:
        error_handler.handle_error(e, "Application startup failed")
        st.error("An error occurred during application startup. Please check the logs.")

if __name__ == "__main__":
    main()