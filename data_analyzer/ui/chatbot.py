"""
Module for the chatbot user interface.
"""
import streamlit as st
import pandas as pd
from typing import Dict, List, Optional, Any
import json
import time
import datetime
from modules.data_loader import DataLoader
from modules.query_converter import QueryConverter
from modules.error_handler import ErrorHandler
from modules.chat_history_manager import ChatHistoryManager
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ChatbotUI:
    """
    Chatbot user interface component.
    """
    def __init__(
        self, 
        data_loader: DataLoader, 
        query_converter: QueryConverter, 
        error_handler: ErrorHandler,
        config: Dict,
        history_manager: Optional[ChatHistoryManager] = None
    ):
        """
        Initialize the chatbot UI.
        
        Args:
            data_loader: Data loader instance
            query_converter: Query converter instance
            error_handler: Error handler instance
            config: Application configuration
            history_manager: Chat history manager instance
        """
        self.data_loader = data_loader
        self.query_converter = query_converter
        self.error_handler = error_handler
        self.config = config
        self.history_manager = history_manager
        self.max_chat_history = config.get("ui", {}).get("max_chat_history", 20)
        self.auto_save = config.get("history", {}).get("auto_save", True)
    
    def render(self):
        """Render the chatbot interface."""
        try:
            st.header("Data Analysis Chat")
            
            # Initialize session state variables if they don't exist
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
            if "current_file" not in st.session_state:
                st.session_state.current_file = None
            if "last_save_time" not in st.session_state:
                st.session_state.last_save_time = time.time()
            if "conversation_name" not in st.session_state:
                st.session_state.conversation_name = f"conversation_{int(time.time())}"
            
            # Sidebar for history management
            with st.sidebar:
                st.header("Conversation Management")
                
                # New chat button
                if st.button("New Chat", use_container_width=True, help="Start a new chat session", type="secondary", icon=":material/chat:"):
                    # Save current chat if auto-save is enabled and there's history
                    if self.auto_save and self.history_manager and len(st.session_state.chat_history) > 0:
                        self._save_current_chat()
                    
                    # Reset chat history
                    st.session_state.chat_history = []
                    st.session_state.conversation_name = f"conversation_{int(time.time())}"
                    st.experimental_rerun()
                
                # Save chat button
                if st.button("Save Chat",use_container_width=True, help="Save current chat sesison", type="secondary", icon=":material/save:") and self.history_manager:
                    self._save_current_chat()
                
                # Load chat section
                if self.history_manager:
                    st.subheader("Load Previous Conversation")
                    history_files = self.history_manager.get_available_histories()
                    
                    if history_files:
                        # Create a dictionary of filenames to paths for the selectbox
                        options = {f"{h['filename']} ({h['message_count']} messages, {h['modified']})": h["path"] 
                                  for h in history_files}
                        
                        selected_history = st.selectbox(
                            "Select a conversation to load",
                            options.keys()
                        )
                        
                        if st.button("Load Selected Chat", help="Load the selected chat history", use_container_width=True) and selected_history:
                            selected_path = options[selected_history]
                            loaded_history = self.history_manager.load_history(selected_path)
                            
                            if loaded_history:
                                # Save current chat first if auto-save is enabled
                                if self.auto_save and len(st.session_state.chat_history) > 0:
                                    self._save_current_chat()
                                
                                # Set the loaded history and update the conversation name
                                st.session_state.chat_history = loaded_history
                                st.session_state.conversation_name = selected_history.split(" (")[0]
                                st.experimental_rerun()
                            else:
                                st.error("Failed to load the selected chat history.")
                    else:
                        st.info("No previous conversations found.")
            
            # Main content area
            
            # File uploader
            uploaded_file = st.file_uploader(
                "Upload your data file (CSV, Excel, or JSON)", 
                type=["csv", "xlsx", "json"]
            )
            
            if uploaded_file is not None:
                # Save the uploaded file temporarily
                with open(f"./data/{uploaded_file.name}", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Check if this is a new file
                if st.session_state.current_file != uploaded_file.name:
                    st.session_state.current_file = uploaded_file.name
                    
                    # Ask if user wants to start a new chat for new file
                    if len(st.session_state.chat_history) > 0:
                        if st.button("Start new chat with this file", use_container_width=True):
                            # Save current chat if auto-save is enabled
                            if self.auto_save and self.history_manager:
                                self._save_current_chat()
                            
                            # Reset chat history
                            st.session_state.chat_history = []
                            st.session_state.conversation_name = f"{uploaded_file.name.split('.')[0]}_{int(time.time())}"
                            st.experimental_rerun()
                
                # Load the data
                df = self.data_loader.load_data(f"./data/{uploaded_file.name}")
                
                if df is not None:
                    st.success(f"Data loaded successfully: {df.shape[0]} rows and {df.shape[1]} columns.")
                    
                    # Display sample data
                    with st.expander("Preview Data"):
                        st.dataframe(df.head(), use_container_width=True)
                    
                    # Display chat history
                    self._display_chat_history()
                    
                    # Query input
                    query = st.chat_input("Show inactive accounts with balance greater than 100000")
                    
                    # # Submit button or suggested prompts
                    # col1, col2 = st.columns([1, 3])
                    
                    # with col1:
                    #     submit_button = st.button("Submit", use_container_width=True)
                    
                    # with col2:
                    #     if self.history_manager and (not query or len(st.session_state.chat_history) > 0 and 
                    #                                 st.session_state.chat_history[-1].get("result") is None):
                    #         # Show suggested prompts
                    #         suggested_prompts = self.history_manager.generate_suggested_prompts(df)
                            
                    #         if suggested_prompts:
                    #             st.write("Try one of these questions:")
                    #             for i, prompt in enumerate(suggested_prompts):
                    #                 if st.button(f"{prompt}", key=f"suggest_{i}"):
                    #                     query = prompt
                    #                     submit_button = True
                    
                    if query: # submit_button and query:
                        self._process_query(query, df)
                        
                        # Auto-save if enabled and it's been long enough since last save
                        current_time = time.time()
                        if (self.auto_save and self.history_manager and 
                            current_time - st.session_state.last_save_time > 
                            self.config.get("history", {}).get("auto_save_interval", 5) * 60):
                            self._save_current_chat()
                            st.session_state.last_save_time = current_time
                else:
                    st.error("Failed to load data. Please check the file format.")
            else:
                st.info("Please upload a data file to start analyzing.")
                
                # Display sample queries
                with st.expander("Sample Queries"):
                    st.markdown("""
                    Here are some examples of questions you can ask:
                    - Show me the average value of [column]
                    - What is the correlation between [column1] and [column2]?
                    - Filter the data where [column] is greater than [value]
                    - Group the data by [column] and show the mean
                    - Create a histogram of [column]
                    - Show the data sorted by [column] in descending order
                    """)
        
        except Exception as e:
            error_msg = self.error_handler.handle_error(e, "Error rendering chatbot UI")
            st.error(error_msg)
    
    def _display_chat_history(self):
        """Display the chat history."""
        st.subheader("Conversation")
        
        for message in st.session_state.chat_history:
            role = message["role"]
            content = message["content"]
            
            # Display timestamp if available
            timestamp_str = ""
            if "timestamp" in message:
                try:
                    dt = datetime.datetime.fromisoformat(message["timestamp"])
                    timestamp_str = f"  *{dt.strftime('%H:%M:%S')}*"
                except:
                    pass
            
            if role == "user":
                st.markdown(f"**You:** {content}{timestamp_str}")
            elif role == "assistant":
                # Check if this is a suggestion message
                if message.get("is_suggestion", False):
                    st.markdown(f"*{content}*")
                else:
                    st.markdown(f"**Assistant:** {content}{timestamp_str}")
                    
                    # If there's a result to display
                    if "result" in message and message["result"] is not None:
                        self._display_result(message["result"], message.get("code", ""))
                    # If there's a result_info to display (for loaded history)
                    elif "result_info" in message:
                        st.info(message["result_info"])
                        
                        # Display code if available
                        if "code" in message:
                            with st.expander("Show Pandas Code"):
                                st.code(message["code"], language="python")
    
    def _save_current_chat(self):
        """Save the current chat history."""
        if not self.history_manager or not st.session_state.chat_history:
            return
        
        try:
            file_path = self.history_manager.save_history(
                st.session_state.chat_history,
                st.session_state.conversation_name
            )
            
            if file_path:
                st.sidebar.success(f"Chat saved successfully!")
                st.session_state.last_save_time = time.time()
            else:
                st.sidebar.error("Failed to save chat history.")
        except Exception as e:
            error_msg = self.error_handler.handle_error(e, "Error saving chat history")
            st.sidebar.error(error_msg)
    
    def _process_query(self, query: str, df: pd.DataFrame):
        """
        Process a user query.
        
        Args:
            query: User query
            df: DataFrame to query
        """
        try:
            # Add user message to chat history with timestamp
            st.session_state.chat_history.append({
                "role": "user",
                "content": query,
                "timestamp": datetime.datetime.now().isoformat()
            })
            
            # Get DataFrame metadata
            df_metadata = self.data_loader.get_metadata()
            
            # Convert query to pandas code
            pandas_code, is_safe = self.query_converter.convert_to_pandas(query, df_metadata)
            
            if not is_safe:
                # Add assistant response for unsafe code
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": "I'm sorry, but the generated code contains potentially unsafe operations. Please rephrase your query.",
                    "code": pandas_code,
                    "result": None,
                    "timestamp": datetime.datetime.now().isoformat()
                })
                
                # Add suggested prompts when result is None
                if self.history_manager:
                    suggestions = self.history_manager.generate_suggested_prompts(df)
                    if suggestions:
                        suggestion_text = "Here are some questions you could try instead:\n- " + "\n- ".join(suggestions[:3])
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": suggestion_text,
                            "is_suggestion": True,
                            "timestamp": datetime.datetime.now().isoformat()
                        })
                return
            
            # Execute the pandas code
            result, error_msg, success = self.query_converter.execute_pandas_query(pandas_code, df)
            
            if success:
                # Format the response
                if isinstance(result, pd.DataFrame):
                    response = f"Here's the result ({result.shape[0]} rows × {result.shape[1]} columns):"
                elif isinstance(result, pd.Series):
                    response = f"Here's the result ({len(result)} items):"
                else:
                    response = f"The result is: {result}"
            else:
                response = f"I encountered an error while executing the query: {error_msg}"
                result = None
                
                # Add suggested prompts when result is None
                if self.history_manager:
                    suggestions = self.history_manager.generate_suggested_prompts(df)
                    if suggestions:
                        suggestion_text = "Here are some questions you could try instead:\n- " + "\n- ".join(suggestions[:3])
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": suggestion_text,
                            "is_suggestion": True,
                            "timestamp": datetime.datetime.now().isoformat()
                        })
            
            # Add assistant message to chat history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response,
                "code": pandas_code,
                "result": result,
                "timestamp": datetime.datetime.now().isoformat()
            })
            
            # Limit chat history size
            if len(st.session_state.chat_history) > self.max_chat_history:
                st.session_state.chat_history = st.session_state.chat_history[-self.max_chat_history:]
            
            # Refresh the page to show the new message
            st.experimental_rerun()
        
        except Exception as e:
            error_msg = self.error_handler.handle_error(e, "Error processing query")
            st.error(error_msg)
    
    def _display_result(self, result: Any, code: str):
        """
        Display the result of a query.
        
        Args:
            result: Result of the query
            code: Pandas code that generated the result
        """
        try:
            # Display the pandas code
            with st.expander("Show Pandas Code"):
                st.code(code, language="python")
            
            # Display the result
            if isinstance(result, pd.DataFrame):
                if len(result) > 0:
                    st.dataframe(result, use_container_width=True)
                else:
                    st.info("The result is an empty DataFrame.")
            
            elif isinstance(result, pd.Series):
                if len(result) > 0:
                    st.dataframe(result.to_frame(), use_container_width=True)
                else:
                    st.info("The result is an empty Series.")
            
            elif isinstance(result, (int, float, str, bool)):
                st.success(f"Result: {result}")
            
            elif result is None:
                st.info("The query did not return any result.")
            
            else:
                # Try to convert to string representation
                try:
                    st.text(str(result))
                except:
                    st.warning("Could not display the result.")
        
        except Exception as e:
            error_msg = self.error_handler.handle_error(e, "Error displaying result")
            st.error(error_msg)