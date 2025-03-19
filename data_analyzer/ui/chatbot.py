"""
Module for the chatbot user interface.
"""
import streamlit as st
import pandas as pd
from typing import Dict, List, Optional, Any
import json
from modules.data_loader import DataLoader
from modules.query_converter import QueryConverter
from modules.error_handler import ErrorHandler

class ChatbotUI:
    """
    Chatbot user interface component.
    """
    def __init__(
        self, 
        data_loader: DataLoader, 
        query_converter: QueryConverter, 
        error_handler: ErrorHandler,
        config: Dict
    ):
        """
        Initialize the chatbot UI.
        
        Args:
            data_loader: Data loader instance
            query_converter: Query converter instance
            error_handler: Error handler instance
            config: Application configuration
        """
        self.data_loader = data_loader
        self.query_converter = query_converter
        self.error_handler = error_handler
        self.config = config
        self.max_chat_history = config.get("ui", {}).get("max_chat_history", 20)
    
    def render(self):
        """Render the chatbot interface."""
        try:
            st.header("Data Analysis Chat")
            
            # Initialize session state for chat history if not exists
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
            
            # File uploader
            uploaded_file = st.file_uploader(
                "Upload your data file (CSV, Excel, or JSON)", 
                type=["csv", "xlsx", "json"]
            )
            
            if uploaded_file is not None:
                # Save the uploaded file temporarily
                with open(f"./data/{uploaded_file.name}", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Load the data
                df = self.data_loader.load_data(f"./data/{uploaded_file.name}")
                
                if df is not None:
                    st.success(f"Data loaded successfully: {df.shape[0]} rows and {df.shape[1]} columns.")
                    
                    # Display sample data
                    with st.expander("Preview Data"):
                        st.dataframe(df.head())
                    
                    # Display chat history
                    self._display_chat_history()
                    
                    # Query input
                    query = st.text_input("Ask a question about your data:")
                    
                    if st.button("Submit") and query:
                        self._process_query(query, df)
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
            
            if role == "user":
                st.markdown(f"**You:** {content}")
            elif role == "assistant":
                st.markdown(f"**Assistant:** {content}")
                
                # If there's a result to display
                if "result" in message and message["result"] is not None:
                    self._display_result(message["result"], message.get("code", ""))
    
    def _process_query(self, query: str, df: pd.DataFrame):
        """
        Process a user query.
        
        Args:
            query: User query
            df: DataFrame to query
        """
        try:
            # Add user message to chat history
            st.session_state.chat_history.append({
                "role": "user",
                "content": query
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
                    "result": None
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
            
            # Add assistant message to chat history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response,
                "code": pandas_code,
                "result": result
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
                    st.dataframe(result)
                else:
                    st.info("The result is an empty DataFrame.")
            
            elif isinstance(result, pd.Series):
                if len(result) > 0:
                    st.dataframe(result.to_frame())
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