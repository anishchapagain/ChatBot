"""
Module for managing chat history with save/load functionality.
"""
import json
import os
import time
from datetime import datetime
import pandas as pd
from typing import Dict, List, Optional, Any
from modules.error_handler import ErrorHandler

class ChatHistoryManager:
    """
    Manages chat history with save and load functionality.
    """
    def __init__(self, error_handler: ErrorHandler, config: Dict):
        """
        Initialize the chat history manager.
        
        Args:
            error_handler: Error handler instance
            config: Application configuration
        """
        self.error_handler = error_handler
        self.config = config
        self.history_dir = config.get("history", {}).get("directory", "./history")
        
        # Create history directory if it doesn't exist
        os.makedirs(self.history_dir, exist_ok=True)
    
    def save_history(self, history: List[Dict], conversation_name: Optional[str] = None) -> str:
        """
        Save chat history to a JSON file.
        
        Args:
            history: List of chat history messages
            conversation_name: Optional name for the conversation
            
        Returns:
            Path to the saved history file
        """
        try:
            # Process history to save only necessary information
            processed_history = []
            
            for message in history:
                processed_message = {
                    "role": message["role"],
                    "content": message["content"],
                    "timestamp": message.get("timestamp", datetime.now().isoformat())
                }
                
                # Handle result differently based on type
                if "result" in message and message["result"] is not None:
                    result = message["result"]
                    
                    if isinstance(result, pd.DataFrame):
                        # For DataFrame, save shape
                        processed_message["result_type"] = "dataframe"
                        processed_message["result_shape"] = list(result.shape)
                    elif isinstance(result, pd.Series):
                        # For Series, save length
                        processed_message["result_type"] = "series"
                        processed_message["result_shape"] = [len(result)]
                    elif isinstance(result, (int, float, bool, str)):
                        # For scalar values, save the actual value
                        processed_message["result_type"] = "scalar"
                        processed_message["result_value"] = result
                    else:
                        # For other types, just save the type
                        processed_message["result_type"] = str(type(result).__name__)
                
                # Save code if available
                if "code" in message:
                    processed_message["code"] = message["code"]
                
                processed_history.append(processed_message)
            
            # Generate filename if not provided
            if not conversation_name:
                timestamp = int(time.time())
                conversation_name = f"conversation_{timestamp}"
            
            # Ensure the filename has .json extension
            if not conversation_name.endswith(".json"):
                conversation_name += ".json"
            
            file_path = os.path.join(self.history_dir, conversation_name)
            
            # Save to file
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump({
                    "history": processed_history,
                    "saved_at": datetime.now().isoformat(),
                    "version": "1.0"
                }, f, indent=2)
            
            return file_path
        
        except Exception as e:
            self.error_handler.handle_error(e, "Error saving chat history")
            return ""
    
    def load_history(self, file_path: str) -> List[Dict]:
        """
        Load chat history from a JSON file.
        
        Args:
            file_path: Path to the history file
            
        Returns:
            List of chat history messages
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Get the history from the data
            history = data.get("history", [])
            
            # Process the history for display
            processed_history = []
            
            for message in history:
                processed_message = {
                    "role": message["role"],
                    "content": message["content"]
                }
                
                # Handle result placeholder
                if "result_type" in message:
                    if message["result_type"] == "dataframe":
                        processed_message["result_info"] = f"DataFrame with {message['result_shape'][0]} rows and {message['result_shape'][1]} columns"
                    elif message["result_type"] == "series":
                        processed_message["result_info"] = f"Series with {message['result_shape'][0]} items"
                    elif message["result_type"] == "scalar":
                        processed_message["result_info"] = f"Result: {message['result_value']}"
                    else:
                        processed_message["result_info"] = f"Result of type: {message['result_type']}"
                
                # Include code if available
                if "code" in message:
                    processed_message["code"] = message["code"]
                
                processed_history.append(processed_message)
            
            return processed_history
        
        except Exception as e:
            self.error_handler.handle_error(e, f"Error loading chat history from {file_path}")
            return []
    
    def get_available_histories(self) -> List[Dict]:
        """
        Get a list of available history files.
        
        Returns:
            List of history file information
        """
        try:
            history_files = []
            
            for filename in os.listdir(self.history_dir):
                if filename.endswith(".json"):
                    file_path = os.path.join(self.history_dir, filename)
                    
                    try:
                        # Get file stats
                        stats = os.stat(file_path)
                        modified_time = datetime.fromtimestamp(stats.st_mtime)
                        
                        # Try to extract more info from the file
                        with open(file_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        
                        # Get number of messages
                        message_count = len(data.get("history", []))
                        
                        # Get save time from file if available
                        saved_at = data.get("saved_at", modified_time.isoformat())
                        
                        history_files.append({
                            "filename": filename,
                            "path": file_path,
                            "modified": modified_time.strftime("%Y-%m-%d %H:%M:%S"),
                            "message_count": message_count,
                            "saved_at": saved_at
                        })
                    except Exception:
                        # If we can't read the file, just include basic info
                        history_files.append({
                            "filename": filename,
                            "path": file_path,
                            "modified": modified_time.strftime("%Y-%m-%d %H:%M:%S"),
                            "message_count": "Unknown",
                            "saved_at": "Unknown"
                        })
            
            # Sort by modification time (newest first)
            history_files.sort(key=lambda x: x["modified"], reverse=True)
            
            return history_files
        
        except Exception as e:
            self.error_handler.handle_error(e, "Error getting available history files")
            return []
    
    def generate_suggested_prompts(self, df: pd.DataFrame) -> List[str]:
        """
        Generate suggested prompts based on the DataFrame.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            List of suggested prompts
        """
        try:
            suggestions = []
            
            # Basic column and shape info
            num_rows, num_cols = df.shape
            columns = df.columns.tolist()
            
            # Check if there are numerical columns
            numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns
            if len(numerical_columns) > 0:
                # Suggest basic statistics
                col = numerical_columns[0]
                suggestions.append(f"What is the average {col}?")
                suggestions.append(f"Show me the distribution of {col}")
                
                # If multiple numerical columns, suggest correlation
                if len(numerical_columns) > 1:
                    col1 = numerical_columns[0]
                    col2 = numerical_columns[1]
                    suggestions.append(f"What is the correlation between {col1} and {col2}?")
                    suggestions.append(f"Create a scatter plot of {col1} vs {col2}")
            
            # Check if there are categorical columns
            categorical_columns = df.select_dtypes(include=["object", "category"]).columns
            if len(categorical_columns) > 0:
                # Suggest counts and grouping
                cat_col = categorical_columns[0]
                suggestions.append(f"Count the occurrences of each {cat_col}")
                
                # If both numerical and categorical, suggest grouped analysis
                if len(numerical_columns) > 0:
                    num_col = numerical_columns[0]
                    suggestions.append(f"Calculate the average {num_col} for each {cat_col}")
                    suggestions.append(f"Show me a box plot of {num_col} by {cat_col}")
            
            # Check for potential date columns
            date_columns = []
            for col in df.columns:
                # Check if column name suggests date
                if any(date_hint in col.lower() for date_hint in ["date", "time", "day", "month", "year"]):
                    date_columns.append(col)
                # Try to convert a sample and see if it works
                elif df[col].dtype == "object":
                    try:
                        pd.to_datetime(df[col].head(), format='%d-%b-%y')
                        date_columns.append(col)
                    except:
                        pass
            
            if date_columns and len(numerical_columns) > 0:
                date_col = date_columns[0]
                num_col = numerical_columns[0]
                suggestions.append(f"Show me the trend of {num_col} over {date_col}")
                suggestions.append(f"What was the highest {num_col} and when did it occur?")
            
            # Add some general suggestions
            suggestions.append("Summarize this dataset")
            suggestions.append(f"Find the top 5 rows with highest values in {columns[0]}")
            suggestions.append("Show me some basic statistics for all numerical columns")
            
            # Randomize and limit suggestions
            import random
            random.shuffle(suggestions)
            return suggestions[:5]  # Limit to 5 suggestions
            
        except Exception as e:
            self.error_handler.handle_error(e, "Error generating suggested prompts")
            return ["Summarize this dataset", "Calculate statistics for all columns"]