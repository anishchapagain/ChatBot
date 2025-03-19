"""
Module for loading and handling data from various sources.
"""
import pandas as pd
import os
import logging
from typing import Dict, List, Optional, Tuple, Union
from modules.error_handler import ErrorHandler

class DataLoader:
    """
    Loads and manages data from various sources.
    """
    def __init__(self, config: Dict, error_handler: ErrorHandler):
        """
        Initialize the data loader with configuration.
        
        Args:
            config: Configuration dictionary for data loading
            error_handler: Error handler instance for managing exceptions
        """
        self.config = config
        self.error_handler = error_handler
        self.default_path = config.get("default_path", "./data/sample_data.csv")
        self.allowed_formats = config.get("allowed_formats", ["csv", "xlsx", "json"])
        self.current_data = None
        self.metadata = {}
        
    def load_data(self, file_path: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Load data from the specified file path.
        
        Args:
            file_path: Path to the data file. If None, uses the default path.
            
        Returns:
            Loaded DataFrame or None if loading fails
        """
        try:
            path = file_path or self.default_path
            
            if not os.path.exists(path):
                raise FileNotFoundError(f"Data file not found: {path}")
            
            file_ext = os.path.splitext(path)[1].lower().replace(".", "")
            
            if file_ext not in self.allowed_formats:
                raise ValueError(f"Unsupported file format: {file_ext}. Allowed formats: {self.allowed_formats}")
            
            # Load data based on file extension
            if file_ext == "csv":
                data = pd.read_csv(path)
            elif file_ext == "xlsx":
                data = pd.read_excel(path)
            elif file_ext == "json":
                data = pd.read_json(path)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
            
            # Store data and extract metadata
            self.current_data = data
            self._extract_metadata()
            
            return data
            
        except Exception as e:
            self.error_handler.handle_error(e, f"Error loading data from {file_path}")
            return None
    
    def _extract_metadata(self) -> None:
        """
        Extract metadata from the loaded DataFrame.
        """
        if self.current_data is None:
            return
        
        try:
            df = self.current_data
            self.metadata = {
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "missing_values": df.isna().sum().to_dict(),
                "numerical_columns": list(df.select_dtypes(include=['int64', 'float64']).columns),
                "categorical_columns": list(df.select_dtypes(include=['object', 'category']).columns),
                "datetime_columns": list(df.select_dtypes(include=['datetime64']).columns)
            }
        except Exception as e:
            self.error_handler.handle_error(e, "Error extracting metadata")
    
    def get_data(self) -> Optional[pd.DataFrame]:
        """
        Get the currently loaded data.
        
        Returns:
            Current DataFrame or None if no data is loaded
        """
        return self.current_data
    
    def get_metadata(self) -> Dict:
        """
        Get metadata about the loaded data.
        
        Returns:
            Dictionary of metadata
        """
        return self.metadata
    
    def get_sample(self, n_rows: int = 5) -> Optional[pd.DataFrame]:
        """
        Get a sample of the loaded data.
        
        Args:
            n_rows: Number of rows to include in the sample
            
        Returns:
            Sample DataFrame or None if no data is loaded
        """
        if self.current_data is None:
            return None
        
        try:
            return self.current_data.head(n_rows)
        except Exception as e:
            self.error_handler.handle_error(e, "Error getting data sample")
            return None