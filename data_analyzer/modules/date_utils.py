"""
Module for handling date-related operations.
"""
import pandas as pd
import re
from typing import Optional, Tuple, List
import streamlit as st
from modules.error_handler import ErrorHandler

class DateUtils:
    """Utilities for handling date operations in the application."""
    
    @staticmethod
    def detect_date_format(sample: str) -> Optional[str]:
        """
        Detect the format of a date string.
        
        Args:
            sample: A sample date string
            
        Returns:
            A date format string for strftime or None if format couldn't be detected
        """
        # Clean the sample string
        sample = sample.strip()
        
        # Check ISO format (YYYY-MM-DD)
        if re.match(r'^\d{4}-\d{2}-\d{2}$', sample):
            return '%Y-%m-%d'
        
        # Check ISO format with time (YYYY-MM-DD HH:MM:SS)
        if re.match(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$', sample):
            return '%Y-%m-%d %H:%M:%S'
        
        # Check ISO format with time and timezone (YYYY-MM-DD HH:MM:SS+HH:MM)
        if re.match(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', sample):
            return '%Y-%m-%dT%H:%M:%S'
        
        # Check MM/DD/YYYY
        if re.match(r'^\d{1,2}/\d{1,2}/\d{4}$', sample):
            return '%m/%d/%Y'
        
        # Check DD/MM/YYYY
        if re.match(r'^\d{1,2}/\d{1,2}/\d{4}$', sample):
            # This is ambiguous with MM/DD/YYYY, but we'll use DD/MM/YYYY as fallback
            return '%d/%m/%Y'
        
        # Check MM-DD-YYYY
        if re.match(r'^\d{1,2}-\d{1,2}-\d{4}$', sample):
            return '%m-%d-%Y'
        
        # Check DD-MM-YYYY
        if re.match(r'^\d{1,2}-\d{1,2}-\d{4}$', sample):
            # This is ambiguous with MM-DD-YYYY, but we'll use DD-MM-YYYY as fallback
            return '%d-%m-%Y'
        
        # Check DD.MM.YYYY
        if re.match(r'^\d{1,2}\.\d{1,2}\.\d{4}$', sample):
            return '%d.%m.%Y'
        
        # Check Month name formats
        if re.match(r'^[A-Za-z]{3,9} \d{1,2}, \d{4}$', sample):
            return '%B %d, %Y'  # e.g., "January 1, 2023"
        
        if re.match(r'^\d{1,2} [A-Za-z]{3,9} \d{4}$', sample):
            return '%d %B %Y'  # e.g., "1 January 2023"
        
        # Default to None if format can't be determined
        return None
    
    @staticmethod
    def convert_column_to_datetime(df: pd.DataFrame, column: str, error_handler: ErrorHandler) -> Tuple[pd.DataFrame, bool, Optional[str]]:
        """
        Convert a column to datetime format with appropriate format detection.
        
        Args:
            df: DataFrame containing the column
            column: Name of the column to convert
            error_handler: Error handler for logging issues
            
        Returns:
            Tuple of (DataFrame with converted column, success flag, error message if any)
        """
        if column not in df.columns:
            return df, False, f"Column '{column}' not found in the data"
        
        # If already datetime, just return
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            return df, True, None
        
        try:
            # Make a copy to avoid modifying the original
            df_copy = df.copy()
            
            # Get a sample of non-null values
            sample_values = df_copy[column].dropna()
            if len(sample_values) == 0:
                return df_copy, False, f"Column '{column}' contains only null values"
            
            # Try to detect format from the first few elements
            format_candidates = []
            for i in range(min(5, len(sample_values))):
                if not isinstance(sample_values.iloc[i], str):
                    # Convert to string first if it's not already
                    sample_str = str(sample_values.iloc[i])
                else:
                    sample_str = sample_values.iloc[i]
                
                detected_format = DateUtils.detect_date_format(sample_str)
                if detected_format and detected_format not in format_candidates:
                    format_candidates.append(detected_format)
            
            # If we found at least one format, try them in order
            if format_candidates:
                for date_format in format_candidates:
                    try:
                        df_copy[column] = pd.to_datetime(df_copy[column], format=date_format, errors="coerce")
                        # Check if conversion was successful (> 80% non-null)
                        if df_copy[column].notna().mean() >= 0.8:
                            return df_copy, True, None
                    except Exception as e:
                        error_handler.handle_error(e, f"Failed to convert column '{column}' with format '{date_format}'")
            
            # If specific formats didn't work, fall back to the default parser with a warning
            df_copy[column] = pd.to_datetime(df_copy[column], errors="coerce", format='%d-%b-%y')
            success_rate = df_copy[column].notna().mean()
            
            if success_rate >= 0.8:
                warning_msg = f"Used automatic date parsing for column '{column}', which may be inconsistent."
                return df_copy, True, warning_msg
            else:
                return df, False, f"Failed to convert column '{column}' to date format (success rate: {success_rate:.1%})"
                
        except Exception as e:
            error_msg = error_handler.handle_error(e, f"Error converting column '{column}' to datetime")
            return df, False, error_msg
    
    @staticmethod
    def find_date_columns(df: pd.DataFrame, error_handler: ErrorHandler) -> List[str]:
        """
        Find columns that are likely to contain date information.
        
        Args:
            df: DataFrame to analyze
            error_handler: Error handler for logging issues
            
        Returns:
            List of column names that are likely to contain dates
        """
        date_columns = []
        
        # First, check existing datetime columns
        date_columns.extend(df.select_dtypes(include=["datetime64"]).columns.tolist())
        
        # Check column names for date-related keywords
        date_keywords = ["date", "time", "day", "month", "year", "dt", "period"]
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in date_keywords) and col not in date_columns:
                # Try to convert and see if it works
                df_test, success, _ = DateUtils.convert_column_to_datetime(df, col, error_handler)
                if success:
                    date_columns.append(col)
        
        # Check string columns without date keywords
        for col in df.select_dtypes(include=["object"]).columns:
            if col not in date_columns:
                # Check if any value has date separators
                sample = df[col].dropna().astype(str).head(10)
                if sample.empty:
                    continue
                
                # Check for date separators in the sample
                has_separators = sample.str.contains('[-/.:T]').any()
                if has_separators:
                    # Try to convert
                    df_test, success, _ = DateUtils.convert_column_to_datetime(df, col, error_handler)
                    if success:
                        date_columns.append(col)
        
        return date_columns