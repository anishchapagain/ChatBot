"""
Module for analyzing data and generating statistics and visualizations.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional, Tuple, Any
from modules.error_handler import ErrorHandler

class DataAnalyzer:
    """
    Analyzes data and generates statistics and visualizations.
    """
    def __init__(self, error_handler: ErrorHandler):
        """
        Initialize the data analyzer.
        
        Args:
            error_handler: Error handler for managing exceptions
        """
        self.error_handler = error_handler
    
    def generate_summary_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate summary statistics for the DataFrame.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary of summary statistics
        """
        try:
            # Basic DataFrame information
            summary = {
                "row_count": len(df),
                "column_count": len(df.columns),
                "memory_usage": df.memory_usage(deep=True).sum() / 1024**2,  # in MB
            }
            
            # Numerical column statistics
            numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns
            if not numerical_columns.empty:
                num_stats = df[numerical_columns].describe().to_dict()
                summary["numerical_stats"] = num_stats
            
            # Categorical column statistics
            categorical_columns = df.select_dtypes(include=["object", "category"]).columns
            cat_stats = {}
            for col in categorical_columns:
                try:
                    value_counts = df[col].value_counts().to_dict()
                    unique_count = len(value_counts)
                    top_categories = {k: v for i, (k, v) in enumerate(value_counts.items()) if i < 5}
                    
                    cat_stats[col] = {
                        "unique_count": unique_count,
                        "top_categories": top_categories,
                        "null_count": df[col].isna().sum()
                    }
                except Exception as e:
                    self.error_handler.handle_error(e, f"Error analyzing column {col}")
            
            summary["categorical_stats"] = cat_stats
            
            # Missing value analysis
            missing_values = df.isna().sum().to_dict()
            summary["missing_values"] = {k: v for k, v in missing_values.items() if v > 0}
            
            return summary
        
        except Exception as e:
            self.error_handler.handle_error(e, "Error generating summary statistics")
            return {"error": str(e)}
    
    def create_correlation_matrix(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Create a correlation matrix for numerical columns.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with correlation matrix data for plotting
        """
        try:
            # Get numerical columns
            numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns
            
            if len(numerical_columns) < 2:
                return None
            
            # Calculate correlation matrix
            corr_matrix = df[numerical_columns].corr().round(2)
            
            # Convert to format suitable for heatmap
            z_data = corr_matrix.values
            x_data = corr_matrix.columns
            y_data = corr_matrix.index
            
            return {
                "z": z_data.tolist(),
                "x": x_data.tolist(),
                "y": y_data.tolist()
            }
        
        except Exception as e:
            self.error_handler.handle_error(e, "Error creating correlation matrix")
            return None
    
    def create_distribution_plots(self, df: pd.DataFrame, max_columns: int = 6) -> List[Dict]:
        """
        Create distribution plots for numerical columns.
        
        Args:
            df: DataFrame to analyze
            max_columns: Maximum number of columns to create plots for
            
        Returns:
            List of dictionaries with plot data
        """
        try:
            # Get numerical columns
            numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns
            
            # Limit to max_columns
            if len(numerical_columns) > max_columns:
                numerical_columns = numerical_columns[:max_columns]
            
            plot_data = []
            for col in numerical_columns:
                try:
                    # Basic statistics
                    mean_val = df[col].mean()
                    median_val = df[col].median()
                    
                    # Create histogram data
                    counts, bins = np.histogram(df[col].dropna(), bins=20)
                    bins_center = (bins[:-1] + bins[1:]) / 2
                    
                    plot_data.append({
                        "column": col,
                        "type": "histogram",
                        "x": bins_center.tolist(),
                        "y": counts.tolist(),
                        "mean": mean_val,
                        "median": median_val
                    })
                except Exception as e:
                    self.error_handler.handle_error(e, f"Error creating distribution plot for {col}")
            
            return plot_data
        
        except Exception as e:
            self.error_handler.handle_error(e, "Error creating distribution plots")
            return []
    
    def create_category_plots(self, df: pd.DataFrame, max_columns: int = 6, max_categories: int = 10) -> List[Dict]:
        """
        Create bar charts for categorical columns.
        
        Args:
            df: DataFrame to analyze
            max_columns: Maximum number of columns to create plots for
            max_categories: Maximum number of categories to include per plot
            
        Returns:
            List of dictionaries with plot data
        """
        try:
            # Get categorical columns
            categorical_columns = df.select_dtypes(include=["object", "category"]).columns
            
            # Limit to max_columns
            if len(categorical_columns) > max_columns:
                categorical_columns = categorical_columns[:max_columns]
            
            plot_data = []
            for col in categorical_columns:
                try:
                    # Get value counts
                    value_counts = df[col].value_counts().reset_index()
                    value_counts.columns = ["category", "count"]
                    
                    # Limit to max_categories
                    if len(value_counts) > max_categories:
                        # Keep top categories and group others
                        top_counts = value_counts.iloc[:max_categories-1]
                        other_count = value_counts.iloc[max_categories-1:]["count"].sum()
                        
                        # Add "Other" category
                        other_row = pd.DataFrame({"category": ["Other"], "count": [other_count]})
                        value_counts = pd.concat([top_counts, other_row], ignore_index=True)
                    
                    plot_data.append({
                        "column": col,
                        "type": "bar",
                        "x": value_counts["category"].tolist(),
                        "y": value_counts["count"].tolist()
                    })
                except Exception as e:
                    self.error_handler.handle_error(e, f"Error creating category plot for {col}")
            
            return plot_data
        
        except Exception as e:
            self.error_handler.handle_error(e, "Error creating category plots")
            return []
    
    def create_time_series_plot(self, df: pd.DataFrame, date_column: str, value_column: str) -> Optional[Dict]:
        """
        Create a time series plot. This method is deprecated in favor of direct plotting in the dashboard.
        
        Args:
            df: DataFrame to analyze
            date_column: Column name containing dates
            value_column: Column name containing values to plot
            
        Returns:
            Dictionary with plot data or None if invalid
        """
        # This method is now deprecated and reimplemented in the dashboard
        # for better date handling
        return None
            
    def detect_possible_date_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Detect columns that might contain date information.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            List of column names that might contain dates
        """
        date_columns = []
        
        # Check datetime columns
        date_columns.extend(df.select_dtypes(include=["datetime64"]).columns.tolist())
        
        # Check string columns that might contain dates
        for col in df.select_dtypes(include=["object"]).columns:
            try:
                # Try to convert a sample to datetime
                sample = df[col].dropna().head(10)
                if sample.empty:
                    continue
                
                # Check if field contains common date separators
                has_date_separators = sample.astype(str).str.contains('[-/:]').any()
                if not has_date_separators:
                    continue
                    
                # Try converting to datetime with specific format detection
                first_value = str(sample.iloc[0])
                
                # Check common date formats
                if len(first_value) >= 10:
                    if '-' in first_value:  # YYYY-MM-DD or similar
                        converted = pd.to_datetime(sample, format='%Y-%m-%d', errors='coerce')
                    elif '/' in first_value:  # MM/DD/YYYY or similar
                        if first_value[2] == '/' and first_value[5] == '/':  # MM/DD/YYYY
                            converted = pd.to_datetime(sample, format='%m/%d/%Y', errors='coerce')
                        else:  # Try common alternatives
                            converted = pd.to_datetime(sample, format='%d/%m/%Y', errors='coerce')
                    else:  # Fall back to parser
                        converted = pd.to_datetime(sample, errors='coerce')
                else:
                    continue  # Skip if format doesn't look like a date
                
                # If at least 80% converted successfully, consider it a date column
                if converted.notna().mean() >= 0.8:
                    date_columns.append(col)
            except Exception:
                pass
        
        return date_columns