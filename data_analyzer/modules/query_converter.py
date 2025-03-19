"""
Module for converting natural language queries to pandas operations.
"""
import pandas as pd
import re
import ast
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from modules.llm_manager import LLMManager
from modules.error_handler import ErrorHandler

class QueryConverter:
    """
    Converts natural language queries to pandas operations.
    """
    def __init__(self, llm_manager: LLMManager, error_handler: ErrorHandler):
        """
        Initialize the query converter.
        
        Args:
            llm_manager: LLM manager for generating pandas code
            error_handler: Error handler for managing exceptions
        """
        self.llm_manager = llm_manager
        self.error_handler = error_handler
    
    def convert_to_pandas(self, query: str, df_metadata: Dict) -> Tuple[str, bool]:
        """
        Convert natural language query to pandas code.
        
        Args:
            query: Natural language query from the user
            df_metadata: Metadata about the DataFrame
            
        Returns:
            Tuple of (pandas code, success flag)
        """
        try:
            # Create system prompt with DataFrame context
            system_prompt = self.llm_manager.create_system_prompt(df_metadata)
            
            # Combine system prompt and user query
            full_prompt = f"{system_prompt}\n\nUser query: {query}\n\nPandas code:"
            
            # Get code from LLM
            pandas_code = self.llm_manager.get_completion(full_prompt)
            
            # Clean up the code - remove markdown formatting if present
            pandas_code = self._clean_code(pandas_code)
            
            # Validate the code is safe to execute
            is_safe = self._validate_code_safety(pandas_code)
            
            return pandas_code, is_safe
        except Exception as e:
            self.error_handler.handle_error(e, "Error converting query to pandas")
            return "# Error generating pandas code", False
    
    def _clean_code(self, code: str) -> str:
        """
        Clean up code from LLM response.
        
        Args:
            code: Raw code from LLM
            
        Returns:
            Cleaned code
        """
        # Remove markdown code blocks if present
        code = re.sub(r'```python\s*', '', code)
        code = re.sub(r'```\s*', '', code)
        
        # Ensure code references DataFrame as 'df'
        code = code.replace('data.', 'df.')
        code = code.replace('dataframe.', 'df.')
        
        return code.strip()
    
    def _validate_code_safety(self, code: str) -> bool:
        """
        Validate that the generated code is safe to execute.
        
        Args:
            code: Code to validate
            
        Returns:
            True if code is safe, False otherwise
        """
        try:
            # Parse code to AST
            tree = ast.parse(code)
            
            # Check for potentially dangerous operations
            for node in ast.walk(tree):
                # Check for imports
                if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                    for name in node.names:
                        # Allow only pandas, numpy, and other safe modules
                        if name.name not in ['pandas', 'numpy', 'pd', 'np', 'matplotlib', 'plt']:
                            return False
                
                # Check for exec, eval, or other dangerous functions
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id in ['exec', 'eval', 'compile', 'open', 'delete', 'remove']:
                        return False
            
            return True
        except SyntaxError:
            # If code can't be parsed, it's not valid Python
            return False
        except Exception as e:
            self.error_handler.handle_error(e, "Error validating code safety")
            return False
    
    def execute_pandas_query(self, code: str, df: pd.DataFrame) -> Tuple[Any, str, bool]:
        """
        Execute pandas code on the DataFrame.
        
        Args:
            code: Pandas code to execute
            df: DataFrame to operate on
            
        Returns:
            Tuple of (result, error message, success flag)
        """
        try:
            # Create a local namespace with the DataFrame
            local_vars = {"df": df, "pd": pd, "np": __import__("numpy")}
            
            # Execute the code
            exec(code, globals(), local_vars)
            
            # Try to get the result - check common variable names
            result = None
            for var_name in ["result", "output", "df_result", "filtered", "grouped", "summary"]:
                if var_name in local_vars:
                    result = local_vars[var_name]
                    break
            
            # If no explicit result variable found, use the modified DataFrame
            if result is None and "df" in local_vars:
                # Check if df was modified
                if not df.equals(local_vars["df"]):
                    result = local_vars["df"]
            
            return result, "", True
        except Exception as e:
            error_msg = f"Error executing pandas query: {str(e)}"
            self.error_handler.handle_error(e, error_msg)
            return None, error_msg, False