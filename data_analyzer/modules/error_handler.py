"""
Module for handling and logging errors throughout the application.
"""
import logging
import sys
import traceback
import os
from datetime import datetime
from typing import Dict, Optional, Any

class ErrorHandler:
    """
    Handles errors and exceptions in the application.
    """
    def __init__(self, config: Dict):
        """
        Initialize the error handler with configuration.
        
        Args:
            config: Configuration dictionary for error handling
        """
        self.config = config
        self.debug_mode = config.get("debug_mode", False)
        
        # Set up logging
        log_file = config.get("log_file", "./logs/app.log")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        logging.basicConfig(
            filename=log_file,
            level=logging.DEBUG if self.debug_mode else logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Also log to console
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        
        self.logger = logging.getLogger('data_analyzer')
        self.logger.addHandler(console_handler)
    
    def handle_error(self, exception: Exception, message: str = "", context: Optional[Dict[str, Any]] = None) -> str:
        """
        Handle an exception, log it, and return a user-friendly message.
        
        Args:
            exception: The exception that occurred
            message: Additional context message
            context: Additional context data
            
        Returns:
            User-friendly error message
        """
        try:
            # Get exception details
            exc_type = type(exception).__name__
            exc_message = str(exception)
            
            # Format the log message
            log_message = f"{message}: {exc_type} - {exc_message}"
            
            # Add traceback in debug mode
            if self.debug_mode:
                tb = traceback.format_exc()
                log_message += f"\n{tb}"
            
            # Log with appropriate level based on exception type
            if isinstance(exception, (ValueError, TypeError, KeyError)):
                self.logger.warning(log_message)
            else:
                self.logger.error(log_message)
            
            # Log additional context if provided
            if context:
                self.logger.info(f"Error context: {context}")
            
            # Return user-friendly message
            if self.debug_mode:
                # In debug mode, return full details
                return f"Error: {exc_type} - {exc_message}"
            else:
                # In production, return simplified message
                if isinstance(exception, (ValueError, KeyError)):
                    # For expected errors, show the message
                    return f"Error: {exc_message}"
                else:
                    # For unexpected errors, show a generic message
                    return "An error occurred. Please check the application logs for details."
                
        except Exception as e:
            # Fallback if error handling itself fails
            print(f"Error in error handler: {str(e)}")
            return "An unexpected error occurred."
    
    def log_info(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Log an informational message.
        
        Args:
            message: Message to log
            context: Additional context data
        """
        self.logger.info(message)
        if context:
            self.logger.info(f"Context: {context}")
    
    def log_warning(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a warning message.
        
        Args:
            message: Message to log
            context: Additional context data
        """
        self.logger.warning(message)
        if context:
            self.logger.warning(f"Context: {context}")
    
    def get_traceback(self, exception: Exception) -> str:
        """
        Get formatted traceback for an exception.
        
        Args:
            exception: Exception to get traceback for
            
        Returns:
            Formatted traceback
        """
        return traceback.format_exception(type(exception), exception, exception.__traceback__)