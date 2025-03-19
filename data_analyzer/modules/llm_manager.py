"""
Module for managing LLM models through Ollama.
"""
import requests
import json
from typing import Dict, List, Optional, Any
import logging
from modules.error_handler import ErrorHandler

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LLMManager:
    """
    Manages LLM model selection and interaction using Ollama.
    """
    def __init__(self, config: Dict, error_handler: ErrorHandler):
        """
        Initialize the LLM manager with configuration.
        
        Args:
            config: Configuration dictionary for LLM settings
            error_handler: Error handler instance for managing exceptions
        """
        self.config = config
        logging.info("LLM Manager initialized")
        self.error_handler = error_handler
        self.provider = config.get("provider", "ollama")
        self.default_model = config.get("default_model", "llama2")
        self.alternative_models = config.get("alternative_models", ["mistral", "gemma"])
        self.temperature = config.get("temperature", 0.1)
        self.max_tokens = config.get("max_tokens", 512)
        self.timeout = config.get("timeout", 30)
        self.base_url = "http://localhost:11434/api"
        logging.info(f"config: {config}")
    
    def list_available_models(self) -> List[str]:
        """
        List available models from Ollama.
        
        Returns:
            List of available model names
        """
        try:
            response = requests.get(f"{self.base_url}/tags", timeout=self.timeout)
            response.raise_for_status()
            
            models_data = response.json()
            models = [model["name"] for model in models_data.get("models", [])]
            
            return models
        except Exception as e:
            self.error_handler.handle_error(e, "Error listing LLM models")
            return [self.default_model]  # Return default model if we can't get the list
    
    def get_completion(self, prompt: str, model: Optional[str] = None) -> str:
        """
        Get completion from the LLM.
        
        Args:
            prompt: Text prompt to send to the model
            model: Model to use (if None, uses default_model)
            
        Returns:
            Generated text response
        """
        model = model or self.default_model
        
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
            
            response = requests.post(
                f"{self.base_url}/generate",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            return response.json().get("response", "")
        except requests.Timeout:
            error_msg = f"LLM request timed out after {self.timeout} seconds"
            self.error_handler.handle_error(TimeoutError(error_msg), error_msg)
            return "Error: The model took too long to respond. Please try again or use a simpler query."
        except Exception as e:
            self.error_handler.handle_error(e, "Error getting LLM completion")
            return f"Error: Could not get a response from the model. {str(e)}"
    
    def check_ollama_availability(self) -> bool:
        """
        Check if Ollama server is available.
        
        Returns:
            True if server is available, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def create_system_prompt(self, context: Dict[str, Any]) -> str:
        """
        Create a system prompt with context information.
        
        Args:
            context: Dictionary containing context information
            
        Returns:
            Formatted system prompt
        """
        try:
            # Extract data metadata from context
            columns = context.get("columns", [])
            dtypes = context.get("dtypes", {})
            
            # Create a prompt that helps the model understand how to convert text to pandas
            prompt = """
            You are a data analyst assistant that converts natural language queries to pandas Python code.
            
            Your task is to convert the user's query into a valid pandas operation that can be executed.
            - Only respond with valid Python code for pandas.
            - Do not include any explanation or markdown formatting.
            - The code should start with 'result = ' and return a pandas DataFrame.
            - The dataframe is already loaded as 'df'.
            
            The available data has the following columns:
            """
            
            # Add column information
            for col in columns:
                dtype = dtypes.get(col, "unknown")
                prompt += f"- {col} ({dtype})\n"
            
            return prompt
        except Exception as e:
            self.error_handler.handle_error(e, "Error creating system prompt")
            return "You are a data analysis assistant that converts natural language to pandas Python code."