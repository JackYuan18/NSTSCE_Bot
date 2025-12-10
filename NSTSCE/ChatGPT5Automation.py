"""
ChatGPT 5 API Integration Module

This module handles integration with ChatGPT 5 through the OpenAI API.
It provides a simple interface for generating responses using the API.
"""

import logging
import os
from typing import Optional, Tuple
from openai import OpenAI


class ChatGPT5Automation:
    """Handles integration with ChatGPT 5 via OpenAI API."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self) -> bool:
        """Initialize the OpenAI client."""
        try:
            if not self.api_key:
                self.logger.error("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
                return False
            
            # Clear any proxy environment variables that might interfere with OpenAI client initialization
            # OpenAI API 1.0+ doesn't support proxies parameter in constructor
            import os
            import httpx
            proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']
            original_proxies = {}
            for var in proxy_vars:
                if var in os.environ:
                    original_proxies[var] = os.environ.pop(var)
            
            try:
                # The issue is that OpenAI's internal code tries to pass 'proxies' (plural) to httpx.Client
                # but httpx only accepts 'proxy' (singular). We need to monkey-patch httpx.Client
                # to filter out the 'proxies' argument before it reaches httpx
                original_init = httpx.Client.__init__
                
                def patched_init(self, *args, **kwargs):
                    # Remove 'proxies' if present (OpenAI might pass it)
                    kwargs.pop('proxies', None)
                    # Also ensure trust_env is False to prevent reading proxy env vars
                    kwargs['trust_env'] = False
                    return original_init(self, *args, **kwargs)
                
                # Apply the patch
                httpx.Client.__init__ = patched_init
                
                try:
                    # Now create the OpenAI client - it should work without the proxies error
                    self.client = OpenAI(api_key=self.api_key)
                    self.logger.info("OpenAI client initialized successfully")
                    return True
                finally:
                    # Restore original httpx.Client.__init__
                    httpx.Client.__init__ = original_init
            finally:
                # Restore proxy environment variables
                for var, value in original_proxies.items():
                    os.environ[var] = value
            
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def generate_response(self, query: str, context: str) -> Tuple[bool, str]:
        """
        Generate a response using ChatGPT 5 API.
        
        Args:
            query: The user's question
            context: The retrieved context from documents
            
        Returns:
            Tuple of (success, response)
        """
        try:
            if not self.client:
                self.logger.error("OpenAI client not initialized")
                return False, "OpenAI client not initialized"
            
            # Create the prompt with context
            prompt = f"""Context: {context}

Question: {query}

Please provide a detailed answer based on the context provided. If the context doesn't contain enough information to answer the question, please say so and provide what information you can based on the available context."""

            # Call the OpenAI API
            response = self.client.chat.completions.create(
                model="gpt-4",  # Using GPT-4 as ChatGPT 5 equivalent
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context. Always be accurate and cite information from the context when possible."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            # Extract the response
            answer = response.choices[0].message.content.strip()
            
            self.logger.info("Successfully generated response using ChatGPT 5 API")
            return True, answer
            
        except Exception as e:
            self.logger.error(f"Failed to generate response with ChatGPT 5 API: {e}")
            return False, f"API Error: {str(e)}"
    
    def validate_api_key(self) -> Tuple[bool, str]:
        """
        Validates the API key by making a simple test request.
        Returns (is_valid: bool, message: str).
        """
        # Try to initialize client if not already initialized
        if not self.client:
            if not self.api_key:
                return False, "OpenAI API key not provided. Please check your API key."
            # Retry initialization
            try:
                if not self._initialize_client():
                    return False, "OpenAI client not initialized. Please check your API key and ensure it's valid."
            except Exception as e:
                return False, f"Failed to initialize OpenAI client: {str(e)}"
        
        try:
            # Make a simple test request to validate the API key
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",  # Use a cheaper model for validation
                messages=[
                    {"role": "user", "content": "Hello"}
                ],
                max_tokens=5
            )
            
            self.logger.info("API key validation successful")
            return True, "API key is valid"
            
        except Exception as e:
            error_msg = f"API key validation failed: {str(e)}"
            self.logger.error(error_msg)
            return False, error_msg

    def test_connection(self) -> bool:
        """Test the API connection."""
        try:
            if not self.client:
                return False
            
            # Make a simple test call
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            
            self.logger.info("API connection test successful")
            return True
            
        except Exception as e:
            self.logger.error(f"API connection test failed: {e}")
            return False
    
    def set_api_key(self, api_key: str) -> bool:
        """Set a new API key and reinitialize the client."""
        try:
            self.api_key = api_key
            self.client = OpenAI(api_key=api_key)
            self.logger.info("API key updated successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to update API key: {e}")
            return False