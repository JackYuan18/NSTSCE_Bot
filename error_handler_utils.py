"""Utility functions for handling dependency and package errors."""

import sys
from typing import Optional


def is_dependency_error(error: Exception) -> bool:
    """
    Check if an error is related to missing dependencies or packages.
    
    Args:
        error: The exception to check
        
    Returns:
        True if the error is related to missing dependencies/packages
    """
    error_str = str(error).lower()
    error_type = type(error).__name__
    
    # Check for common dependency-related error types
    if error_type in ('ImportError', 'ModuleNotFoundError'):
        return True
    
    # Check for common dependency-related error messages
    dependency_keywords = [
        'no module named',
        'cannot import',
        'failed to import',
        'missing dependency',
        'package not found',
        'module not found',
        'cannot find module',
        'import error',
    ]
    
    return any(keyword in error_str for keyword in dependency_keywords)


def enhance_error_message(error: Exception, conda_env: str = 'rag-testing') -> str:
    """
    Enhance an error message with a prompt to activate the conda environment.
    
    Args:
        error: The exception that occurred
        conda_env: Name of the conda environment to activate (default: 'rag-testing')
        
    Returns:
        Enhanced error message with conda activation prompt
    """
    original_message = str(error)
    
    if is_dependency_error(error):
        conda_prompt = (
            f"\n\n"
            f"⚠️  DEPENDENCY ERROR DETECTED ⚠️\n"
            f"Please activate the conda environment '{conda_env}' before running this script:\n"
            f"  conda activate {conda_env}\n"
            f"\n"
            f"Original error: {original_message}"
        )
        return conda_prompt
    
    return original_message


def handle_dependency_error(error: Exception, conda_env: str = 'rag-testing') -> Exception:
    """
    Handle a dependency error by enhancing the message and re-raising with conda prompt.
    
    Args:
        error: The exception that occurred
        conda_env: Name of the conda environment to activate (default: 'rag-testing')
        
    Returns:
        A new exception with enhanced error message
    """
    if is_dependency_error(error):
        enhanced_message = enhance_error_message(error, conda_env)
        error_type = type(error)
        # Create a new exception of the same type with enhanced message
        return error_type(enhanced_message)
    return error

