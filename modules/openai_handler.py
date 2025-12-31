"""
OpenAI API integration for AP Calculus BC AI Mastermind
"""

import os
from typing import List, Dict, Optional
from dotenv import load_dotenv
from openai import OpenAI
from modules.prompts import get_system_prompt

# Load environment variables
load_dotenv()

# Initialize OpenAI client
_api_key = os.getenv("OPENAI_API_KEY")
if not _api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please create a .env file with your API key.")

_client = OpenAI(api_key=_api_key)


def send_message(
    messages: List[Dict[str, str]],
    unit_focus: Optional[str] = None
) -> str:
    """
    Send a message to GPT-4o-mini and get a response.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        unit_focus: Optional unit focus to enhance the system prompt
    
    Returns:
        AI response text
    
    Raises:
        Exception: If API call fails
    """
    try:
        # Build system prompt based on unit focus
        system_prompt = get_system_prompt(unit_focus)
        
        # Prepare messages with system prompt
        api_messages = [
            {"role": "system", "content": system_prompt}
        ] + messages
        
        # Call OpenAI API
        response = _client.chat.completions.create(
            model="gpt-4o-mini",
            messages=api_messages,
            temperature=0.7,
            max_tokens=2000
        )
        
        # Extract and return the response text
        return response.choices[0].message.content
        
    except Exception as e:
        error_msg = f"Error calling OpenAI API: {str(e)}"
        raise Exception(error_msg)


def validate_api_key() -> bool:
    """
    Validate that the API key is set and working.
    
    Returns:
        True if API key is valid, False otherwise
    """
    try:
        if not _api_key:
            return False
        
        # Test with a simple API call
        _client.models.list()
        return True
    except Exception:
        return False

