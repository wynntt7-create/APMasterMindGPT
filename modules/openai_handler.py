"""
OpenAI API integration for AP Calculus BC AI Mastermind
"""

import os
import base64
import json
import re
from typing import List, Dict, Optional, Any
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
    Send a message to GPT-5 mini and get a response.
    
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
            model="gpt-5-mini",
            messages=api_messages,
            temperature=1,
            max_completion_tokens=2000
        )
        
        # Extract and return the response text
        return response.choices[0].message.content
        
    except Exception as e:
        error_msg = f"Error calling OpenAI API: {str(e)}"
        raise Exception(error_msg)


def _extract_json_object(text: str) -> Dict[str, Any]:
    """
    Best-effort extraction of a single JSON object from model output.
    """
    if not text:
        raise ValueError("Empty response (expected JSON).")

    # If it's already valid JSON, accept it.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # Try to find the first {...} block.
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError("Model response did not contain a JSON object.")

    candidate = m.group(0)
    parsed = json.loads(candidate)
    if not isinstance(parsed, dict):
        raise ValueError("Parsed JSON was not an object.")
    return parsed


def analyze_image_problem(
    image_bytes: bytes,
    mime_type: str,
    unit_focus: Optional[str] = None,
    solve: bool = True,
) -> Dict[str, str]:
    """
    Use a vision-capable model to read a math/word problem from an uploaded image,
    convert it to clean LaTeX, and optionally solve it.

    Returns a dict with:
      - extracted_text
      - problem_latex
      - clean_latex_code
      - solution_markdown
    """
    if not image_bytes:
        raise ValueError("No image bytes provided.")
    mime_type = (mime_type or "").strip().lower() or "image/png"

    system_prompt = get_system_prompt(unit_focus)
    task_prompt = f"""
You are given an image that contains a math or word problem (often AP Calculus BC).
Your job is to:
1) Read the problem in the image (OCR).
2) Convert the problem statement into clean LaTeX.
3) {"Solve it with clear steps and a final answer." if solve else "Do NOT solve it."}

Return ONLY a single JSON object with exactly these keys:
- "extracted_text": string (plain-text transcription; preserve math meaning)
- "problem_latex": string (LaTeX for the problem statement only; no solution)
- "clean_latex_code": string (copy-ready LaTeX snippet. Include the problem in a display math / align environment where appropriate)
- "solution_markdown": string (a student-friendly solution in Markdown with LaTeX math where needed; if solve is false, set this to an empty string)

Rules:
- Do not wrap the JSON in markdown fences.
- If the image is too blurry/ambiguous, still return JSON but explain uncertainty in "solution_markdown" and include your best transcription.
""".strip()

    b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{mime_type};base64,{b64}"

    try:
        response = _client.chat.completions.create(
            model="gpt-5-mini",
            temperature=1,
            max_completion_tokens=2000,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": task_prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
        )

        raw = response.choices[0].message.content or ""
        obj = _extract_json_object(raw)

        # Normalize expected keys (avoid KeyError if model misses one).
        return {
            "extracted_text": str(obj.get("extracted_text", "")).strip(),
            "problem_latex": str(obj.get("problem_latex", "")).strip(),
            "clean_latex_code": str(obj.get("clean_latex_code", "")).strip(),
            "solution_markdown": str(obj.get("solution_markdown", "")).strip(),
        }

    except Exception as e:
        error_msg = f"Error calling OpenAI vision API: {str(e)}"
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

