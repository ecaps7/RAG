"""LLM utilities for output processing."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional


def parse_llm_output(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort parsing of LLM output.
    
    This function attempts to parse various formats of LLM output into JSON.
    
    Args:
        text: The raw LLM output text.
        
    Returns:
        Parsed JSON data if successful, None otherwise.
    """
    if not text:
        return None
    
    # Try direct JSON parsing first
    try:
        return json.loads(text)
    except Exception:
        pass
    
    # Try to extract JSON from markdown or other wrappers
    try:
        # Extract JSON block with regex
        json_match = re.search(r"\{[\s\S]*\}", text)
        if json_match:
            return json.loads(json_match.group(0))
    except Exception:
        pass
    
    # Try to extract array from markdown or other wrappers
    try:
        # Extract JSON array with regex
        array_match = re.search(r"\[[\s\S]*\]", text)
        if array_match:
            return {"results": json.loads(array_match.group(0))}
    except Exception:
        pass
    
    # Try to fix common formatting issues
    try:
        # Replace single quotes with double quotes
        fixed = re.sub(r"'(\s*[:\]},])", r'"\1', text)
        fixed = fixed.replace("'", '"')
        return json.loads(fixed)
    except Exception:
        pass
    
    return None
