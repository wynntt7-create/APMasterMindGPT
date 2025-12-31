"""
Graphing engine for AP Calculus BC functions using Desmos Calculator
"""

import sympy as sp
from typing import Tuple, Optional
import re
import urllib.parse
import uuid
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables for DESMOS_API_KEY when this module is imported directly.
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")


def generate_calc_plot(
    equation_string: str,
    x_range: Tuple[float, float] = (-10, 10),
    title: Optional[str] = None
) -> str:
    """
    Generate a Desmos calculator embed HTML for a given mathematical function.
    Supports Cartesian (y = f(x)), polar (r = f(theta)), and parametric equations.
    
    Args:
        equation_string: String representation of the function (e.g., "x^2", "r = 1 + cos(theta)", "y = x^2 + 2*x + 1")
        x_range: Tuple of (min, max) x-values for the plot
        title: Optional title for the plot
    
    Returns:
        HTML string ready for Streamlit components.v1.html display
    
    Raises:
        ValueError: If equation cannot be parsed
    """
    try:
        # Clean up the equation string
        original_input = equation_string.strip()
        equation_string = original_input
        
        # Detect equation type
        is_polar = re.match(r'^r\s*=', equation_string, re.IGNORECASE)
        is_cartesian_y = re.match(r'^y\s*=', equation_string, re.IGNORECASE)
        is_function = re.match(r'^[a-zA-Z]\(x\)\s*=', equation_string, re.IGNORECASE)
        
        def _latexify_for_desmos(expr_str: str) -> str:
            """
            Convert common plain-text tokens to Desmos-friendly LaTeX tokens.

            IMPORTANT: When using re.sub, replacement strings treat backslashes as escape
            sequences (e.g., "\\c" is invalid). Use a replacement function to emit
            literal backslashes safely.
            """
            out = expr_str
            # theta -> \theta
            out = re.sub(r'(?<!\\)\btheta\b', lambda _: r'\theta', out, flags=re.IGNORECASE)
            # trig funcs -> \cos, \sin, \tan
            out = re.sub(r'(?<!\\)\bcos\b', lambda _: r'\cos', out, flags=re.IGNORECASE)
            out = re.sub(r'(?<!\\)\bsin\b', lambda _: r'\sin', out, flags=re.IGNORECASE)
            out = re.sub(r'(?<!\\)\btan\b', lambda _: r'\tan', out, flags=re.IGNORECASE)
            return out

        # Handle polar equations differently - skip SymPy parsing, use Desmos LaTeX directly
        if is_polar:
            # Extract the expression after "r ="
            polar_expr = re.sub(r'^r\s*=\s*', '', equation_string, flags=re.IGNORECASE).strip()
            # Clean up but preserve LaTeX notation (Desmos understands LaTeX)
            polar_expr = _latexify_for_desmos(polar_expr)
            # Generate Desmos HTML for polar equation (no SymPy parsing needed)
            return _generate_desmos_html(polar_expr, x_range, title, equation_type='polar')
        
        # Check if equation contains LaTeX notation (like \cos, \sin, \theta, etc.)
        # or if it contains theta (indicating polar coordinates)
        # If it does, we should pass it directly to Desmos without SymPy parsing
        has_latex = bool(re.search(r'\\[a-zA-Z]+', equation_string))
        has_theta = bool(re.search(r'\btheta\b', equation_string, re.IGNORECASE))
        
        # If it has theta, it's likely a polar equation (even without "r =")
        if has_theta and not is_polar:
            # Treat as polar equation
            polar_expr = equation_string.strip()
            # Ensure LaTeX notation
            polar_expr = _latexify_for_desmos(polar_expr)
            return _generate_desmos_html(polar_expr, x_range, title, equation_type='polar')
        
        # Remove common prefixes and notation for Cartesian equations
        equation_string = re.sub(r'f\(x\)\s*=\s*', '', equation_string)
        equation_string = re.sub(r'y\s*=\s*', '', equation_string)
        equation_string = re.sub(r'∫\s*\d*x\s*dx\s*=\s*', '', equation_string)  # Remove integral notation like "∫ 2x dx ="
        equation_string = re.sub(r'[+\-]\s*C\s*$', '', equation_string)  # Remove + C at the end
        
        # If equation contains LaTeX, pass it directly to Desmos (which understands LaTeX)
        if has_latex:
            # Clean up LaTeX notation for Desmos
            # Ensure common functions are in LaTeX format
            equation_string = _latexify_for_desmos(equation_string)
            # Pass directly to Desmos without SymPy parsing
            return _generate_desmos_html(equation_string.strip(), x_range, title, equation_type='cartesian')
        
        # Step 1: Convert ^ to ** for exponentiation (SymPy uses **)
        # Do this FIRST before handling implicit multiplication
        equation_string = re.sub(r'\^', '**', equation_string)
        
        # Step 2: Handle implicit multiplication: number followed by variable (e.g., 2x -> 2*x, 2x**1 -> 2*x**1)
        # Pattern: one or more digits followed directly by a letter
        # This handles cases like: 2x, 3x**2, 5sin(x), etc.
        equation_string = re.sub(r'(\d+)([a-zA-Z])', r'\1*\2', equation_string)
        
        # Step 3: Handle variable followed by number for exponentiation (e.g., x2 -> x**2)
        # But skip if there's already a * operator (to avoid double conversion)
        # Pattern: letter followed by digits, but NOT if followed by * (which means it's already handled)
        # We use a negative lookahead to ensure we don't match things like "x**2" or "x*2"
        equation_string = re.sub(r'([a-zA-Z])(\d+)(?![*\d])', r'\1**\2', equation_string)
        
        # Fix cases where we incorrectly converted (e.g., 2**x should be 2*x if original was 2x)
        # But preserve 2**x if original was 2^x
        # Actually, this is tricky - let's be more conservative
        
        # Clean up: remove any spaces around operators for cleaner parsing
        equation_string = re.sub(r'\s+', '', equation_string)
        
        # Store original for error messages (before final space removal)
        original_eq = equation_string
        
        # Parse the equation using SymPy to validate and convert to LaTeX
        x = sp.Symbol('x')
        try:
            # Try parsing with transformations enabled
            expr = sp.sympify(equation_string, convert_xor=False, evaluate=False)
        except Exception as parse_error:
            # If that fails, try with evaluate=True
            try:
                expr = sp.sympify(equation_string, convert_xor=False, evaluate=True)
            except Exception as parse_error2:
                # Last attempt: try more aggressive preprocessing
                fallback_eq = equation_string
                fallback_eq = re.sub(r'(\d+)([a-zA-Z])', r'\1*\2', fallback_eq)
                fallback_eq = re.sub(r'([a-zA-Z])(\d+)(?![*\d])', r'\1**\2', fallback_eq)
                
                try:
                    expr = sp.sympify(fallback_eq, convert_xor=False)
                except Exception as parse_error3:
                    error_msg = str(parse_error3)
                    raise ValueError(f"Could not parse expression '{original_eq}'. "
                                   f"After preprocessing: '{equation_string}' -> '{fallback_eq}'. "
                                   f"Error: {error_msg}")
        
        # Convert SymPy expression to Desmos-compatible format
        # Desmos uses standard math notation, so we'll convert from SymPy format
        desmos_expr = _convert_to_desmos_format(expr, x)
        
        # Generate HTML with embedded Desmos calculator (Cartesian by default)
        html_content = _generate_desmos_html(desmos_expr, x_range, title, equation_type='cartesian')
        
        return html_content
        
    except Exception as e:
        # Return error HTML
        error_html = f"""
        <div style="padding: 20px; border: 2px solid #ff6b6b; border-radius: 5px; background-color: #ffe0e0;">
            <h3 style="color: #c92a2a;">Error plotting function</h3>
            <p style="color: #862e2e;">{str(e)}</p>
        </div>
        """
        return error_html


def _convert_to_desmos_format(expr: sp.Expr, x: sp.Symbol) -> str:
    """
    Convert SymPy expression to Desmos-compatible LaTeX format.
    Desmos accepts LaTeX notation, which is more reliable for complex expressions.
    
    Args:
        expr: SymPy expression
        x: Symbol for x variable
    
    Returns:
        String in Desmos LaTeX format
    """
    # Convert SymPy expression to LaTeX (Desmos supports LaTeX)
    latex_str = sp.latex(expr)
    
    # Clean up LaTeX for Desmos compatibility
    # Remove \left and \right (Desmos handles parentheses automatically)
    latex_str = re.sub(r'\\left|\\right', '', latex_str)
    # Replace \cdot with * (Desmos prefers * for multiplication)
    latex_str = re.sub(r'\\cdot', '*', latex_str)
    
    return latex_str


def _generate_desmos_html(equation: str, x_range: Tuple[float, float], title: Optional[str] = None, equation_type: str = 'cartesian') -> str:
    """
    Generate HTML with embedded Desmos calculator using their JavaScript API.
    
    Args:
        equation: Equation string in Desmos format
        x_range: Tuple of (min, max) x-values
        title: Optional title
        equation_type: Type of equation - 'cartesian', 'polar', or 'parametric'
    
    Returns:
        HTML string for embedding
    """
    # Create a unique ID for this calculator instance
    calc_id = f"desmos-calc-{uuid.uuid4().hex[:8]}"

    # Desmos API key: required as a query param on calculator.js (otherwise it will 403)
    # Set DESMOS_API_KEY in your environment/.env (or Streamlit secrets in production).
    desmos_api_key = os.getenv("DESMOS_API_KEY", "").strip()
    
    # Escape the equation for JavaScript
    equation_js = equation.replace('\\', '\\\\').replace("'", "\\'").replace('"', '\\"')
    
    # Determine the LaTeX expression based on equation type
    if equation_type == 'polar':
        latex_expr = f'r = {equation_js}'
        display_expr = f'r = {equation}'
    else:  # cartesian
        latex_expr = f'y = {equation_js}'
        display_expr = f'y = {equation}'
    
    # Generate HTML with Desmos Graph API
    html = f"""
    <div style="width: 100%; margin: 10px 0;">
        {f'<h3 style="margin-bottom: 10px; color: #333;">{title}</h3>' if title else ''}
        <div id="{calc_id}" style="width: 100%; height: 500px; border: 1px solid #ddd; border-radius: 5px; margin: 10px 0;"></div>
        <p style="font-size: 12px; color: #666; margin-top: 5px;">
            Function: <code style="background: #f0f0f0; padding: 2px 5px; border-radius: 3px;">{display_expr}</code>
        </p>
        <script>
            (function() {{
                function showError(msg) {{
                    var elt = document.getElementById('{calc_id}');
                    if (!elt) return;
                    elt.innerHTML = '<div style="padding:12px;color:#842029;background:#f8d7da;border:1px solid #f5c2c7;border-radius:6px;">'
                        + '<strong>Desmos failed to load.</strong><div style="margin-top:6px; font-family: monospace; white-space: pre-wrap;">'
                        + (msg || 'Unknown error') + '</div></div>';
                }}

                function init() {{
                    if (typeof Desmos === 'undefined') {{
                        showError('Desmos global was not defined after loading calculator.js');
                        return;
                    }}
                    var elt = document.getElementById('{calc_id}');
                    var calculator = Desmos.GraphingCalculator(elt, {{
                        keypad: true,
                        zoomButtons: true,
                        expressions: true,
                        settingsMenu: true,
                        xAxisStep: 1,
                        yAxisStep: 1,
                        xAxisLabel: 'x',
                        yAxisLabel: 'y',
                        xAxisDomain: [{x_range[0]}, {x_range[1]}]
                    }});

                    calculator.setExpression({{
                        id: 'graph1',
                        latex: '{latex_expr}',
                        color: Desmos.Colors.BLUE
                    }});
                }}

                // Load Desmos script with explicit callbacks (more reliable in embedded iframes)
                if (!'{desmos_api_key}') {{
                    showError('DESMOS_API_KEY is not set.\\n\\nCreate a .env file with:\\nDESMOS_API_KEY=YOUR_DESMOS_KEY\\n(or set it in Streamlit secrets).');
                    return;
                }}
                var script = document.createElement('script');
                script.src = 'https://www.desmos.com/api/v1.11/calculator.js?apiKey={urllib.parse.quote(desmos_api_key)}';
                script.onload = init;
                script.onerror = function() {{
                    showError('Failed to load ' + script.src + '.\\nThis is usually caused by a missing/invalid DESMOS_API_KEY or a network filter blocking desmos.com.');
                }};
                document.head.appendChild(script);
            }})();
        </script>
    </div>
    """
    
    return html


def extract_equation_from_text(text: str) -> Optional[str]:
    """
    Try to extract a mathematical equation from text.
    Handles various formats: f(x) = ..., y = ..., r = ..., etc.
    
    Args:
        text: Text that may contain an equation
    
    Returns:
        Extracted equation string or None if not found
    """
    # Common patterns for equations - order matters (more specific first)
    patterns = [
        # Polar equations: r = 1 + cos(theta) or r = 2*sin(3*theta)
        r'r\s*=\s*([^\s,;.]+(?:\s*[+\-*/]\s*[^\s,;.]+)*)',
        # Function notation: f(x) = x^2 + 1
        r'f\(x\)\s*=\s*([^\s,;.]+(?:\s*[+\-*/]\s*[^\s,;.]+)*)',
        # Explicit y = format: y = x^2 + 2*x + 1
        r'y\s*=\s*([^\s,;.]+(?:\s*[+\-*/]\s*[^\s,;.]+)*)',
        # Generic function: g(x) = ..., h(x) = ...
        r'([a-zA-Z]\(x\)\s*=\s*[^\s,;.]+(?:\s*[+\-*/]\s*[^\s,;.]+)*)',
        # Parametric: x(t) = ... or y(t) = ...
        r'[xy]\(t\)\s*=\s*([^\s,;.]+(?:\s*[+\-*/]\s*[^\s,;.]+)*)',
        # Simple expressions: x^2, 2x+1, etc.
        r'([a-zA-Z]\s*\^\s*\d+|[\d]+\s*[a-zA-Z]\s*\^\s*\d+|[\d]+\s*[a-zA-Z]\s*[+\-]\s*[\d]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            extracted = match.group(1) if len(match.groups()) > 0 else match.group(0)
            # Clean up the extracted equation
            extracted = extracted.strip()
            # Remove trailing punctuation that might have been captured
            extracted = re.sub(r'[.,;]+$', '', extracted)
            if extracted:
                return extracted
    
    return None

