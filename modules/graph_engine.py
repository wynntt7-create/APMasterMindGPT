"""
Graphing engine for AP Calculus BC functions using SymPy and Matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from typing import Tuple, Optional
import re


def generate_calc_plot(
    equation_string: str,
    x_range: Tuple[float, float] = (-10, 10),
    title: Optional[str] = None
) -> plt.Figure:
    """
    Generate a matplotlib plot for a given mathematical function.
    
    Args:
        equation_string: String representation of the function (e.g., "x^2", "sin(x)", "x^2 + 2*x + 1")
        x_range: Tuple of (min, max) x-values for the plot
        title: Optional title for the plot
    
    Returns:
        matplotlib Figure object ready for Streamlit display
    
    Raises:
        ValueError: If equation cannot be parsed
    """
    try:
        # Clean up the equation string
        equation_string = equation_string.strip()
        
        # Replace common LaTeX/math notation with Python-friendly syntax
        # Handle cases like f(x) = x^2 or just x^2
        equation_string = re.sub(r'f\(x\)\s*=\s*', '', equation_string)
        equation_string = re.sub(r'y\s*=\s*', '', equation_string)
        
        # Parse the equation using SymPy
        x = sp.Symbol('x')
        try:
            expr = sp.sympify(equation_string)
        except:
            # Try adding x if it's missing (e.g., "2" becomes "2*x^0")
            expr = sp.sympify(equation_string)
        
        # Convert to a callable function
        f = sp.lambdify(x, expr, modules=['numpy'])
        
        # Generate x values
        x_vals = np.linspace(x_range[0], x_range[1], 1000)
        
        # Evaluate y values
        try:
            y_vals = f(x_vals)
            # Handle cases where result might be a scalar
            if np.isscalar(y_vals):
                y_vals = np.full_like(x_vals, y_vals)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Cannot evaluate function: {str(e)}")
        
        # Handle complex numbers or invalid values
        y_vals = np.real(y_vals)
        y_vals = np.nan_to_num(y_vals, nan=0.0, posinf=10, neginf=-10)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x_vals, y_vals, 'b-', linewidth=2, label=f'$f(x) = {sp.latex(expr)}$')
        ax.axhline(y=0, color='k', linewidth=0.5, linestyle='--')
        ax.axvline(x=0, color='k', linewidth=0.5, linestyle='--')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        else:
            ax.set_title(f'Graph of $f(x) = {sp.latex(expr)}$', fontsize=14, fontweight='bold')
        
        ax.legend(loc='best')
        ax.set_xlim(x_range)
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        # Return an error plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f'Error plotting function:\n{str(e)}', 
                ha='center', va='center', fontsize=12, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        return fig


def extract_equation_from_text(text: str) -> Optional[str]:
    """
    Try to extract a mathematical equation from text.
    
    Args:
        text: Text that may contain an equation
    
    Returns:
        Extracted equation string or None if not found
    """
    # Common patterns for equations
    patterns = [
        r'f\(x\)\s*=\s*([^\s]+(?:\s+[+\-*/]\s+[^\s]+)*)',
        r'y\s*=\s*([^\s]+(?:\s+[+\-*/]\s+[^\s]+)*)',
        r'([a-zA-Z]\(x\)\s*=\s*[^\s]+(?:\s+[+\-*/]\s+[^\s]+)*)',
        r'(\w+\^?\d+\s*[+\-*/]?\s*\w*\^?\d*)',  # Simple polynomial patterns
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1) if len(match.groups()) > 0 else match.group(0)
    
    return None

