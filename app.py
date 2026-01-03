"""
AP Calculus BC AI Mastermind - Main Streamlit Application
"""

import streamlit as st
import streamlit.components.v1 as components
from modules.openai_handler import send_message, validate_api_key, analyze_image_problem
from modules.graph_engine import generate_calc_plot, extract_equation_from_text, detect_area_request
import re
import os
import mimetypes
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables (OpenAI + Desmos keys)
# Streamlit reruns this script, so keep it here (not only inside imported modules).
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")


def _is_graph_request(text: str) -> bool:
    """
    Heuristic: treat messages containing 'graph/plot' as requests to render a graph now.
    """
    t = (text or "").strip().lower()
    if not t:
        return False
    # common student phrasing: "graph y=...", "plot f(x)=...", "show me the graph of ..."
    return bool(re.search(r'\b(graph|plot|sketch|draw|visualize|visualise)\b', t))


def _extract_visualization_code(text: str) -> list:
    """
    Extract Python code blocks containing matplotlib/st.pyplot or plotly/st.plotly_chart from markdown text.
    
    Returns:
        List of code strings found in ```python blocks that contain visualization libraries
    """
    if not text:
        return []
    
    # Pattern to match ```python ... ``` blocks
    pattern = r'```python\s*\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    
    visualization_blocks = []
    for code in matches:
        code_lower = code.lower()
        # Check if it contains matplotlib, plotly, or st.pyplot/st.plotly_chart
        if any(keyword in code_lower for keyword in [
            'matplotlib', 'plotly', 'st.pyplot', 'st.plotly_chart', 
            'plt.', 'go.', 'px.', 'fig.add_trace'
        ]):
            visualization_blocks.append(code.strip())
    
    return visualization_blocks


def _execute_visualization_code(code: str) -> bool:
    """
    Safely execute visualization code block (Matplotlib or Plotly) and render the plot.
    
    Args:
        code: Python code string containing matplotlib/st.pyplot or plotly/st.plotly_chart calls
    
    Returns:
        True if execution succeeded, False otherwise
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import hashlib
        import time
        
        # Modify code to inject unique keys into st.plotly_chart calls
        # Pattern: st.plotly_chart(fig) -> st.plotly_chart(fig, key="unique_key")
        # Handle various formats: st.plotly_chart(fig), st.plotly_chart(fig, use_container_width=True), etc.
        modified_code = code
        if 'st.plotly_chart' in code.lower():
            # Counter for multiple plotly_chart calls in the same code block
            chart_counter = [0]
            
            # Pattern to match st.plotly_chart calls with or without existing parameters
            # This regex matches the function call and captures the arguments
            pattern = r'st\.plotly_chart\s*\(([^)]*)\)'
            
            def add_key(match):
                chart_counter[0] += 1
                # Generate unique key for this specific chart call
                unique_key = hashlib.md5((code + str(time.time()) + str(chart_counter[0])).encode()).hexdigest()[:12]
                
                args = match.group(1).strip()
                # Check if key already exists
                if 'key=' in args:
                    return match.group(0)  # Already has a key, don't modify
                
                # Check if there are existing kwargs (comma-separated)
                if args and (',' in args or '=' in args):
                    # Add key to existing kwargs
                    return f'st.plotly_chart({args}, key="{unique_key}")'
                elif args:
                    # Single argument case
                    return f'st.plotly_chart({args}, key="{unique_key}")'
                else:
                    # No arguments case (shouldn't happen, but handle it)
                    return f'st.plotly_chart(key="{unique_key}")'
            
            modified_code = re.sub(pattern, add_key, code)
        
        # Create a safe execution context
        # Note: This executes code, so be cautious in production
        exec_globals = {
            'plt': plt,
            'np': np,
            'st': st,
            'matplotlib': __import__('matplotlib'),
            'numpy': np,
            '__builtins__': __builtins__
        }
        
        # Try to import plotly if needed
        code_lower = code.lower()
        if 'plotly' in code_lower or 'st.plotly_chart' in code_lower:
            try:
                import plotly.graph_objects as go
                import plotly.express as px
                exec_globals['go'] = go
                exec_globals['px'] = px
                exec_globals['plotly'] = __import__('plotly')
            except ImportError:
                st.warning("Plotly is not installed. Install it with: pip install plotly")
                return False
        
        # Execute the modified code
        exec(modified_code, exec_globals)
        
        return True
    except Exception as e:
        st.error(f"Error executing visualization code: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return False


def _remove_visualization_code_blocks(text: str) -> str:
    """
    Remove Python code blocks containing visualization code from markdown text.
    This hides the code from display while still allowing execution.
    
    Args:
        text: Markdown text that may contain visualization code blocks
    
    Returns:
        Text with visualization code blocks removed
    """
    if not text:
        return text
    
    # Pattern to match ```python ... ``` blocks
    pattern = r'```python\s*\n(.*?)```'
    
    def should_remove(match):
        code = match.group(1)
        code_lower = code.lower()
        # Check if it contains visualization libraries
        return any(keyword in code_lower for keyword in [
            'matplotlib', 'plotly', 'st.pyplot', 'st.plotly_chart', 
            'plt.', 'go.', 'px.', 'fig.add_trace'
        ])
    
    # Replace visualization code blocks with empty string
    result = re.sub(pattern, lambda m: '' if should_remove(m) else m.group(0), text, flags=re.DOTALL)
    
    # Clean up multiple consecutive newlines
    result = re.sub(r'\n{3,}', '\n\n', result)
    
    return result.strip()


# Page configuration
st.set_page_config(
    page_title="AP Calculus BC AI Mastermind",
    page_icon="📐",
    layout="wide"
)

# Dark UI polish (Streamlit theme is set in `.streamlit/config.toml`, this just tightens a few surfaces)
st.markdown(
    """
    <style>
      /* Make the app feel consistently dark (some components keep light borders by default). */
      html, body { background: #0E1117 !important; }
      .stApp { background: #0E1117; }

      /* Top header (Streamlit toolbar area) */
      header[data-testid="stHeader"] { background: #0B0F14; }
      header[data-testid="stHeader"] * { color: #E6EDF3 !important; }

      /* Sidebar */
      section[data-testid="stSidebar"] { background: #0B0F14; }
      section[data-testid="stSidebar"] * { color: #E6EDF3; }

      /* Inputs (selectbox/text inputs) - BaseWeb components */
      div[data-baseweb="select"] > div {
        background-color: #0B0F14 !important;
        border-color: rgba(230, 237, 243, 0.18) !important;
        color: #E6EDF3 !important;
      }
      div[data-baseweb="select"] * { color: #E6EDF3 !important; }
      div[data-baseweb="popover"] { background: #0B0F14 !important; }

      /* Expanders / containers */
      div[data-testid="stExpander"] { border-color: rgba(230, 237, 243, 0.12); }
      div[data-testid="stExpander"] details { background: rgba(22, 27, 34, 0.6); border-radius: 10px; }

      /* Chat input bar */
      div[data-testid="stBottomBlockContainer"] { background: #0E1117 !important; }
      div[data-testid="stChatInput"] { background: #0E1117 !important; }
      div[data-testid="stChatInput"] > div { background: #0E1117 !important; }
      div[data-testid="stChatInput"] textarea {
        background: #0B0F14 !important;
        color: #E6EDF3 !important;
        border: 1px solid rgba(230, 237, 243, 0.18) !important;
      }

      /* Buttons */
      button[kind="secondary"], button[kind="primary"], button {
        border-radius: 10px !important;
      }
      button[kind="secondary"], button {
        background: #0B0F14 !important;
        border: 1px solid rgba(230, 237, 243, 0.18) !important;
        color: #E6EDF3 !important;
      }
      button[kind="primary"] {
        background: #7C5CFF !important;
        border-color: #7C5CFF !important;
        color: #0E1117 !important;
      }

      /* File uploader surface */
      section[data-testid="stFileUploaderDropzone"] {
        background: rgba(22, 27, 34, 0.6) !important;
        border-color: rgba(230, 237, 243, 0.18) !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "unit_focus" not in st.session_state:
    st.session_state.unit_focus = None
if "last_selected_unit" not in st.session_state:
    st.session_state.last_selected_unit = "None (General)"
if "generate_followup" not in st.session_state:
    st.session_state.generate_followup = False
if "show_desmos" not in st.session_state:
    st.session_state.show_desmos = False

# Unit → subunit structure (AP Calculus BC)
# These are used only for UI focus; the tutor can still answer anything.
UNIT_STRUCTURE = {
    "Unit 1: Limits and Continuity": [
        "1.1 Introducing Calculus: Can Change Occur at an Instant?",
        "1.2 Defining Limits and Using Limit Notation",
        "1.3 Estimating Limit Values from Graphs",
        "1.4 Estimating Limit Values from Tables",
        "1.5 Determining Limits Using Algebraic Properties of Limits",
        "1.6 Determining Limits Using Algebraic Manipulation",
        "1.7 Selecting Procedures for Determining Limits",
        "1.8 Determining Limits Using the Squeeze Theorem",
        "1.9 Connecting Limits and Continuity",
        "1.10 Defining Types of Discontinuities",
    ],
    "Unit 2: Differentiation: Definition and Fundamental Properties": [
        "2.1 Defining the Derivative",
        "2.2 The Derivative as a Limit",
        "2.3 Differentiability and Continuity",
        "2.4 Applying the Derivative to Motion",
        "2.5 Differentiation Rules: Constant, Power, Sum, Difference",
        "2.6 Product Rule",
        "2.7 Quotient Rule",
        "2.8 Chain Rule",
        "2.9 Implicit Differentiation",
        "2.10 Related Rates",
    ],
    "Unit 3: Differentiation: Composite, Implicit, and Inverse Functions": [
        "3.1 Derivatives of Trigonometric Functions",
        "3.2 Derivatives of Inverse Trigonometric Functions",
        "3.3 Derivatives of Exponential and Logarithmic Functions",
        "3.4 Differentiating Compositions of Functions",
        "3.5 Differentiating Implicit Relations",
        "3.6 Derivatives of Inverse Functions",
        "3.7 Second Derivatives and Concavity",
    ],
    "Unit 4: Contextual Applications of Differentiation": [
        "4.1 Interpreting the Meaning of the Derivative in Context",
        "4.2 Straight-Line Motion: Velocity and Acceleration",
        "4.3 Rates of Change in Applied Contexts",
        "4.4 Particle Motion: Speed vs Velocity",
        "4.5 Motion with Position Given by a Graph/Function",
    ],
    "Unit 5: Analytical Applications of Differentiation": [
        "5.1 Mean Value Theorem",
        "5.2 Extreme Value Theorem and Critical Points",
        "5.3 Increasing/Decreasing & First Derivative Test",
        "5.4 Concavity & Second Derivative Test",
        "5.5 Curve Sketching and Analysis",
        "5.6 Optimization",
        "5.7 L’Hôpital’s Rule (BC)",
    ],
    "Unit 6: Integration and Accumulation of Change": [
        "6.1 Riemann Sums and Approximating Areas",
        "6.2 Definite Integrals and Accumulation Functions",
        "6.3 Fundamental Theorem of Calculus (Part 1)",
        "6.4 Fundamental Theorem of Calculus (Part 2)",
        "6.5 Properties of Integrals",
        "6.6 Substitution (u-substitution)",
    ],
    "Unit 7: Differential Equations": [
        "7.1 Slope Fields",
        "7.2 Euler’s Method",
        "7.3 Separation of Variables",
        "7.4 Exponential Growth and Decay",
        "7.5 Logistic Differential Equations (BC)",
    ],
    "Unit 8: Applications of Integration": [
        "8.1 Average Value of a Function",
        "8.2 Area Between Curves",
        "8.3 Volumes with Cross Sections",
        "8.4 Volume: Disk and Washer Methods",
        "8.5 Volume: Shell Method (BC)",
        "8.6 Arc Length (BC)",
    ],
    "Unit 9: Parametric Equations, Polar Coordinates, and Vector-Valued Functions": [
        "9.1 Parametric Equations and Derivatives",
        "9.2 Parametric Speed and Arc Length (BC)",
        "9.3 Polar Coordinates and Graphing Polar Curves",
        "9.4 Area in Polar Coordinates (BC)",
        "9.5 Slope and Tangent Lines in Polar (BC)",
        "9.6 Vector-Valued Functions: Position, Velocity, Acceleration (BC)",
    ],
    "Unit 10: Infinite Sequences and Series (BC Only)": [
        "10.1 Defining Convergent and Divergent Infinite Series",
        "10.2 Working with Geometric Series",
        "10.3 The nth Term Test for Divergence",
        "10.4 Integral Test for Convergence",
        "10.5 Harmonic Series and p-Series",
        "10.6 Comparison Tests for Convergence",
        "10.7 Alternating Series Test for Convergence",
        "10.8 Ratio Test for Convergence",
        "10.9 Determining Absolute or Conditional Convergence",
        "10.10 Alternating Series Error Bound",
        "10.11 Finding Taylor Polynomial Approximations of Functions",
        "10.12 Lagrange Error Bound",
        "10.13 Radius and Interval of Convergence of Power Series",
        "10.14 Finding Taylor or Maclaurin Series for a Function",
        "10.15 Representing Functions as Power Series",
    ],
}

# Sidebar
with st.sidebar:
    st.title("📐 AP Calculus BC")
    st.title("AI Mastermind")
    
    st.divider()
    
    # Unit Focus Selection (Unit → Subunit)
    st.subheader("Unit Focus")
    unit_options = ["None (General)"] + list(UNIT_STRUCTURE.keys())

    selected_unit = st.selectbox(
        "Select a unit to focus on:",
        unit_options,
        index=0,
        key="unit_selector",
    )

    # If the main unit changed, reset the subunit selector to avoid invalid stale values
    if selected_unit != st.session_state.last_selected_unit:
        st.session_state.last_selected_unit = selected_unit
        if "subunit_selector" in st.session_state:
            del st.session_state["subunit_selector"]

    selected_subunit = None
    if selected_unit != "None (General)":
        selected_subunit = st.selectbox(
            "Select a subunit/topic:",
            UNIT_STRUCTURE[selected_unit],
            key="subunit_selector",
        )

    # Store a single focus string for prompts
    if selected_unit == "None (General)":
        st.session_state.unit_focus = None
    else:
        st.session_state.unit_focus = (
            f"{selected_unit} → {selected_subunit}" if selected_subunit else selected_unit
        )
    
    st.divider()
    
    # Clear Chat Button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.generate_followup = False
        st.rerun()
    
    st.divider()
    
    # API Status
    st.subheader("API Status")
    if validate_api_key():
        st.success("✅ API Key Valid")
    else:
        st.error("❌ API Key Invalid")
        st.info("Please check your .env file")

    # Desmos Status (graphing)
    desmos_key = (os.getenv("DESMOS_API_KEY") or "").strip()
    if desmos_key:
        st.success("✅ Desmos Key Set")
    else:
        st.warning("⚠️ Desmos Key Missing")
        st.caption("Set `DESMOS_API_KEY` in your `.env` (graphs won't load without it).")
    
    st.divider()
    
    # Desmos Sandbox Toggle
    if st.button("📊 Desmos Sandbox", use_container_width=True, type="primary" if st.session_state.show_desmos else "secondary"):
        st.session_state.show_desmos = not st.session_state.show_desmos
        st.rerun()

# Main content area - split screen if Desmos is enabled
if st.session_state.show_desmos:
    # Split screen: Chat on left, Desmos on right
    chat_col, desmos_col = st.columns([1, 1])
    
    with chat_col:
        st.title("AP Calculus BC AI Mastermind")
        st.markdown("Your intelligent calculus tutor powered by GPT-5 mini")
        
        # All chat content goes here
        chat_content_container = st.container()
    
    with desmos_col:
        st.subheader("📊 Desmos Sandbox")
        # Embed Desmos calculator iframe
        desmos_html = """
        <iframe src="https://www.desmos.com/calculator" 
                width="100%" 
                height="800px" 
                frameborder="0" 
                style="border: 1px solid #ccc; border-radius: 5px;">
        </iframe>
        """
        components.html(desmos_html, height=800)
else:
    # Full width chat
    st.title("AP Calculus BC AI Mastermind")
    st.markdown("Your intelligent calculus tutor powered by GPT-5 mini")
    chat_content_container = st.container()

# All main content wrapped in container (for split screen support)
with chat_content_container:
    # Image upload: OCR → LaTeX → Solve
    with st.expander("🖼️ Upload an image (OCR → LaTeX → Solve)", expanded=False):
        st.caption(
            "Upload a screenshot/photo of a math or word problem. The AI will read it, convert it to clean LaTeX, and solve it."
        )

        uploaded = st.file_uploader(
            "Upload a problem image (PNG/JPG/WebP)",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=False,
        )

        if uploaded is not None:
            st.image(uploaded, caption="Uploaded image", use_container_width=True)

            col_a, col_b = st.columns(2)
            do_extract = col_a.button("🧾 Extract → LaTeX", use_container_width=True)
            do_solve = col_b.button("🧠 Extract + Solve", use_container_width=True, type="primary")

            if do_extract or do_solve:
                with st.spinner("Reading the image..."):
                    try:
                        image_bytes = uploaded.getvalue()
                        mime_type = (uploaded.type or "").strip() or mimetypes.guess_type(uploaded.name)[0] or "image/png"

                        result = analyze_image_problem(
                            image_bytes=image_bytes,
                            mime_type=mime_type,
                            unit_focus=st.session_state.unit_focus,
                            solve=bool(do_solve),
                        )

                        extracted_text = result.get("extracted_text", "")
                        problem_latex = result.get("problem_latex", "")
                        clean_latex = result.get("clean_latex_code", "")
                        solution_md = result.get("solution_markdown", "")

                        # Render a nicely formatted assistant message and also store it in chat history.
                        assistant_md_parts = []
                        if extracted_text:
                            assistant_md_parts.append("### Extracted text")
                            assistant_md_parts.append(extracted_text)

                        if problem_latex:
                            assistant_md_parts.append("### Problem (LaTeX)")
                            assistant_md_parts.append(f"$$\n{problem_latex}\n$$")

                        if solution_md:
                            assistant_md_parts.append("### Solution")
                            assistant_md_parts.append(solution_md)

                        if clean_latex:
                            assistant_md_parts.append("### Copy‑ready LaTeX")
                            assistant_md_parts.append(f"```latex\n{clean_latex}\n```")

                        assistant_md = "\n\n".join(assistant_md_parts).strip() or "I couldn't extract readable text from that image. Try a higher-resolution screenshot."

                        st.session_state.messages.append({"role": "assistant", "content": assistant_md})
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ Image processing error: {str(e)}")

    # Removed automatic followup practice problem generation
    # Users can explicitly ask for practice problems if they want them

    # Display chat history
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            content = message.get("content", "")
            # Remove visualization code blocks from display (hide the code)
            display_content = _remove_visualization_code_blocks(content)
            st.markdown(display_content)
            
            # Check for visualization code blocks (Matplotlib/Plotly) in assistant messages and execute them
            if message["role"] == "assistant" and content:
                visualization_blocks = _extract_visualization_code(content)
                for code_block in visualization_blocks:
                    with st.expander("📊 View Generated Plot", expanded=True):
                        _execute_visualization_code(code_block)
            
            # Note: Desmos graph generation via graph_equation is disabled
            # Graphs are now generated via Plotly/Matplotlib code blocks in the AI response

    # Chat input
    if prompt := st.chat_input("Ask me anything about AP Calculus BC..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Prepare messages for API - FILTER OUT EMPTY MESSAGES
                    api_messages = [
                        {"role": msg["role"], "content": msg["content"]}
                        for msg in st.session_state.messages
                        if "role" in msg 
                        and "content" in msg 
                        and msg.get("content", "").strip()  # Only include non-empty messages
                    ]
                    
                    # Validate we have messages to send
                    if not api_messages:
                        raise ValueError("No valid messages to send to API")
                    
                    # Get AI response
                    response = send_message(
                        api_messages,
                        unit_focus=st.session_state.unit_focus
                    )
                    
                    # Validate response is not empty
                    if not response or not response.strip():
                        raise ValueError("Empty response from API")
                    
                    # Remove visualization code blocks from display (hide the code)
                    display_response = _remove_visualization_code_blocks(response)
                    st.markdown(display_response)
                    
                    # Check for visualization code blocks (Matplotlib/Plotly) and execute them automatically
                    visualization_blocks = _extract_visualization_code(response)
                    for code_block in visualization_blocks:
                        with st.expander("📊 View Generated Plot", expanded=True):
                            _execute_visualization_code(code_block)
                    
                    # Add assistant response to chat history
                    message_data = {
                        "role": "assistant",
                        "content": response
                    }
                    
                    st.session_state.messages.append(message_data)
                    
                    # Removed automatic followup practice problem generation
                    # Users can explicitly ask for practice problems if they want them
                    st.rerun()
                    
                except Exception as e:
                    error_message = f"❌ Error: {str(e)}"
                    st.error(error_message)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_message
                    })
                    # Don't generate followup on error

    # Footer
    st.divider()
    st.caption("Powered by OpenAI GPT-5 mini | Built with Streamlit")

