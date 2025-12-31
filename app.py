"""
AP Calculus BC AI Mastermind - Main Streamlit Application
"""

import streamlit as st
import streamlit.components.v1 as components
from modules.openai_handler import send_message, validate_api_key
from modules.graph_engine import generate_calc_plot, extract_equation_from_text
import re
import os
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


# Page configuration
st.set_page_config(
    page_title="AP Calculus BC AI Mastermind",
    page_icon="📐",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "unit_focus" not in st.session_state:
    st.session_state.unit_focus = None
if "last_selected_unit" not in st.session_state:
    st.session_state.last_selected_unit = "None (General)"

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

# Main content area
st.title("AP Calculus BC AI Mastermind")
st.markdown("Your intelligent calculus tutor powered by GPT-4o-mini")

# Display chat history
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display graph if equation is stored
        if "graph_equation" in message:
            try:
                html_content = generate_calc_plot(message["graph_equation"])
                components.html(html_content, height=550)
            except Exception as e:
                st.error(f"Error displaying graph: {str(e)}")
        
        # Show graph button if equation detected but not yet graphed
        if "equation_detected" in message and "graph_equation" not in message:
            eq = message["equation_detected"]
            button_key = f"graph_btn_{idx}"
            # Show the equation and button clearly
            st.write(f"**Function detected:** `{eq}`")
            graph_button = st.button(f"📊 Graph This Function", key=button_key, use_container_width=True, type="primary")
            
            if graph_button:
                try:
                    # Update the message in session state to include the graph
                    st.session_state.messages[idx]["graph_equation"] = eq
                    st.rerun()
                except Exception as e:
                    st.error(f"Error generating graph: {str(e)}")

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
                # Prepare messages for API (exclude graph data)
                api_messages = [
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in st.session_state.messages
                    if "role" in msg and "content" in msg
                ]
                
                # Get AI response
                response = send_message(
                    api_messages,
                    unit_focus=st.session_state.unit_focus
                )
                
                # Display response
                st.markdown(response)
                
                # Check for equations in the conversation that could be graphed
                equation_detected = None
                
                # Try to extract equation from user's prompt or AI's response
                equation_detected = extract_equation_from_text(prompt)
                if not equation_detected:
                    equation_detected = extract_equation_from_text(response)

                wants_graph_now = _is_graph_request(prompt)
                
                # Add assistant response to chat history
                message_data = {
                    "role": "assistant",
                    "content": response
                }
                
                # If user explicitly asked to graph/plot, render immediately.
                # Otherwise, store the detected equation and offer a button right away.
                if equation_detected and wants_graph_now:
                    message_data["graph_equation"] = equation_detected
                elif equation_detected:
                    message_data["equation_detected"] = equation_detected
                
                st.session_state.messages.append(message_data)
                msg_idx = len(st.session_state.messages) - 1

                # Render graph or button immediately in this same assistant message
                if equation_detected and wants_graph_now:
                    try:
                        html_content = generate_calc_plot(equation_detected)
                        components.html(html_content, height=550)
                    except Exception as e:
                        st.error(f"Error generating graph: {str(e)}")
                elif equation_detected:
                    st.write(f"**Function detected:** `{equation_detected}`")
                    graph_button = st.button(
                        "📊 Graph This Function",
                        key=f"graph_btn_live_{msg_idx}",
                        use_container_width=True,
                        type="primary",
                    )
                    if graph_button:
                        st.session_state.messages[msg_idx]["graph_equation"] = equation_detected
                        st.rerun()
                
            except Exception as e:
                error_message = f"❌ Error: {str(e)}"
                st.error(error_message)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message
                })

# Footer
st.divider()
st.caption("Powered by OpenAI GPT-4o-mini | Built with Streamlit")

