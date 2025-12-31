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

# Sidebar
with st.sidebar:
    st.title("📐 AP Calculus BC")
    st.title("AI Mastermind")
    
    st.divider()
    
    # Unit Focus Selection
    st.subheader("Unit Focus")
    unit_options = [
        "None (General)",
        "Integration Techniques",
        "Differential Equations",
        "Parametrics & Vectors",
        "Polar Curves",
        "Infinite Series"
    ]
    
    selected_unit = st.radio(
        "Select a unit to focus on:",
        unit_options,
        index=0,
        key="unit_selector"
    )
    
    # Update unit focus (remove "None (General)" prefix)
    if selected_unit == "None (General)":
        st.session_state.unit_focus = None
    else:
        st.session_state.unit_focus = selected_unit
    
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

