"""
AP Calculus BC AI Mastermind - Main Streamlit Application
"""

import streamlit as st
from modules.openai_handler import send_message, validate_api_key
from modules.graph_engine import generate_calc_plot, extract_equation_from_text
import re

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
                fig = generate_calc_plot(message["graph_equation"])
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error displaying graph: {str(e)}")
        
        # Show graph button if equation detected but not yet graphed
        if "equation_detected" in message and "graph_equation" not in message:
            eq = message["equation_detected"]
            button_key = f"graph_btn_{idx}"
            if st.button(f"📊 Graph: {eq}", key=button_key):
                try:
                    fig = generate_calc_plot(eq)
                    st.pyplot(fig)
                    message["graph_equation"] = eq
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
                
                # Add assistant response to chat history
                message_data = {
                    "role": "assistant",
                    "content": response
                }
                
                # Store detected equation if found (AI will prompt user via system prompt)
                if equation_detected:
                    message_data["equation_detected"] = equation_detected
                    st.info(f"💡 Function detected: `{equation_detected}` - Click the graph button below if you'd like to visualize it!")
                
                st.session_state.messages.append(message_data)
                
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

