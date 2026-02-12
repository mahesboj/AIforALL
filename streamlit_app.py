import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Azure Key Vault Integration (restored from original code)
try:
    from azure.identity import DefaultAzureCredential
    from azure.keyvault.secrets import SecretClient
    
    key_vault_name = "aiforall-dev-keyvault"
    key_vault_uri = f"https://{key_vault_name}.vault.azure.net"
    
    # Use managed identity / dev creds
    credential = DefaultAzureCredential()
    client = SecretClient(vault_url=key_vault_uri, credential=credential)
    
    # Set environment variables from Key Vault
    # Note: This might overwrite .env values if they exist, which matches "keep it the same way"
    os.environ['OPENAI_API_KEY'] = client.get_secret('openai-api-key').value
    os.environ['TAVILY_API_KEY'] = client.get_secret('tavily-api-key').value
    print("Successfully loaded secrets from Azure Key Vault.")
    
except Exception as e:
    print(f"Key Vault loading skipped or failed: {e}")
    print("Relying on .env or existing environment variables.")

from agent_graph import create_workflow, get_trained_model

st.set_page_config(page_title="Multi-Agent Supervisor", page_icon="🤖")

st.title("🤖 Multi-Agent Supervisor Chat")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    st.markdown("""
    This app uses a multi-agent supervisor system to handle:
    - 📊 **ML Predictions** (Titanic)
    - 🧮 **Math**
    - 🐍 **Python Coding**
    - 🌐 **Web Search**
    """)
    
    # Check for API Keys
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY not found in environment!")
        st.info("Please set it in your .env file or Azure settings.")
    
    if not os.getenv("TAVILY_API_KEY"):
        st.warning("TAVILY_API_KEY not found. Search might fail.")

    # Check for Titanic Data
    if not os.path.exists("titanic.csv"):
        st.warning("`titanic.csv` not found. ML Agent will be disabled.")
        uploaded_file = st.file_uploader("Upload titanic.csv", type="csv")
        if uploaded_file:
            with open("titanic.csv", "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success("File uploaded! Please refresh.")

# Initialize Graph
@st.cache_resource
def get_graph():
    return create_workflow()

try:
    app = get_graph()
except Exception as e:
    st.error(f"Failed to initialize agent graph: {e}")
    st.stop()

# specific message handling to format LangChain messages
def format_message(message):
    role = "user"
    if hasattr(message, "type"): 
        # map langchain types to streamlit roles
        if message.type == "human": role = "user"
        elif message.type == "ai": role = "assistant"
        elif message.type == "tool": role = "tool" # normally we might hide these or show in expander
        
    return role, message.content

def display_agent_thoughts(messages):
    """
    Parses conversation history to display key events:
    - Tool Calls (from AI messages)
    - Tool Outputs (from Tool messages)
    - Agent "Thought" content (if any)
    """
    for msg in messages:
        # Check for Tool Calls (AIMessage with tool_calls)
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tool_call in msg.tool_calls:
                name = tool_call.get("name", "Unknown Tool")
                args = tool_call.get("args", {})
                st.info(f"🛠️ **Agent Decision:** calling `{name}` with args: `{args}`")
        
        # Check for Tool Outputs (ToolMessage)
        elif hasattr(msg, "type") and msg.type == "tool":
            tool_name = getattr(msg, "name", "Tool")
            content = msg.content
            # Truncate if too long (optional, but good for UI)
            preview = content[:500] + "..." if len(content) > 500 else content
            with st.expander(f"✅ Comparison/Result from `{tool_name}`"):
                st.code(preview)
        
        # Check for standard Agent thought/text (AIMessage without tool_calls)
        elif hasattr(msg, "type") and msg.type == "ai" and msg.content:
            # If it's the final response, we might not want to show it here as "reasoning", 
            # but sometimes agents "think" out loud. Use a lighter styling.
            # We skip the *final* response which is usually handled outside this loop, 
            # but intermediate thoughts are good.
            # Heuristic: if it has no tool calls, it's a thought or final answer.
            pass

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    if message["role"] != "tool": # Hide tool outputs from main chat implementation for cleanliness, or show if preferred
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Invoke the graph
        # We need to construct the input payload
        # The graph expects {"messages": ...}
        
        input_messages = [{"role": "user", "content": prompt}]
        
        # We can stream intermediate steps if supported, but for now let's just invoke
        with st.spinner("Agents are thinking..."):
            try:
                # Run the graph
                final_state = app.invoke({"messages": input_messages})
                
                # The output is a state dict with "messages". 
                # We usually want the LAST message.
                conversation_history = final_state.get("messages", [])
                
                # Filter interactions to show "thought process"
                # We can show intermediate agent steps in an expander
                
                with st.expander("Show Agent Reasoning"):
                    display_agent_thoughts(conversation_history)

                if conversation_history:
                    last_msg = conversation_history[-1]
                    full_response = last_msg.content
                    message_placeholder.markdown(full_response)
                else:
                    message_placeholder.markdown("No response generated.")
                    
            except Exception as e:
                st.error(f"Error during execution: {e}")
                full_response = f"Error: {e}"

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
