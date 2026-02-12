try:
    import streamlit
    print("streamlit: OK")
except ImportError as e:
    print(f"streamlit: {e}")

try:
    import langchain
    print("langchain: OK")
except ImportError as e:
    print(f"langchain: {e}")

try:
    import langgraph_supervisor
    print("langgraph_supervisor: OK")
except ImportError as e:
    print(f"langgraph_supervisor: {e}")

try:
    from agent_graph import create_workflow
    print("agent_graph: OK")
except ImportError as e:
    print(f"agent_graph: {e}")
except Exception as e:
    print(f"agent_graph error: {e}")
