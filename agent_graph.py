import os
import pandas as pd
from typing import Annotated, Tuple, Any
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.utilities import PythonREPL
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph_supervisor import create_supervisor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import streamlit as st

# Initialize Models and Search (can be initialized lazily or globally)
llm_model = ChatOpenAI(model="gpt-4o")
tv_search = TavilySearchResults(max_results=5, search_depth='advanced', max_tokens=10000)

# --- ML Helper Functions ---
# We use st.cache_resource if running in streamlit to avoid retraining on every call
# If not in streamlit, this is just a standard function, but we might want a global cache variable.
_ml_cache = {"model": None, "data": None}

def get_trained_model() -> Tuple[Any, pd.DataFrame]:
    """
    Returns cached model and data, or trains/loads if not present.
    In a real app, you might save the model to disk (pickle/joblib) and load it.
    """
    if _ml_cache["model"] is not None and _ml_cache["data"] is not None:
        return _ml_cache["model"], _ml_cache["data"]

    # Try to find the file
    file_path = 'titanic.csv'
    if not os.path.exists(file_path):
        # Fallback for Streamlit upload if we implement that, or just raise
        raise FileNotFoundError(f"'{file_path}' not found.")

    try:
        titanic_data = pd.read_csv(file_path)
    except Exception as e:
        raise e

    # Data preprocessing
    if 'Age' in titanic_data.columns:
        titanic_data['Age'] = titanic_data['Age'].fillna(titanic_data['Age'].median())
    if 'Embarked' in titanic_data.columns:
        titanic_data['Embarked'] = titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0])
    
    cols_to_drop = ['Cabin', 'Ticket', 'PassengerId']
    titanic_data = titanic_data.drop([c for c in cols_to_drop if c in titanic_data.columns], axis=1)
    
    titanic_data = pd.get_dummies(titanic_data, columns=['Sex', 'Embarked'], drop_first=True)

    # Feature selection
    if 'Survived' in titanic_data.columns and 'Name' in titanic_data.columns:
        X = titanic_data.drop(['Survived', 'Name'], axis=1)
        y = titanic_data['Survived']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        # Cache results
        _ml_cache["model"] = model
        _ml_cache["data"] = titanic_data
        return model, titanic_data
    else:
        raise ValueError("Dataset missing required columns 'Survived' or 'Name'.")

# --- Tools ---

@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

@tool
def subtract(a: float, b: float) -> float:
    """Subtract two numbers."""
    return a - b

@tool
def python_repl(code: Annotated[str, "The python code to execute to generate your chart."]):
    """Use this to execute python code.
       If you want to see the output of a value,
       you should print it out with `print(...)`.
       This is visible to the user."""
    repl = PythonREPL()
    try:
        print("generated code by researcher agent \n", code)
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return (result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER.")

@tool
def search_web(query: str) -> list:
    """Search the web for a query."""
    return tv_search.invoke(query)

@tool
def predict_survival(name: Annotated[str, "name of the passenger for prediction of survival"]):
    """Predicts the survival of a passenger in titanic given their name."""
    try:
        model, titanic_data = get_trained_model()
        
        # Find the passenger's data by name
        passenger = titanic_data[titanic_data['Name'].str.contains(name, case=False, na=False)]

        if passenger.empty:
            return "Passenger not found in the dataset."

        # Extract relevant features for prediction (excluding 'Name', 'Survived')
        # We need to ensure the columns match the training features (X)
        # The model expects specific features.
        # Since we processed `titanic_data` to have dummies, we need to be careful.
        # However, `titanic_data` in `_ml_cache` is the PROCESSED data (with dummies) INCLUDING Name and Survived?
        # Wait, in `train_ml_model`, `titanic_data` is reassigned: `titanic_data = pd.get_dummies(...)`
        # So it DOES have the dummy columns. 
        # But we dropped 'Survived' and 'Name' to create X. 
        # So we should drop them here too.
        
        feature_cols = [c for c in titanic_data.columns if c not in ['Name', 'Survived']]
        passenger_features = passenger[feature_cols]

        # Make prediction using the model
        prediction = model.predict(passenger_features)
        return int(prediction[0])  # Return the prediction (1 or 0)
    except FileNotFoundError:
        return "Model not trained (titanic.csv not found)."
    except KeyError as e:
        return f"Error: Missing feature: {e}"
    except Exception as e:
        return f"An error occurred: {e}"

# --- Agent Definitions ---

def create_workflow():
    """Create and return the compiled supervisor workflow."""
    
    # helper to check if we can actually use the ML agent
    tools_ml = [predict_survival]
    
    websearch_agent = create_agent(
        model=llm_model,
        tools=[search_web],
        name="tavily_search",
        system_prompt="You are a world class researcher with access to web search. Do not do any math or predictions."
    )

    math_agent = create_agent(
        model=llm_model,
        tools=[add, multiply, subtract],
        name="math_expert",
        system_prompt="You are a math expert. Always use one tool at a time."
    )

    coding_agent = create_agent(
        model=llm_model,
        tools=[python_repl],
        name="python_coder",
        system_prompt="You are a python coding expert. Always use one tool at a time."
    )

    ml_agent = create_agent(
        model=llm_model,
        tools=tools_ml,
        name="ml_expert",
        system_prompt="You are a machine learning expert in predicting the survival chances of a passenger in titanic. Always use one tool at a time"
    )

    # Supervisor
    workflow = create_supervisor(
        [math_agent, coding_agent, ml_agent, websearch_agent],
        model=llm_model,
        prompt=(
            "You are a team supervisor managing a research expert, coding agent, ml expert and a math expert. "
            "For any predictions, use ml expert. Always use ml_expert for predictions. "
            "For current events, use tavily_search. "
            "For math problems, use math_expert. "
            "For any coding request using python, use python_coder. "
            "If data is passed from the supervisor use only that, if not generate the code and give the option for user to execute it manually."
        )
    )
    
    return workflow.compile()
