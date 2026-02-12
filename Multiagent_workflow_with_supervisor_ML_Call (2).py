#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install langchain langchain_community langchain_experimental langgraph-supervisor langchain-openai')


# In[3]:


pip install pandas scikit-learn


# In[4]:


from langchain_openai import ChatOpenAI
from langgraph_supervisor import create_supervisor
#from langgraph.prebuilt import create_react_agent
from langchain.agents import create_agent
from typing import Annotated
from langchain_experimental.utilities import PythonREPL
from langchain_community.tools.tavily_search import TavilySearchResults
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[5]:


# prompt: suppress warnings

import warnings
warnings.filterwarnings('ignore')


# In[10]:


pip install azure-identity azure-keyvault-secrets


# In[12]:


from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient


# In[17]:


from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

# Key Vault details
key_vault_name = "aiforall-dev-keyvault"          # e.g. "my-dev-kv"

# Build vault URL
key_vault_uri = f"https://{key_vault_name}.vault.azure.net"

# Use managed identity / dev creds (VS Code, Azure CLI, etc.)
credential = DefaultAzureCredential()

# Create client
client = SecretClient(vault_url=key_vault_uri, credential=credential)


# In[21]:


import os
os.environ['OPENAI_API_KEY']=client.get_secret('openai-api-key').value
os.environ['TAVILY_API_KEY']=client.get_secret('tavily-api-key').value
tv_search = TavilySearchResults(max_results=5, search_depth='advanced',
                                max_tokens=10000)


# In[22]:


def train_ml_model():
# Load the Titanic dataset (replace 'titanic.csv' with the actual file path)
  try:
      titanic_data = pd.read_csv(r'titanic.csv')
  except FileNotFoundError:
      print("Error: 'titanic.csv' not found. Please upload the file to the current directory or provide the correct path.")
      # You might want to handle this error more gracefully, e.g., by exiting the script or prompting the user for the file path.
      exit()


  # Data preprocessing (basic example)
  titanic_data['Age'].fillna({'Age':titanic_data['Age'].median()}, inplace=True)
  titanic_data['Embarked'].fillna({'Embarked':titanic_data['Embarked'].mode()[0]}, inplace=True)
  titanic_data = titanic_data.drop(['Cabin', 'Ticket', 'PassengerId'], axis=1)
  titanic_data = pd.get_dummies(titanic_data, columns=['Sex', 'Embarked'], drop_first=True)


  # Feature selection and target variable
  X = titanic_data.drop(['Survived','Name'], axis=1)
  y = titanic_data['Survived']

  # Split data into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Initialize and train a RandomForestClassifier
  model = RandomForestClassifier(random_state=42)
  model.fit(X_train, y_train)

  # Make predictions on the test set
  y_pred = model.predict(X_test)

  # Evaluate the model
  accuracy = accuracy_score(y_test, y_pred)
  return model,titanic_data


# In[23]:


model = ChatOpenAI(model="gpt-4o")

def get_weather(city: str) -> str:  # (1)!
    """Get weather for a given city."""
    print('inside weather tool')
    return f"It's always rainy in {city}!"

# agent = create_agent(
#     model,
#     tools=[get_weather],
#     name='weather tool',# (3)!
#     system_prompt="You are a helpful assistant.Do not change the messages from tools. Use as it is"  # (4)!
# )

# # Run the agent
# whether_result=agent.invoke(
#     {"messages": [{"role": "user", "content": "what is the weather in bengaluru"}]}
# )


# In[24]:


from langchain.tools import tool
# Create specialized agents
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
    """Multiply two numbers."""
    return a - b

@tool
def python_repl(code: Annotated[str, "The python code to execute to generate your chart."],):
    """Use this to execute python code.
       If you want to see the output of a value,
       you should print it out with `print(...)`.

       This is visible to the user."""
    repl = PythonREPL()
    try:
        print("generated code by researcher agent \n",code)
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return (result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER.")
@tool
def search_web(query: str) -> list:
    """Search the web for a query."""
    results = tv_search.invoke(query)
    return results


def predict_survival(name: Annotated[str,"name of the passenger for prediction of survival"]):
    """Predicts the survival of a passenger in titanic given their name.

    Args:
        name: The name of the passenger.
    Returns:
        The predicted survival (1 for survived, 0 for not survived) or an error message.
    """
    try:
        model,titanic_data=train_ml_model()
        # Find the passenger's data by name
        passenger = titanic_data[titanic_data['Name'].str.contains(name, case=False, na=False)]

        if passenger.empty:
            return "Passenger not found in the dataset."

        # Extract relevant features for prediction (excluding 'Name', 'Survived')
        passenger_features = passenger.drop(['Name', 'Survived'], axis=1)

        # Make prediction using the model
        prediction = model.predict(passenger_features)
        return prediction[0]  # Return the prediction
    except KeyError as e:
        return f"Error: Missing feature: {e}"
    except Exception as e:  # Catch other potential errors during prediction
        return f"An error occurred: {e}"


# # Agent Definition

# In[26]:


websearch_agent = create_agent(
    model=model,
    tools=[search_web],
    name="taveli_search",
    system_prompt="You are a world class researcher with access to web search. Do not do any math or predictions."
)

math_agent = create_agent(
    model=model,
    tools=[add, multiply],
    name="math_expert",
    system_prompt="You are a math expert. Always use one tool at a time."
)

coding_agent = create_agent(
    model=model,
    tools=[python_repl],
    name="python_coder",
    system_prompt="You are a python coding expert.Always use one tool at a time."
)

ml_agent = create_agent(
    model=model,
    tools=[predict_survival],
    name="ml_expert",
    system_prompt="You are a machine learning expert in predicting the survival chances of a passenger in titanic.Always use one tool at a time"
)


# # Supervisor creation

# In[27]:


# Create supervisor workflow
workflow = create_supervisor(
    [math_agent,coding_agent,ml_agent,websearch_agent],
    model=model,
    prompt=(
        "You are a team supervisor managing a research expert, coding agent, ml expert and a math expert. "
        "For any predictions, use ml expert. Always use ml_expert for predictions."
        "For current events, use websearch_agent. "
        "For math problems, use math_agent."
        "For any coding request using python, use coding_agent.If data is passed from the supervisor use only that, if not generate the code and give the option for user to execute it manually."
    )
)


# In[28]:


# Compile and run
app = workflow.compile()


# In[29]:


workflow.compile()


# # Testing

# In[30]:


name='Braund, Mr. Owen Harris'
result = app.invoke({
    "messages": [
        {
            "role": "user",
            "content": f"""did the passenger {name} survive in titanic.use ml expert to give the prediction""".format(name)
        }
    ]
})
result['messages'][-1].content


# In[32]:


name='Heikkinen, Miss. Laina'
result = app.invoke({
    "messages": [
        {
            "role": "user",
            "content": f"""did the passenger {name} survive in titanic.use ml expert to give the prediction""".format(name)
        }
    ]
})
result['messages'][-1].content


# In[33]:


for i in range(len(result['messages'])):
  print('call'+str(i),result['messages'][i].content)


# In[34]:


result = app.invoke({
    "messages": [
        {
            "role": "user",
            "content": "multiply 11 ,2"
        }
    ]
})
result['messages'][-1].content


# In[35]:


for i in range(len(result['messages'])):
  print('call'+str(i),result['messages'][i].content)
#result['messages'][1].tool_calls[0]['name']


# In[36]:


for messag in result['messages']:
  print(messag.content)


# In[37]:


result = app.invoke({
    "messages": [
        {
            "role": "user",
            "content": "how many people survived tinanic accident"
        }
    ]
})
result['messages'][-1].content


# In[39]:


for messag in result['messages']:
  print(messag.content)


# In[40]:


for i in range(len(result['messages'])):
  print('call'+str(i),result['messages'][i])


# In[41]:


result = app.invoke({
    "messages": [
        {
            "role": "user",
            "content": "get the total head count of Meta"
        }
    ]
})
result['messages'][-1].content


# In[42]:


for i in range(len(result['messages'])):
  print('call'+str(i),result['messages'][i])


# In[44]:


get_ipython().run_line_magic('pip', 'install matplotlib')


# In[47]:


result = app.invoke({
    "messages": [
        {
            "role": "user",
            "content": """Fetch the data of the top 10 countries with the highest GDP in the world.
                          Using math expert calculate the cumulative sum.
                          Then use this data and draw a bar chart"""
        }
    ]
})
result['messages'][-1].content


# In[45]:


for i in range(len(result['messages'])):
  print('call'+str(i),result['messages'][i])


# In[46]:


for messag in result['messages']:
  print(messag.content)


# In[ ]:


import matplotlib.pyplot as plt

# GDP data for the top 10 countries
countries = [
    "United States", "China", "Germany", "Japan", "India",
    "United Kingdom", "France", "Italy", "Brazil", "Canada"
]
gdp_values = [
    27.72, 17.79, 4.53, 4.20, 3.55, 3.34, 3.03, 2.25, 2.17, 2.14
]

# Creating a line chart
plt.figure(figsize=(12, 6))
plt.plot(countries, gdp_values, marker='o')

# Adding titles and labels
plt.title('Top 10 Countries by GDP (in Trillions USD)')
plt.xlabel('Countries')
plt.ylabel('GDP (Trillions USD)')
plt.grid(True)

# Display the plot
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[ ]:


# prompt: function to give answers on a csv file using agent

def process_csv_with_agent(csv_file_path, user_query):
    """
    Processes a CSV file using the provided agent and user query.

    Args:
        csv_file_path: The path to the CSV file.
        user_query: The user's question about the CSV data.

    Returns:
        The agent's response to the user's query.
    """

    # Placeholder for the actual agent interaction logic.
    # Replace this with the code from your previous example.

    # Example:
    # Assuming 'workflow' and 'app' are defined as in the previous code.
    result = app.invoke({
        "messages": [
            {
                "role": "user",
                "content": f"Analyze the data from the CSV file '{csv_file_path}' and answer: {user_query}"
            }
        ]
    })
    return result

# Example usage
csv_file_path = '/content/titanic.csv' # Replace with your CSV file path.
user_query = "get me the list of survivors and write it to a file /content/survivors.txt"  # Replace with your query
response = process_csv_with_agent(csv_file_path, user_query)



# In[ ]:


user_query = "how many male survived the titanic disaster who are from 1st class?"  # Replace with your query
response = process_csv_with_agent(csv_file_path, user_query)
response['messages'][-1].content


# In[ ]:


for i in range(len(response['messages'])):
  print('call'+str(i),response['messages'][i])


# In[ ]:


for messag in result['messages']:
  print(messag.content)


# In[ ]:


csv_file_path = '/content/titanic.csv' # Replace with your CSV file path.
user_query = "plot the graphs to get relation ship between Survived and rest of the features which are affects the survival rate.use python coder to execute the python code "  # Replace with your query
response = process_csv_with_agent(csv_file_path, user_query)
response['messages'][-1].content

