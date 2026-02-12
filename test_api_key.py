import os
from dotenv import load_dotenv

# Force load .env
load_dotenv(override=True)

print("--- Environment Check ---")
# Mimic the app's logic
try:
    from azure.identity import DefaultAzureCredential
    from azure.keyvault.secrets import SecretClient
    
    key_vault_name = "aiforall-dev-keyvault"
    key_vault_uri = f"https://{key_vault_name}.vault.azure.net"
    
    # Use managed identity / dev creds
    credential = DefaultAzureCredential()
    client = SecretClient(vault_url=key_vault_uri, credential=credential)
    
    # Set environment variables from Key Vault
    os.environ['OPENAI_API_KEY'] = client.get_secret('openai-api-key').value
    print("Azure Key Vault: Success")
    
except Exception as e:
    print(f"Azure Key Vault: Failed or Skipped ({e})")
    print("Relying on .env")

key = os.environ.get('OPENAI_API_KEY', '')
if len(key) > 4:
    print(f"Current OPENAI_API_KEY ends with: ...{key[-4:]}")
else:
    print("Current OPENAI_API_KEY: Not found or too short")

print("\n--- LLM Connection Check ---")
try:
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o")
    response = llm.invoke("Hello, are you working?")
    print("LLM Call: Success")
    print(f"Response: {response.content}")
except Exception as e:
    print(f"LLM Call: Failed ({e})")
