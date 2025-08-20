# API.py
import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

# 1) Use the GitHub Models endpoint (per MS docs)
ENDPOINT = "https://models.inference.ai.azure.com"
# (Org-specific endpoint also exists: "https://models.github.ai/inference")

# 2) Set an environment variable named GITHUB_TOKEN (see steps below)
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise RuntimeError("GITHUB_TOKEN is not set. Define it in your environment.")

# 3) Pick a valid model name from the catalog (e.g., 'gpt-4o', 'gpt-4.1', 'mistral-large', etc.)
MODEL = "gpt-4o"

client = ChatCompletionsClient(
    endpoint=ENDPOINT,
    credential=AzureKeyCredential(GITHUB_TOKEN),
    model=MODEL,  # you can also pass model in the .complete(...) call
)

response = client.complete(
    messages=[
        SystemMessage("You are a helpful assistant."),
        UserMessage("What is the capital of France?"),
    ]
)

print(response.choices[0].message.content)
