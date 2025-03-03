import openai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get DeepSeek API key from environment variable
api_key = os.getenv("DEEPSEEK_API_KEY")  # Make sure it's called DEEPSEEK_API_KEY in your .env file

# Set OpenAI client with DeepSeek Base URL
client = openai.OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com/v1"  # DeepSeek's compatible API endpoint
)

try:
    # Test listing models (this endpoint should work if DeepSeek follows OpenAI's compatibility)
    response = client.models.list()
    print("✅ Authentication successful! Available models:")
    for model in response.data:
        print(f"- {model.id}")
except openai.AuthenticationError as e:
    print(f"❌ Authentication failed: {e}")
except Exception as e:
    print(f"⚠️ Some other error occurred: {e}")
