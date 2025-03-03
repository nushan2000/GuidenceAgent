from langchain.schema.runnable import Runnable
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DeepSeekAPI(Runnable):
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = "https://api.deepseek.com/v1"  # Replace with actual DeepSeek API endpoint

    def invoke(self, input_text, **kwargs):
        """
        This method is required by Langchain's Runnable interface.
        It replaces your generate_response() method.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "deepseek-model-name",  # Replace with actual model name
            "messages": [
                {"role": "system", "content": "You are an AI assistant that answers questions about faculty bylaws."},
                {"role": "user", "content": input_text}
            ]
        }
        response = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=data)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']

