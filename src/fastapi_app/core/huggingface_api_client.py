""" Handles interaction with Hugging Face Inference API. """

import os

import requests
from dotenv import load_dotenv


class HuggingFaceAPIClient:
    """
    Handles interaction with Hugging Face Inference API.
    """

    def __init__(self):
        load_dotenv()
        self.api_token = os.getenv("HUGGING_FACE_API_KEY")
        self.model_name = "meta-llama/Llama-3.2-3B-Instruct"
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"
        self.headers = {"Authorization": f"Bearer {self.api_token}"}

    def generate_response(self, prompt: str) -> str:
        """
        Generates a response from the Hugging Face API based on the provided prompt.

        Args:
            prompt (str): The input text prompt to generate a response for.

        Returns:
            str: The generated response text from the Hugging Face API. If an error occurs,
                 returns an error message indicating the failure.

        Raises:
            requests.exceptions.RequestException: If there is an issue with the HTTP request.
        """
        payload = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": 50, "temperature": 0.7},
        }
        try:
            response = requests.post(
                self.api_url, headers=self.headers, json=payload, timeout=30
            )
            response.raise_for_status()
            generated_text = response.json()[0]["generated_text"]
            return generated_text.strip()
        except requests.exceptions.RequestException as e:
            print(f"Error details: {e}")
            print(
                f"Response content: {response.text if 'response' in locals() else 'No response received'}"
            )
            return "Error generating response from Hugging Face API"
