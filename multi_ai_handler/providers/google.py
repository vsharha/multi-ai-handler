import google.generativeai as genai
from dotenv import load_dotenv
import os
from pathlib import Path

from multi_ai_handler.ai_provider import AIProvider
from multi_ai_handler.generate_payload import generate_google_payload

load_dotenv()

class GoogleProvider(AIProvider):
    def generate(self, system_prompt: str, user_text: str=None, file: str | Path | dict | None=None, model:str=None, temperature: float=0.0, link:str | None=None, api_key:str | None = None) -> str:
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

        genai.configure(api_key=GEMINI_API_KEY)

        generative_model = genai.GenerativeModel(model, system_instruction=system_prompt)

        payload: list = generate_google_payload(user_text, file)

        response = generative_model.generate_content(
            contents=payload,
            generation_config=genai.GenerationConfig(
                temperature=temperature,
            ),
        )

        return response.text