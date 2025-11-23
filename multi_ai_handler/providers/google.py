from google import genai
from google.genai import types
from pathlib import Path
from typing import Iterator, AsyncIterator

from multi_ai_handler.ai_provider import AIProvider
from multi_ai_handler.utils import AIResponse, parse_ai_response
from multi_ai_handler.generate_payload import generate_google_payload, build_google_user_parts

class GoogleProvider(AIProvider):
    def __init__(self):
        super().__init__()
        self.client = genai.Client()
        self.async_client = self.client.aio

    def generate(self, system_prompt: str, user_text: str=None, messages: list[dict]=None, file: str | Path | dict | None=None, model:str=None, temperature: float=0.0, local: bool=False, json_output: bool=False) -> AIResponse:
        payload: list = generate_google_payload(user_text, file, local=local, messages=messages)

        response = self.client.models.generate_content(
            model=model,
            contents=payload,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature
            )
        )

        response_text = response.text

        # Build history (Google uses "model" for assistant role)
        new_user_parts = build_google_user_parts(user_text, file, local)
        history = list(messages) if messages else []
        history.append({"role": "user", "parts": new_user_parts})
        history.append({"role": "model", "parts": [{"text": response_text}]})

        content = parse_ai_response(response_text) if json_output else response_text
        return AIResponse(content=content, history=history)

    def stream(self, system_prompt: str, user_text: str=None, messages: list[dict]=None, file: str | Path | dict | None=None, model: str=None, temperature: float=0.0, local: bool=False) -> Iterator[str]:
        payload: list = generate_google_payload(user_text, file, local=local, messages=messages)

        response = self.client.models.generate_content_stream(
            model=model,
            contents=payload,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature
            )
        )

        for chunk in response:
            if chunk.text:
                yield chunk.text

    def list_models(self) -> list[str]:
        response = self.client.models.list()
        return [model.name for model in response]

    def get_model_info(self, model: str) -> dict:
        response = self.client.models.get(model=model)
        return {
            "name": response.name,
            "display_name": response.display_name,
            "input_token_limit": response.input_token_limit,
            "output_token_limit": response.output_token_limit,
        }

    async def agenerate(self, system_prompt: str, user_text: str=None, messages: list[dict]=None, file: str | Path | dict | None=None, model: str=None, temperature: float=0.0, local: bool=False, json_output: bool=False) -> AIResponse:
        payload: list = generate_google_payload(user_text, file, local=local, messages=messages)

        response = await self.async_client.models.generate_content(
            model=model,
            contents=payload,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature
            )
        )

        response_text = response.text

        # Build history (Google uses "model" for assistant role)
        new_user_parts = build_google_user_parts(user_text, file, local)
        history = list(messages) if messages else []
        history.append({"role": "user", "parts": new_user_parts})
        history.append({"role": "model", "parts": [{"text": response_text}]})

        content = parse_ai_response(response_text) if json_output else response_text
        return AIResponse(content=content, history=history)

    async def astream(self, system_prompt: str, user_text: str=None, messages: list[dict]=None, file: str | Path | dict | None=None, model: str=None, temperature: float=0.0, local: bool=False) -> AsyncIterator[str]:
        payload: list = generate_google_payload(user_text, file, local=local, messages=messages)

        response = await self.async_client.models.generate_content_stream(
            model=model,
            contents=payload,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature
            )
        )

        async for chunk in response:
            if chunk.text:
                yield chunk.text