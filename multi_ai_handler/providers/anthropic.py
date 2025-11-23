from anthropic import Anthropic

from multi_ai_handler.ai_provider import AIProvider
from pathlib import Path
from typing import Iterator

from multi_ai_handler.generate_payload import generate_claude_payload


class AnthropicProvider(AIProvider):
    def __init__(self):
        super().__init__()
        self.client = Anthropic()

    def generate(self, system_prompt: str, user_text: str=None, file: str | Path | dict | None=None, model:str=None, temperature: float=0.0, local: bool=False) -> str:
        messages: list = generate_claude_payload(user_text, file, local=local)

        response: str = ""

        with self.client.messages.stream(
            model=model,
            max_tokens=20000,
            temperature=temperature,
            system=system_prompt,
            messages=messages
        ) as stream:
            for text in stream.text_stream:
                response += text

        return response

    def stream(self, system_prompt: str, user_text: str=None, file: str | Path | dict | None=None, model: str=None, temperature: float=0.0, local: bool=False) -> Iterator[str]:
        messages: list = generate_claude_payload(user_text, file, local=local)

        with self.client.messages.stream(
            model=model,
            max_tokens=20000,
            temperature=temperature,
            system=system_prompt,
            messages=messages
        ) as stream:
            for text in stream.text_stream:
                yield text

    def list_models(self) -> list[str]:
        response = self.client.models.list()
        return [model.id for model in response.data]

    def get_model_info(self, model: str) -> dict:
        response = self.client.models.retrieve(model)
        return {
            "id": response.id,
            "created_at": response.created_at,
            "display_name": response.display_name,
        }