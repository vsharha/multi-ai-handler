from anthropic import Anthropic, AsyncAnthropic

from multi_ai_handler.ai_provider import AIProvider
from multi_ai_handler.utils import AIResponse, parse_ai_response
from pathlib import Path
from typing import Iterator, AsyncIterator

from multi_ai_handler.generate_payload import generate_claude_payload, build_claude_user_content


class AnthropicProvider(AIProvider):
    def __init__(self):
        super().__init__()
        self.client = Anthropic()
        self.async_client = AsyncAnthropic()

    def generate(self, system_prompt: str, user_text: str=None, messages: list[dict]=None, file: str | Path | dict | None=None, model:str=None, temperature: float=0.0, local: bool=False, json_output: bool=False) -> AIResponse:
        payload: list = generate_claude_payload(user_text, file, local=local, messages=messages)

        response_text: str = ""

        with self.client.messages.stream(
            model=model,
            max_tokens=20000,
            temperature=temperature,
            system=system_prompt,
            messages=payload
        ) as stream:
            for text in stream.text_stream:
                response_text += text

        # Build history
        new_user_content = build_claude_user_content(user_text, file, local)
        history = list(messages) if messages else []
        history.append({"role": "user", "content": new_user_content})
        history.append({"role": "assistant", "content": response_text})

        content = parse_ai_response(response_text) if json_output else response_text
        return AIResponse(content=content, history=history)

    def stream(self, system_prompt: str, user_text: str=None, messages: list[dict]=None, file: str | Path | dict | None=None, model: str=None, temperature: float=0.0, local: bool=False) -> Iterator[str]:
        payload: list = generate_claude_payload(user_text, file, local=local, messages=messages)

        with self.client.messages.stream(
            model=model,
            max_tokens=20000,
            temperature=temperature,
            system=system_prompt,
            messages=payload
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

    async def agenerate(self, system_prompt: str, user_text: str=None, messages: list[dict]=None, file: str | Path | dict | None=None, model: str=None, temperature: float=0.0, local: bool=False, json_output: bool=False) -> AIResponse:
        payload: list = generate_claude_payload(user_text, file, local=local, messages=messages)

        response_text: str = ""

        async with self.async_client.messages.stream(
            model=model,
            max_tokens=20000,
            temperature=temperature,
            system=system_prompt,
            messages=payload
        ) as stream:
            async for text in stream.text_stream:
                response_text += text

        # Build history
        new_user_content = build_claude_user_content(user_text, file, local)
        history = list(messages) if messages else []
        history.append({"role": "user", "content": new_user_content})
        history.append({"role": "assistant", "content": response_text})

        content = parse_ai_response(response_text) if json_output else response_text
        return AIResponse(content=content, history=history)

    async def astream(self, system_prompt: str, user_text: str=None, messages: list[dict]=None, file: str | Path | dict | None=None, model: str=None, temperature: float=0.0, local: bool=False) -> AsyncIterator[str]:
        payload: list = generate_claude_payload(user_text, file, local=local, messages=messages)

        async with self.async_client.messages.stream(
            model=model,
            max_tokens=20000,
            temperature=temperature,
            system=system_prompt,
            messages=payload
        ) as stream:
            async for text in stream.text_stream:
                yield text