from openai import OpenAI, AsyncOpenAI

from multi_ai_handler.ai_provider import AIProvider
from multi_ai_handler.utils import AIResponse, parse_ai_response
import os
from pathlib import Path
from typing import Iterator, AsyncIterator

from multi_ai_handler.generate_payload import generate_openai_payload, build_openai_user_content

class OpenAIProvider(AIProvider):
    def __init__(self, base_url: str | None=None, api_key: str | None=None, local: bool=False) -> None:
        super().__init__()
        self.local = local
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self.async_client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
        )

    def generate(self, system_prompt: str, user_text: str=None, messages: list[dict]=None, file: str | Path | dict | None=None, model:str=None, temperature: float=0.0, local: bool=False, json_output: bool=False) -> AIResponse:
        if self.local:
            local = True

        payload: list = generate_openai_payload(user_text, system_prompt, file, local=local, messages=messages)

        completion = self.client.chat.completions.create(
            model=model,
            messages=payload,
            temperature=temperature
        )

        response_text = completion.choices[0].message.content

        # Build history (without system message)
        new_user_content = build_openai_user_content(user_text, file, local)
        history = list(messages) if messages else []
        history.append({"role": "user", "content": new_user_content})
        history.append({"role": "assistant", "content": response_text})

        content = parse_ai_response(response_text) if json_output else response_text
        return AIResponse(content=content, history=history)

    def stream(self, system_prompt: str, user_text: str=None, messages: list[dict]=None, file: str | Path | dict | None=None, model: str=None, temperature: float=0.0, local: bool=False) -> Iterator[str]:
        if self.local:
            local = True

        payload: list = generate_openai_payload(user_text, system_prompt, file, local=local, messages=messages)

        stream = self.client.chat.completions.create(
            model=model,
            messages=payload,
            temperature=temperature,
            stream=True
        )

        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    def list_models(self) -> list[str]:
        response = self.client.models.list()
        return [model.id for model in response.data]

    def get_model_info(self, model: str) -> dict:
        response = self.client.models.retrieve(model)
        return {
            "id": response.id,
            "created": response.created,
            "owned_by": response.owned_by,
        }

    async def agenerate(self, system_prompt: str, user_text: str=None, messages: list[dict]=None, file: str | Path | dict | None=None, model: str=None, temperature: float=0.0, local: bool=False, json_output: bool=False) -> AIResponse:
        if self.local:
            local = True

        payload: list = generate_openai_payload(user_text, system_prompt, file, local=local, messages=messages)

        completion = await self.async_client.chat.completions.create(
            model=model,
            messages=payload,
            temperature=temperature
        )

        response_text = completion.choices[0].message.content

        # Build history (without system message)
        new_user_content = build_openai_user_content(user_text, file, local)
        history = list(messages) if messages else []
        history.append({"role": "user", "content": new_user_content})
        history.append({"role": "assistant", "content": response_text})

        content = parse_ai_response(response_text) if json_output else response_text
        return AIResponse(content=content, history=history)

    async def astream(self, system_prompt: str, user_text: str=None, messages: list[dict]=None, file: str | Path | dict | None=None, model: str=None, temperature: float=0.0, local: bool=False) -> AsyncIterator[str]:
        if self.local:
            local = True

        payload: list = generate_openai_payload(user_text, system_prompt, file, local=local, messages=messages)

        stream = await self.async_client.chat.completions.create(
            model=model,
            messages=payload,
            temperature=temperature,
            stream=True
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content