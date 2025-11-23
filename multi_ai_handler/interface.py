from pathlib import Path
from typing import Iterator, AsyncIterator

from multi_ai_handler.multi_ai_handler import AIProviderManager

_handler = AIProviderManager()

def request_ai(
    provider: str | None = None,
    model: str | None = None,
    system_prompt: str | None = None,
    user_text: str | None = None,
    file: str | Path | dict | None = None,
    temperature: float = 0.2,
    json_output: bool = False,
    local: bool = False,
) -> dict | str:
    return _handler.generate(
        provider=provider,
        model=model,
        system_prompt=system_prompt,
        user_text=user_text,
        file=file,
        temperature=temperature,
        json_output=json_output,
        local=local,
    )

def stream_ai(
    provider: str | None = None,
    model: str | None = None,
    system_prompt: str | None = None,
    user_text: str | None = None,
    file: str | Path | dict | None = None,
    temperature: float = 0.2,
    local: bool = False,
) -> Iterator[str]:
    yield from _handler.stream(
        provider=provider,
        model=model,
        system_prompt=system_prompt,
        user_text=user_text,
        file=file,
        temperature=temperature,
        local=local,
    )

def get_model_info(provider: str, model: str) -> dict:
    return _handler.get_model_info(provider=provider, model=model)

def list_models() -> dict[str, list[str]]:
    return _handler.list_models()

async def arequest_ai(
    provider: str | None = None,
    model: str | None = None,
    system_prompt: str | None = None,
    user_text: str | None = None,
    file: str | Path | dict | None = None,
    temperature: float = 0.2,
    json_output: bool = False,
    local: bool = False,
) -> dict | str:
    return await _handler.agenerate(
        provider=provider,
        model=model,
        system_prompt=system_prompt,
        user_text=user_text,
        file=file,
        temperature=temperature,
        json_output=json_output,
        local=local,
    )

async def astream_ai(
    provider: str | None = None,
    model: str | None = None,
    system_prompt: str | None = None,
    user_text: str | None = None,
    file: str | Path | dict | None = None,
    temperature: float = 0.2,
    local: bool = False,
) -> AsyncIterator[str]:
    async for chunk in _handler.astream(
        provider=provider,
        model=model,
        system_prompt=system_prompt,
        user_text=user_text,
        file=file,
        temperature=temperature,
        local=local,
    ):
        yield chunk
