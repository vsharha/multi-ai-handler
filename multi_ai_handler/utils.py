import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, AsyncIterator, TYPE_CHECKING

if TYPE_CHECKING:
    from multi_ai_handler.ai_provider import AIProvider


@dataclass
class AIResponse:
    content: str
    history: list[dict] | None = None

    def __str__(self) -> str:
        return self.content

    def __repr__(self) -> str:
        return f"AIResponse(content='{self.content[:50]}...', history={len(self.history) if self.history else 0} messages)"


class Conversation:
    def __init__(
        self,
        handler: "AIProvider",
        model: str | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.2,
        local: bool = False,
    ):
        self.handler = handler
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.local = local
        self.history: list[dict] = []

    def send(
        self,
        user_text: str | None = None,
        file: str | Path | dict | None = None,
        json_output: bool = False,
    ) -> AIResponse:
        response = self.handler.generate(
            system_prompt=self.system_prompt,
            user_text=user_text,
            messages=self.history if self.history else None,
            file=file,
            model=self.model,
            temperature=self.temperature,
            local=self.local,
            json_output=json_output,
        )

        if response.history:
            self.history = response.history

        return response

    def stream(
        self,
        user_text: str | None = None,
        file: str | Path | dict | None = None,
    ) -> Iterator[str]:
        yield from self.handler.stream(
            system_prompt=self.system_prompt,
            user_text=user_text,
            messages=self.history if self.history else None,
            file=file,
            model=self.model,
            temperature=self.temperature,
            local=self.local,
        )

    async def asend(
        self,
        user_text: str | None = None,
        file: str | Path | dict | None = None,
        json_output: bool = False,
    ) -> AIResponse:
        response = await self.handler.agenerate(
            system_prompt=self.system_prompt,
            user_text=user_text,
            messages=self.history if self.history else None,
            file=file,
            model=self.model,
            temperature=self.temperature,
            local=self.local,
            json_output=json_output,
        )

        if response.history:
            self.history = response.history

        return response

    async def astream(
        self,
        user_text: str | None = None,
        file: str | Path | dict | None = None,
    ) -> AsyncIterator[str]:
        async for chunk in self.handler.astream(
            system_prompt=self.system_prompt,
            user_text=user_text,
            messages=self.history if self.history else None,
            file=file,
            model=self.model,
            temperature=self.temperature,
            local=self.local,
        ):
            yield chunk

    def clear(self) -> None:
        self.history = []

    def __len__(self) -> int:
        return len(self.history)

    def __repr__(self) -> str:
        return f"Conversation(model={self.model}, messages={len(self.history)})"


def parse_ai_response(response_text: str) -> dict:
    response_text = response_text.strip()
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        start: int = 0
        if "```json" in response_text:
            start = response_text.find("```json") + 7
        elif "```" in response_text:
            start = response_text.find("```") + 3

        if start != 0:
            end: int = response_text.find("```", start)
            if end != -1:
                response_text = response_text[start:end]

    try:
        return json.loads(response_text)
    except json.decoder.JSONDecodeError as e:
        raise Exception(e)
