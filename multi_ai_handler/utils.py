import json
from dataclasses import dataclass


@dataclass
class AIResponse:
    content: str
    history: list[dict] | None = None

    def __str__(self) -> str:
        return self.content

    def __repr__(self) -> str:
        return f"AIResponse(content='{self.content[:50]}...', history={len(self.history) if self.history else 0} messages)"


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
