import json
from pathlib import Path

from enum import Enum, auto

from multi_ai_handler import OpenrouterProvider
from multi_ai_handler import AnthropicProvider
from multi_ai_handler import CerebrasProvider
from multi_ai_handler import GoogleProvider
from multi_ai_handler import OllamaProvider
from multi_ai_handler import OpenAIProvider

class LowercaseEnum(str, Enum):
    def _generate_next_value_(self, start, count, last_values):
        return self.lower()

class Providers(LowercaseEnum):
    GOOGLE = auto()
    ANTHROPIC = auto()
    OPENAI = auto()
    OPENROUTER = auto()
    OLLAMA = auto()
    CEREBRAS = auto()

SUPPORTED_MODELS = {
    Providers.GOOGLE: ["gemini-2.5-pro", "gemini-2.5-flash"],
    Providers.ANTHROPIC: ['claude-sonnet-4-5-20250929', 'claude-opus-4-1-20250805'],
    Providers.OPENAI: ['gpt-5', 'gpt-4o'],
    Providers.OPENROUTER: ['google/gemini-2.5-pro', 'google/gemini-2.5-flash', 'anthropic/claude-sonnet-4.5', 'anthropic/claude-opus-4.1'],
    Providers.OLLAMA: [],
    Providers.CEREBRAS: ['gpt-oss-120b', 'qwen-3-235b-a22b-instruct-2507', 'zai-glm-4.6'],
}

class MultiAIHandler:
    def __init__(self):
        self.providers = {
            Providers.GOOGLE: GoogleProvider,
            Providers.ANTHROPIC: AnthropicProvider,
            Providers.OPENAI: OpenAIProvider,
            Providers.OPENROUTER: OpenrouterProvider,
            Providers.OLLAMA: OllamaProvider,
            Providers.CEREBRAS: CerebrasProvider,
        }

    def request_ai(self, system_prompt: str, user_text: str=None, file: str | Path | dict | None=None, provider: str | Providers | None = None, model:str | None=None, temperature: float=0.2, json_output: bool = False) -> dict | str:
        if provider is None:
            provider = Providers.GOOGLE
        else:
            provider = Providers(provider)

        if model is None:
            if len(SUPPORTED_MODELS[provider]) > 0:
                model = SUPPORTED_MODELS[provider][0]
            else:
                raise ValueError("No model provided and no default model is available for the provider")

        Provider = self.providers[provider]
        provider_obj = Provider()

        response_text: str = provider_obj.generate(system_prompt, user_text, file, model, temperature)

        if json_output:
            return parse_ai_response(response_text)

        else:
            return response_text

_handler = MultiAIHandler()

def request_ai(
    system_prompt: str,
    user_text: str | None = None,
    file: str | Path | dict | None = None,
    provider: str | Providers | None = None,
    model: str | None = None,
    temperature: float = 0.2,
    json_output: bool = False,
) -> dict | str:
    return _handler.request_ai(
        system_prompt=system_prompt,
        user_text=user_text,
        file=file,
        provider=provider,
        model=model,
        temperature=temperature,
        json_output=json_output,
    )

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