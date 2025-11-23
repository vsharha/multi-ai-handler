from dotenv import load_dotenv

load_dotenv()

from multi_ai_handler.multi_ai_handler import (
    request_ai,
    MultiAIHandler,
    parse_ai_response,
    Providers,
    SUPPORTED_MODELS,
)

from multi_ai_handler.providers.anthropic import AnthropicProvider
from multi_ai_handler.providers.cerebras import CerebrasProvider
from multi_ai_handler.providers.google import GoogleProvider
from multi_ai_handler.providers.ollama import OllamaProvider
from multi_ai_handler.providers.openai import OpenAIProvider
from multi_ai_handler.providers.openrouter import OpenrouterProvider

__all__ = [
    # Main unified interface
    "request_ai",
    "MultiAIHandler",
    "parse_ai_response",
    "Providers",
    "SUPPORTED_MODELS",
    # Provider-specific classes
    "AnthropicProvider",
    "CerebrasProvider",
    "GoogleProvider",
    "OllamaProvider",
    "OpenAIProvider",
    "OpenrouterProvider",
]