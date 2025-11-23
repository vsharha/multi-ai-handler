from multi_ai_handler.ai_provider import AIProvider
from multi_ai_handler.utils import AIResponse, parse_ai_response
from pathlib import Path
from typing import Iterator, AsyncIterator
import requests

from multi_ai_handler.generate_payload import generate_ollama_payload, build_ollama_user_content

try:
    import ollama
    from ollama import AsyncClient
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    AsyncClient = None

class OllamaServerError(RuntimeError):
    pass


class OllamaProvider(AIProvider):
    def __init__(self, base_url: str = "http://localhost:11434"):
        super().__init__()
        self.base_url = base_url.rstrip("/")

    def _check_server(self):
        if not OLLAMA_AVAILABLE:
            raise ImportError(
                "Ollama is not installed. Install it with: pip install multi-ai-handler[ollama]"
            )

        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=2)
        except requests.exceptions.ConnectionError:
            raise OllamaServerError(
                f"Ollama server is not running at {self.base_url}. "
                f"Start it with: `ollama serve`"
            )
        except requests.exceptions.RequestException as e:
            raise OllamaServerError(
                f"Could not communicate with Ollama server at {self.base_url}: {e}"
            )

        if resp.status_code >= 500:
            raise OllamaServerError(
                f"Ollama server responded with {resp.status_code} (server error)"
            )

    def generate(self, system_prompt: str, user_text: str = None, messages: list[dict] = None, file: str | Path | dict | None = None, model: str = None, temperature: float = 0.0, local: bool=False, json_output: bool=False) -> AIResponse:
        self._check_server()

        payload: list = generate_ollama_payload(user_text, system_prompt, file, messages=messages)

        response = ollama.chat(
            model=model,
            messages=payload,
            options={"temperature": temperature},
        )

        response_text = response['message']['content']

        # Build history (without system message)
        new_user_content = build_ollama_user_content(user_text, file)
        history = list(messages) if messages else []
        history.append({"role": "user", "content": new_user_content})
        history.append({"role": "assistant", "content": response_text})

        content = parse_ai_response(response_text) if json_output else response_text
        return AIResponse(content=content, history=history)

    def stream(self, system_prompt: str, user_text: str=None, messages: list[dict]=None, file: str | Path | dict | None=None, model: str=None, temperature: float=0.0, local: bool=False) -> Iterator[str]:
        self._check_server()

        payload: list = generate_ollama_payload(user_text, system_prompt, file, messages=messages)

        stream = ollama.chat(
            model=model,
            messages=payload,
            options={"temperature": temperature},
            stream=True
        )

        for chunk in stream:
            if chunk['message']['content']:
                yield chunk['message']['content']

    def list_models(self) -> list[str]:
        self._check_server()

        resp = requests.get(f"{self.base_url}/api/tags")
        resp.raise_for_status()
        data = resp.json()

        return [model["name"] for model in data.get("models", [])]

    def get_model_info(self, model: str) -> dict:
        self._check_server()

        resp = requests.post(f"{self.base_url}/api/show", json={"name": model})
        resp.raise_for_status()
        data = resp.json()

        return {
            "name": model,
            "modified_at": data.get("modified_at"),
            "size": data.get("size"),
            "parameters": data.get("details", {}).get("parameter_size"),
        }

    async def agenerate(self, system_prompt: str, user_text: str=None, messages: list[dict]=None, file: str | Path | dict | None=None, model: str=None, temperature: float=0.0, local: bool=False, json_output: bool=False) -> AIResponse:
        self._check_server()

        payload: list = generate_ollama_payload(user_text, system_prompt, file, messages=messages)

        async_client = AsyncClient(host=self.base_url)
        response = await async_client.chat(
            model=model,
            messages=payload,
            options={"temperature": temperature},
        )

        response_text = response['message']['content']

        # Build history (without system message)
        new_user_content = build_ollama_user_content(user_text, file)
        history = list(messages) if messages else []
        history.append({"role": "user", "content": new_user_content})
        history.append({"role": "assistant", "content": response_text})

        content = parse_ai_response(response_text) if json_output else response_text
        return AIResponse(content=content, history=history)

    async def astream(self, system_prompt: str, user_text: str=None, messages: list[dict]=None, file: str | Path | dict | None=None, model: str=None, temperature: float=0.0, local: bool=False) -> AsyncIterator[str]:
        self._check_server()

        payload: list = generate_ollama_payload(user_text, system_prompt, file, messages=messages)

        async_client = AsyncClient(host=self.base_url)
        stream = await async_client.chat(
            model=model,
            messages=payload,
            options={"temperature": temperature},
            stream=True
        )

        async for chunk in stream:
            if chunk['message']['content']:
                yield chunk['message']['content']
