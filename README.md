# Multi AI Handler

A unified Python library for interacting with multiple AI providers through a consistent interface. Supports text and file inputs across OpenAI, Anthropic Claude, Google Gemini, OpenRouter, Cerebras and Ollama (local LLMs).

## Features

- Unified interface for multiple AI providers
- **Streaming support** for real-time token output
- **Async support** for concurrent workloads
- Support for text-only, file-only, or combined text and file inputs
- Automatic payload formatting for each provider's API requirements
- Support for images and documents (PDF)
- Local LLM support with Ollama
- Advanced document processing with Docling (OCR, table extraction)
- Model information retrieval
- Environment-based API key management
- Optional dependencies for lightweight installations

## Supported Providers

- Anthropic Claude
- Google Gemini
- OpenAI
- OpenRouter
- Cerebras
- Ollama (Local LLMs)

## Installation

### Prerequisites

- Python 3.11 or higher

### Basic Installation

```bash
pip install multi-ai-handler
```

This installs the core package with support for Anthropic, Google, OpenAI (as well as Cerebras and OpenRouter through the OpenAI api).

### Optional Dependencies

For local LLM support and advanced document processing, install optional dependencies:

```bash
# For Ollama support (local LLMs)
pip install multi-ai-handler[ollama]

# For document processing with Docling (OCR, table extraction)
pip install multi-ai-handler[docling]

# For full local LLM support with document processing
pip install multi-ai-handler[local]

# Install all optional dependencies
pip install multi-ai-handler[all]
```

**Optional features:**
- `ollama` - Local LLM support via Ollama
- `docling` - Advanced document processing (OCR, table extraction) with EasyOCR
- `local` - Both ollama and docling for complete local setup
- `all` - All optional dependencies

## Setup

### 1. Create a `.env` file

Create a `.env` file in your project root with your API keys:

```env
ANTHROPIC_API_KEY=your_anthropic_api_key_here
CEREBRAS_API_KEY=your_cerebras_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

### 2. Import the library

```python
from multi_ai_handler import request_ai
```

## Usage

### Using `request_ai()` (Recommended)

The `request_ai()` function provides a unified interface across all providers with automatic routing and JSON parsing support.

#### Basic text request

```python
from multi_ai_handler import request_ai

response = request_ai(
    provider="google",
    model="gemini-2.5-flash",
    system_prompt="You are a helpful assistant.",
    user_text="What is the capital of France?"
)
print(response)
```

#### Using different providers

```python
# Anthropic Claude
response = request_ai(
    provider="anthropic",
    model="claude-sonnet-4-5-20250929",
    system_prompt="You are a data extraction expert.",
    user_text="Extract key information from: John Doe, age 30, lives in NYC"
)

# OpenAI
response = request_ai(
    provider="openai",
    model="gpt-4o",
    system_prompt="You are helpful.",
    user_text="Hello!"
)
```

Supported providers: `"google"`, `"anthropic"`, `"openai"`, `"openrouter"`, `"cerebras"`, `"ollama"`

*Ollama requires: `pip install multi-ai-handler[ollama]`

#### JSON output parsing

```python
# Automatically parses JSON from response
data = request_ai(
    provider="google",
    model="gemini-2.5-flash",
    system_prompt="You are a JSON formatter. Return valid JSON only.",
    user_text="Convert to JSON: Name: Alice, Age: 25, City: London",
    json_output=True
)
print(data)  # Returns parsed dict: {'name': 'Alice', 'age': 25, 'city': 'London'}
```

#### File processing (images and documents)

```python
# With images
response = request_ai(
    provider="google",
    model="gemini-2.5-flash",
    system_prompt="You are an image analysis expert.",
    user_text="Describe what you see in this image.",
    file="image.jpg"
)

# With documents
response = request_ai(
    provider="anthropic",
    model="claude-sonnet-4-5-20250929",
    system_prompt="Summarize this document.",
    file="document.pdf"
)

# Using pathlib.Path
from pathlib import Path
response = request_ai(
    provider="anthropic",
    model="claude-sonnet-4-5-20250929",
    system_prompt="Analyze this document.",
    file=Path("documents/report.pdf")
)

# Local file processing (text extraction instead of native file upload)
response = request_ai(
    provider="openai",
    model="gpt-4o",
    system_prompt="Summarize this document.",
    file="document.pdf",
    local=True  # Extracts text using Docling instead of uploading file
)
```

#### Local LLM with Ollama

```python
# Requires: pip install multi-ai-handler[ollama]
response = request_ai(
    provider="ollama",
    model="llama3.2",
    system_prompt="You are a helpful assistant.",
    user_text="What is the capital of France?"
)
```

### Streaming Responses

Stream tokens in real-time as they are generated:

```python
from multi_ai_handler import stream_ai

for chunk in stream_ai(
    provider="google",
    model="gemini-2.0-flash",
    user_text="Write a short poem about coding"
):
    print(chunk, end="", flush=True)
print()  # Newline at end
```

### Async Support

Use async/await for concurrent operations:

```python
import asyncio
from multi_ai_handler import arequest_ai, astream_ai

async def main():
    # Async generation
    response = await arequest_ai(
        provider="anthropic",
        model="claude-sonnet-4-20250514",
        user_text="Hello!"
    )
    print(response)

    # Async streaming
    async for chunk in astream_ai(
        provider="google",
        model="gemini-2.0-flash",
        user_text="Write a haiku"
    ):
        print(chunk, end="", flush=True)

asyncio.run(main())
```

#### Concurrent requests

```python
import asyncio
from multi_ai_handler import arequest_ai

async def main():
    # Run multiple requests concurrently
    responses = await asyncio.gather(
        arequest_ai(provider="google", model="gemini-2.0-flash", user_text="Hello from Google"),
        arequest_ai(provider="anthropic", model="claude-sonnet-4-20250514", user_text="Hello from Anthropic"),
        arequest_ai(provider="openai", model="gpt-4o", user_text="Hello from OpenAI"),
    )
    for i, response in enumerate(responses):
        print(f"Response {i+1}: {response[:50]}...")

asyncio.run(main())
```

### Model Information

List available models and get model details:

```python
from multi_ai_handler import list_models, get_model_info

# List all models from all providers
all_models = list_models()
print(all_models)
# {'google': ['models/gemini-2.0-flash', ...], 'anthropic': [...], ...}

# Get info for a specific model
info = get_model_info(provider="google", model="models/gemini-2.0-flash")
print(info)
# {'name': 'models/gemini-2.0-flash', 'display_name': 'Gemini 2.0 Flash', 'input_token_limit': 1048576, ...}
```

### Using AIProviderManager

For more control, use the `AIProviderManager` class directly. This allows you to register custom providers and manage the provider lifecycle.

```python
from multi_ai_handler import AIProviderManager

manager = AIProviderManager()

# Generate a response
response = manager.generate(
    provider="anthropic",
    model="claude-sonnet-4-5-20250929",
    system_prompt="You are a helpful assistant.",
    user_text="What is the capital of France?"
)

# List available models from all providers
models = manager.list_models()
print(models)  # {'google': ['gemini-2.5-pro', ...], 'anthropic': [...], ...}
```

#### Registering Custom Providers

You can register custom providers that implement the `AIProvider` interface:

```python
from multi_ai_handler import AIProviderManager
from multi_ai_handler.ai_provider import AIProvider

class MyCustomProvider(AIProvider):
    def generate(self, system_prompt, user_text=None, file=None, model=None, temperature=0.0, local=False):
        # Your implementation here
        return "response"

    def list_models(self):
        return ["model-1", "model-2"]

manager = AIProviderManager()
manager.register_provider("custom", MyCustomProvider)

response = manager.generate(
    provider="custom",
    model="model-1",
    user_text="Hello!"
)
```

### Using Provider Classes Directly

For advanced use cases, you can instantiate provider classes directly:

```python
from multi_ai_handler import (
    AnthropicProvider,
    GoogleProvider,
    OpenAIProvider,
    OpenrouterProvider,
    OllamaProvider,
    CerebrasProvider
)

# Direct provider usage
anthropic = AnthropicProvider()
response = anthropic.generate(
    system_prompt="You are helpful.",
    user_text="Hello!",
    model="claude-sonnet-4-5-20250929"
)

# List models from a specific provider
models = anthropic.list_models()
```

## API Reference

### `request_ai()` (Unified Interface)

**Recommended** - Unified function that automatically routes to the appropriate provider with built-in JSON parsing support.

```python
def request_ai(
    provider: str,
    model: str,
    system_prompt: str = None,
    user_text: str = None,
    file: str | Path | dict = None,
    temperature: float = 0.2,
    json_output: bool = False,
    local: bool = False
) -> str | dict
```

**Parameters:**
- `provider` (str, required): Provider name - `"google"`, `"anthropic"`, `"openai"`, `"openrouter"`, `"cerebras"`, or `"ollama"`
- `model` (str, required): Model to use (e.g., `"gemini-2.5-flash"`, `"claude-sonnet-4-5-20250929"`)
- `system_prompt` (str, optional): The system instruction for the AI model
- `user_text` (str, optional): The user's text input
- `file` (str | Path | dict, optional): File to process. Can be:
  - **File path**: `"image.jpg"` or `Path("image.jpg")` - automatically reads and encodes
  - **Dict**: `{"filename": "image.jpg", "encoded_data": "base64..."}` - for pre-encoded data
- `temperature` (float, optional): Controls randomness (0.0 = deterministic, 1.0 = creative). Default: 0.2
- `json_output` (bool, optional): If True, parses and returns JSON as dict. Default: False
- `local` (bool, optional): If True, extracts file text locally using Docling instead of uploading. Default: False

**Returns:**
- `str` if `json_output=False`
- `dict` if `json_output=True`

**Note**: Either `user_text` or `file` must be provided.

---

### `stream_ai()` (Streaming)

Stream tokens in real-time as they are generated.

```python
def stream_ai(
    provider: str,
    model: str,
    system_prompt: str = None,
    user_text: str = None,
    file: str | Path | dict = None,
    temperature: float = 0.2,
    local: bool = False
) -> Iterator[str]
```

**Returns:** Iterator yielding string chunks as they are generated.

---

### `arequest_ai()` (Async)

Async version of `request_ai()`.

```python
async def arequest_ai(
    provider: str,
    model: str,
    system_prompt: str = None,
    user_text: str = None,
    file: str | Path | dict = None,
    temperature: float = 0.2,
    json_output: bool = False,
    local: bool = False
) -> str | dict
```

---

### `astream_ai()` (Async Streaming)

Async streaming version.

```python
async def astream_ai(
    provider: str,
    model: str,
    system_prompt: str = None,
    user_text: str = None,
    file: str | Path | dict = None,
    temperature: float = 0.2,
    local: bool = False
) -> AsyncIterator[str]
```

---

### `list_models()`

List available models from all registered providers.

```python
def list_models() -> dict[str, list[str]]
```

**Returns:** Dictionary mapping provider names to lists of model names.

---

### `get_model_info()`

Get metadata for a specific model.

```python
def get_model_info(provider: str, model: str) -> dict
```

**Returns:** Dictionary with provider-specific model metadata.

---

### `AIProviderManager` Class

The main class for managing AI providers.

```python
class AIProviderManager:
    def __init__(self)

    def register_provider(self, name: str, provider: type[AIProvider]) -> None
        """Register a custom provider class."""

    def generate(...) -> str | dict
        """Generate a response from the specified provider."""

    def stream(...) -> Iterator[str]
        """Stream a response from the specified provider."""

    async def agenerate(...) -> str | dict
        """Async generate a response."""

    async def astream(...) -> AsyncIterator[str]
        """Async stream a response."""

    def list_models(self) -> dict[str, list[str]]
        """List available models from all registered providers."""

    def get_model_info(self, provider: str, model: str) -> dict
        """Get model metadata."""
```

---

### Provider Classes

All provider classes implement the `AIProvider` interface:

```python
class AIProvider(ABC):
    # Sync methods
    def generate(..., json_output: bool = False) -> str | dict
    def stream(...) -> Iterator[str]

    # Async methods
    async def agenerate(..., json_output: bool = False) -> str | dict
    async def astream(...) -> AsyncIterator[str]

    # Model info
    def list_models(self) -> list[str]
    def get_model_info(self, model: str) -> dict
```

**Available providers:**
- `AnthropicProvider` - Anthropic Claude API
- `GoogleProvider` - Google Gemini API
- `OpenAIProvider` - OpenAI API (supports custom base_url)
- `OpenrouterProvider` - OpenRouter API
- `OllamaProvider` - Local Ollama instance
- `CerebrasProvider` - Cerebras cloud inference

---

## Payload Generation

The library automatically formats payloads for each provider using specialized functions:

- `generate_openai_payload()`: Creates OpenAI-compatible content blocks
- `generate_gemini_payload()`: Creates Gemini-compatible parts
- `generate_claude_payload()`: Creates Claude-compatible content blocks
- `generate_ollama_payload()`: Creates Ollama-compatible messages with text extraction

These functions handle:
- MIME type detection from filenames
- Base64 data URL formatting
- Provider-specific content structure
- Document text extraction (Ollama with Docling)
- Validation of required inputs

## Error Handling

The library raises errors in the following cases:

- `ValueError`: Neither `user_text` nor `file` is provided
- `ValueError`: MIME type cannot be detected from filename
- `FileNotFoundError`: File path provided but file doesn't exist
- `ValueError`: Path provided is not a file (e.g., is a directory)
- Invalid API credentials

Example:

```python
from multi_ai_handler import request_ai

try:
    response = request_ai(
        provider="anthropic",
        model="claude-sonnet-4-5-20250929",
        system_prompt="You are helpful.",
        file="document.pdf"
    )
except FileNotFoundError as e:
    print(f"File not found: {e}")
except ValueError as e:
    print(f"Error: {e}")
```

## Best Practices

1. **Use `request_ai()` for most tasks**: The unified interface provides consistent behavior across all providers
2. **Use specific model names**: Always specify the exact model version when needed (e.g., `claude-3-5-sonnet-20241022`)
3. **Handle errors**: Wrap API calls in try-except blocks
4. **Manage API keys securely**: Never commit `.env` files to version control
5. **Optimize temperature**: Use lower values (0.0-0.3) for factual tasks, higher (0.7-1.0) for creative tasks

## License

MIT

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Support

For issues and questions, please open an issue on the GitHub repository.

