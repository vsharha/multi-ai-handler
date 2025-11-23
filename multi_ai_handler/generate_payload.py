import mimetypes
from typing import Any
import base64
from pathlib import Path

from multi_ai_handler.extract_md import extract_structured_md

def _process_file(file: str | Path | dict | None) -> tuple[str | None, str | None]:
    if file is None:
        return None, None

    if isinstance(file, dict):
        return file.get("filename"), file.get("encoded_data")

    file_path = Path(file)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    with open(file_path, "rb") as f:
        file_data = f.read()

    encoded = base64.b64encode(file_data).decode()
    return file_path.name, encoded


def process_local_file(filename: str, encoded_data: str) -> str:
    file_text = extract_structured_md(filename, encoded_data)
    return (f"""
<<<FILE CONTENT ({filename})>>>
{file_text}
<<<END FILE CONTENT>>>
""")


def build_openai_user_content(user_text: str | None, file: str | Path | dict | None=None, local: bool=False) -> list[dict[str, Any]]:
    """Build user message content for OpenAI format."""
    if not file and not user_text:
        raise ValueError("Either filename or user_text must be provided.")

    content = []

    if user_text and not file:
        content.append({
            "type": "text",
            "text": user_text
        })

    if file:
        filename, encoded_data = _process_file(file)

        if local:
            content.append({
                "type": "text",
                "text": (user_text + "\n" if user_text else "") + process_local_file(filename, encoded_data)
            })
        else:
            mime_type, _ = mimetypes.guess_type(filename)
            if not mime_type:
                raise ValueError("Could not detect MIME type from filename.")

            if user_text:
                content.append({
                    "type": "text",
                    "text": user_text
                })

            data_url = f"data:{mime_type};base64,{encoded_data}"

            if mime_type.startswith("image/"):
                content.append({
                    "type": "image_url",
                    "image_url": {"url": data_url}
                })
            elif mime_type == "application/pdf":
                content.append({
                    "type": "file",
                    "file": {
                        "filename": filename,
                        "file_data": data_url
                    }
                })

    return content


def generate_openai_payload(user_text: str | None, system_prompt: str, file: str | Path | dict | None=None, local: bool=False, messages: list[dict] | None=None) -> list[dict[str, Any]]:
    """Generate full message payload for OpenAI API.

    If messages is provided, it should contain the conversation history (without system message).
    The system message will be prepended, and the new user message will be appended.
    """
    result = []

    if system_prompt:
        result.append({
            "role": "system",
            "content": system_prompt
        })

    # Add previous conversation history
    if messages:
        result.extend(messages)

    # Add new user message
    content = build_openai_user_content(user_text, file, local)
    result.append({
        "role": "user",
        "content": content
    })

    return result

def build_google_user_parts(user_text: str | None, file: str | Path | dict | None=None, local: bool=False) -> list[dict[str, Any]]:
    """Build user message parts for Google format."""
    if not file and not user_text:
        raise ValueError("Either filename or user_text must be provided.")

    parts = []

    if user_text and not file:
        parts.append({"text": user_text})

    if file:
        filename, encoded_data = _process_file(file)

        if local:
            parts.append({
                "text": (user_text + "\n" if user_text else "") + process_local_file(filename, encoded_data)
            })
        else:
            if user_text:
                parts.append({"text": user_text})

            mime_type, _ = mimetypes.guess_type(filename)
            if not mime_type:
                raise ValueError("Could not detect MIME type from filename.")

            parts.append({
                "inline_data": {
                    "mime_type": mime_type,
                    "data": encoded_data
                }
            })

    return parts


def generate_google_payload(user_text: str | None, file: str | Path | dict | None=None, local: bool=False, messages: list[dict] | None=None) -> list[dict[str, Any]]:
    """Generate full contents payload for Google API.

    If messages is provided, it should contain the conversation history.
    Format: [{"role": "user", "parts": [...]}, {"role": "model", "parts": [...]}]
    """
    contents = []

    # Add previous conversation history
    if messages:
        contents.extend(messages)

    # Add new user message
    parts = build_google_user_parts(user_text, file, local)
    contents.append({
        "role": "user",
        "parts": parts
    })

    return contents

def build_claude_user_content(user_text: str | None, file: str | Path | dict | None=None, local: bool=False) -> list[dict[str, Any]]:
    """Build user message content for Claude format."""
    if not file and not user_text:
        raise ValueError("Either filename or user_text must be provided.")

    content = []

    if user_text and not file:
        content.append({
            "type": "text",
            "text": user_text
        })

    if file:
        filename, encoded_data = _process_file(file)

        if local:
            content.append({
                "type": "text",
                "text": (user_text + "\n" if user_text else "") + process_local_file(filename, encoded_data)
            })
        else:
            mime_type, _ = mimetypes.guess_type(filename)
            if not mime_type:
                raise ValueError("Could not detect MIME type from filename.")

            if user_text:
                content.append({
                    "type": "text",
                    "text": user_text
                })

            if mime_type.startswith("image/"):
                content_type = "image"
            else:
                content_type = "document"

            content.append({
                "type": content_type,
                "source": {
                    "type": "base64",
                    "media_type": mime_type,
                    "data": encoded_data
                }
            })

    return content


def generate_claude_payload(user_text: str | None, file: str | Path | dict | None=None, local: bool=False, messages: list[dict] | None=None) -> list[dict[str, Any]]:
    """Generate full messages payload for Claude API.

    If messages is provided, it should contain the conversation history.
    Format: [{"role": "user", "content": [...]}, {"role": "assistant", "content": "..."}]
    """
    result = []

    # Add previous conversation history
    if messages:
        result.extend(messages)

    # Add new user message
    content = build_claude_user_content(user_text, file, local)
    result.append({
        "role": "user",
        "content": content
    })

    return result

def build_ollama_user_content(user_text: str | None, file: str | Path | dict | None=None) -> str:
    """Build user message content for Ollama format (plain text)."""
    if not file and not user_text:
        raise ValueError("Either filename or user_text must be provided.")

    content = []

    if user_text:
        content.append(user_text)
    if file:
        filename, encoded_data = _process_file(file)
        content.append(process_local_file(filename, encoded_data))

    return "\n".join(content) if content else ""


def generate_ollama_payload(user_text: str | None, system_prompt: str, file: str | Path | dict | None=None, messages: list[dict] | None=None) -> list[dict[str, Any]]:
    """Generate full messages payload for Ollama API.

    If messages is provided, it should contain the conversation history (without system message).
    Format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    """
    result = []

    if system_prompt:
        result.append({
            "role": "system",
            "content": system_prompt
        })

    # Add previous conversation history
    if messages:
        result.extend(messages)

    # Add new user message
    content = build_ollama_user_content(user_text, file)
    result.append({
        "role": "user",
        "content": content
    })

    return result
