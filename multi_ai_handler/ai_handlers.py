import os
from pathlib import Path
from dotenv import load_dotenv

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from cerebras.cloud.sdk import Cerebras
    CEREBRAS_AVAILABLE = True
except ImportError:
    CEREBRAS_AVAILABLE = False

import google.generativeai as genai
from openai import OpenAI
import anthropic

from multi_ai_handler.generate_payload import generate_openai_payload, generate_google_payload, generate_claude_payload, generate_openai_payload_local

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def request_google(system_prompt: str, user_text: str=None, file: str | Path | dict | None=None, model:str=None, temperature: float=0.0) -> str:
    genai.configure(api_key=GEMINI_API_KEY)

    generative_model = genai.GenerativeModel(model, system_instruction=system_prompt)

    payload: list = generate_google_payload(user_text, file)

    response = generative_model.generate_content(
        contents=payload,
        generation_config=genai.GenerationConfig(
            temperature=temperature,
        ),
    )

    return response.text

def request_anthropic(system_prompt: str, user_text: str=None, file: str | Path | dict | None=None, model:str=None, temperature: float=0.0):
    client = anthropic.Anthropic(
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )

    messages: list = generate_claude_payload(user_text, file)

    response: str = ""

    with client.messages.stream(
        model=model,
        max_tokens=20000,
        temperature=temperature,
        system=system_prompt,
        messages=messages
    ) as stream:
        for text in stream.text_stream:
            response += text

    return response

def request_openrouter(system_prompt: str, user_text: str=None, file: str | Path | dict | None=None, model:str=None, temperature: float=0.0) -> str:
    link: str="https://openrouter.ai/api/v1"
    api_key = os.getenv("OPENROUTER_API_KEY")
    return request_openai(system_prompt, user_text, file, model, temperature, link, api_key)

def request_openai(system_prompt: str, user_text: str=None, file: str | Path | dict | None=None, model:str=None, temperature: float=0.0, link:str | None=None, api_key:str | None = None) -> str:
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")

    client = OpenAI(
        base_url=link,
        api_key=api_key,
    )

    messages: list = generate_openai_payload(user_text, system_prompt, file)

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature
    )

    return completion.choices[0].message.content

def request_ollama(system_prompt: str, user_text: str=None, file: str | Path | dict | None=None, model: str=None, temperature: float=0.0) -> str:
    if not OLLAMA_AVAILABLE:
        raise ImportError(
            "Ollama is not installed. Install it with: pip install multi-ai-handler[ollama]"
        )

    messages: list = generate_openai_payload_local(user_text, system_prompt, file)

    response = ollama.chat(
        model=model,
        messages=messages,
        options={"temperature": temperature},
    )

    return response['message']['content']

def request_cerebras(system_prompt: str, user_text: str=None, file: str | Path | dict | None=None, model: str=None, temperature: float=0.0):
    if not CEREBRAS_AVAILABLE:
        raise ImportError(
            "Cerebras is not installed. Install it with: pip install multi-ai-handler[extra]"
        )

    messages: list = generate_openai_payload_local(user_text, system_prompt, file)

    client = Cerebras(
        api_key=os.environ.get("CEREBRAS_API_KEY")
    )

    response = client.chat.completions.create(
        messages=messages,
        model=model,
        stream=False,
        max_completion_tokens=20000,
        temperature=temperature
    )

    return response.choices[0].message.content