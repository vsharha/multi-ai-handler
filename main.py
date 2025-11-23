import asyncio
from multi_ai_handler import request_ai, stream_ai, get_model_info, list_models, arequest_ai, astream_ai

def main():
    print(request_ai(provider="google", model="gemini-2.5-pro", user_text="What's in the file", file="test/2024-10-31_aliexpress_02.pdf"))

def stream_example():
    for chunk in stream_ai(provider="cerebras", model="gpt-oss-120b", user_text="What's in the file", file="test/2024-10-31_aliexpress_02.pdf"):
        print(chunk, end="", flush=True)
    print()

def model_info_example():
    all_models = list_models()

    provider = "ollama"
    models = all_models.get(provider, [])
    print(f"Available {provider} models: {len(models)}")

    if models:
        model = models[0]
        print(f"\nGetting info for: {model}")
        info = get_model_info(provider, model)
        print(info)

async def async_example():
    response = await arequest_ai(provider="google", model="gemini-2.0-flash", user_text="Write a haiku about async programming")
    print(response)

async def async_stream_example():
    async for chunk in astream_ai(provider="cerebras", model="gpt-oss-120b", user_text="Write 20 haikus about async programming"):
        print(chunk, end="", flush=True)
    print()

def chat_history_example():
    """Example demonstrating multi-turn conversation with history."""
    # First turn
    response1 = request_ai(
        provider="google",
        model="gemini-2.0-flash",
        system_prompt="You are a helpful assistant. Be concise.",
        user_text="My name is Alice. What's 2+2?"
    )
    print(f"Turn 1: {response1.content}")
    print(f"History length: {len(response1.history)} messages")

    # Second turn - uses history from first response
    response2 = request_ai(
        provider="google",
        model="gemini-2.0-flash",
        system_prompt="You are a helpful assistant. Be concise.",
        user_text="What's my name?",
        messages=response1.history
    )
    print(f"\nTurn 2: {response2.content}")
    print(f"History length: {len(response2.history)} messages")

    # Third turn
    response3 = request_ai(
        provider="google",
        model="gemini-2.0-flash",
        system_prompt="You are a helpful assistant. Be concise.",
        user_text="What was the math question I asked?",
        messages=response2.history
    )
    print(f"\nTurn 3: {response3.content}")
    print(f"History length: {len(response3.history)} messages")

if __name__ == "__main__":
    chat_history_example()
