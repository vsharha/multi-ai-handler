from multi_ai_handler import request_ai, stream_ai, get_model_info, list_models, AIProviderManager

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

def file_conversation_example():
    manager = AIProviderManager()
    conv = manager.conversation(
        provider="ollama",
        model="gpt-oss",
        system_prompt="You are a helpful assistant that analyzes documents.",
    )

    response = conv.send("Summarize this document", file="test/2024-10-31_aliexpress_02.pdf")
    print(f"Summary: {response.content}\n")

    response = conv.send("What is the total amount?")
    print(f"Total: {response.content}\n")

    response = conv.send("List all the items")
    print(f"Items: {response.content}\n")

    print(f"Conversation: {conv}")

if __name__ == "__main__":
    file_conversation_example()
