from multi_ai_handler.providers.openai import OpenAIProvider
import os

class CerebrasProvider(OpenAIProvider):
    def __init__(self):
        super().__init__(
            base_url="https://api.cerebras.ai/v1",
            api_key=os.getenv("CEREBRAS_API_KEY"),
            local=True
        )