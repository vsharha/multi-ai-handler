from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

class AIProvider(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_text: str=None, file: str | Path | dict | None=None, model:str=None, temperature: float=0.0) -> str:
        pass

    @abstractmethod
    def list_models(self) -> list[str]:
        pass