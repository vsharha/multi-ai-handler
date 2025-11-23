from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

class AIProvider(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_text: str=None, file: str | Path | dict | None=None, model:str=None, temperature: float=0.0, local:bool=False) -> str:
        pass

    @abstractmethod
    def stream(self, system_prompt: str, user_text: str=None, file: str | Path | dict | None=None, model: str=None, temperature: float=0.0, local: bool=False) -> Iterator[str]:
        pass

    @abstractmethod
    def list_models(self) -> list[str]:
        pass

    @abstractmethod
    def get_model_info(self, model: str) -> dict:
        pass