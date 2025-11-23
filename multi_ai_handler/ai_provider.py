from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, AsyncIterator, TYPE_CHECKING

if TYPE_CHECKING:
    from multi_ai_handler.utils import AIResponse

class AIProvider(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_text: str=None, messages: list[dict]=None, file: str | Path | dict | None=None, model: str=None, temperature: float=0.0, local: bool=False, json_output: bool=False) -> "AIResponse":
        pass

    @abstractmethod
    def stream(self, system_prompt: str, user_text: str=None, messages: list[dict]=None, file: str | Path | dict | None=None, model: str=None, temperature: float=0.0, local: bool=False) -> Iterator[str]:
        pass

    @abstractmethod
    async def agenerate(self, system_prompt: str, user_text: str=None, messages: list[dict]=None, file: str | Path | dict | None=None, model: str=None, temperature: float=0.0, local: bool=False, json_output: bool=False) -> "AIResponse":
        pass

    @abstractmethod
    async def astream(self, system_prompt: str, user_text: str=None, messages: list[dict]=None, file: str | Path | dict | None=None, model: str=None, temperature: float=0.0, local: bool=False) -> AsyncIterator[str]:
        pass

    @abstractmethod
    def list_models(self) -> list[str]:
        pass

    @abstractmethod
    def get_model_info(self, model: str) -> dict:
        pass
