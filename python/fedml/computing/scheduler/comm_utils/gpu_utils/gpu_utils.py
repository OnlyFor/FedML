from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List


class GPUCardType(Enum):
    NVIDIA = auto()
    QUALCOMM = auto()
    UNKNOWN = auto()

    def __str__(self):
        return self.name


@dataclass
class GPUCard:
    id: int
    name: str
    driver: str
    serial: str
    memoryTotal: float
    memoryFree: float
    memoryUsed: float
    memoryUtil: float
    load: Optional[float] = 0.0
    uuid: Optional[str] = ""
    display_mode: Optional[str] = ""
    display_active: Optional[str] = ""
    temperature: Optional[float] = 0.0


class GPUCardUtil(ABC):

    @classmethod
    @abstractmethod
    def detect_gpu_card_type(cls) -> Optional[GPUCardType]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_available_gpu_card_ids(order: str, limit: int, max_load: float, max_memory: float) -> List[int]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_gpu_cards() -> List[GPUCard]:
        raise NotImplementedError
