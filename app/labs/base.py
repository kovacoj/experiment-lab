from __future__ import annotations

from abc import ABC, abstractmethod

from app.labs.schemas import LabContext, LabResult


class BaseLab(ABC):
    lab_id: str
    lab_name: str
    scenario: str

    @abstractmethod
    def run(self, context: LabContext) -> LabResult:
        raise NotImplementedError
