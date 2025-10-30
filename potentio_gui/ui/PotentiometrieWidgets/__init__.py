from dataclasses import dataclass
from typing import Optional


@dataclass
class Datapoint:
    volume: float
    ph: float


@dataclass
class OptionalDatapoint:
    volume: float | None
    ph: float | None
    enabled: bool = True

    def to_datapoint(self) -> Optional[Datapoint]:
        if self.volume is not None and self.ph is not None and self.enabled:
            return Datapoint(self.volume, self.ph)
        return None
