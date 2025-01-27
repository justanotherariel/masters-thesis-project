from dataclasses import dataclass
from enum import Enum


class VariableType(Enum):
    OBJECT = "object"
    COLOR = "color"
    STATE = "state"
    AGENT = "agent"


@dataclass
class GridPosition:
    x: int
    y: int

    def to_tuple(self) -> tuple[int, int]:
        return (self.x, self.y)

    def __hash__(self):
        return hash((self.x, self.y))


@dataclass
class GridSize:
    width: int
    height: int

    def to_pixel_size(self, tile_size: int) -> tuple[int, int]:
        return (self.width * tile_size, self.height * tile_size)
