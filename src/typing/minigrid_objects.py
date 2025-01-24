from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class VariableType(Enum):
    OBJECT = "object"
    COLOR = "color"
    STATE = "state"
    AGENT = "agent"

@dataclass
class GridPosition:
    x: int
    y: int

    def to_tuple(self) -> Tuple[int, int]:
        return (self.x, self.y)
    
    def __hash__(self):
        return hash((self.x, self.y))

@dataclass
class GridSize:
    width: int
    height: int

    def to_pixel_size(self, tile_size: int) -> Tuple[int, int]:
        return (self.width * tile_size, self.height * tile_size)
