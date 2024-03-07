

from typing import Generator


class Car:
    def __init__(self, x: int, y: int, horizontal: bool, length: int):
        self.x: int = x
        self.y: int = y
        self.horizontal: bool = horizontal
        self.length: int = length
        self.id: int = 0

    def hits(self, other: "Car") -> bool:
        for p1 in self.points():
            for p2 in other.points():
                if p1 == p2:
                    return True
        return False

    def points(self) -> Generator[(int, int)]:
        for i in range(self.length):
            if self.horizontal:
                yield (self.x, self.y + i)
            else:
                yield (self.x + i, self.y)

