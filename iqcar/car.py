from typing import Generator


class Car:
    """
    An IQCar.
    """

    def __init__(self, x: int, y: int, horizontal: bool, length: int):
        self.x: int = x
        self.y: int = y
        self.horizontal: bool = horizontal
        self.length: int = length
        self.id: int = 0

    def hits(self, other: "Car") -> bool:
        """
        Returns `True` if this car intersects with the given car and `False`
        otherwise.
        """
        for p1 in self.points():
            for p2 in other.points():
                if p1 == p2:
                    return True
        return False

    def pos(self) -> tuple[int, int]:
        """
        Returns the (x, y) position of this car.
        """
        return (self.x, self.y)

    def points(self) -> Generator[tuple[int, int], None, None]:
        """
        Returns all points that this `Car` occupies.
        """
        if self.horizontal:
            for i in range(self.length):
                yield (self.x + i, self.y)
        else:
            for i in range(self.length):
                yield (self.x, self.y + i)

    def __str__(self) -> str:
        return f"<Car(x={self.x},y={self.y},horizontal={self.horizontal},length={self.length},id={self.id})>"

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __hash__(self):
        return hash((self.x, self.y, self.horizontal, self.length))
