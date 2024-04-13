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
        for i in range(self.length):
            if self.horizontal:
                yield (self.x, self.y + i)
            else:
                yield (self.x + i, self.y)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return (self.x, self.y, self.horizontal, self.length) == \
            (other.x, other.y, other.horizontal, other.length)
