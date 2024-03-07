

class Car:
    def __init__(self, x: int, y: int, horizontal: bool, length: int):
        self.x: int = x
        self.y: int = y
        self.horizontal: bool = horizontal
        self.length: int = length

    def hits(self, other: "Car") -> bool:
        pass
