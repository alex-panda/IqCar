"""A game board."""
from iqcar.car import Car


class Gameboard:
    """Gameboard"""
    def __init__(self, goal_car: Car, cars: list[Car]) -> None:
        self.width: int  = 6
        self.height: int = 6

        # Make sure to orient the board such that this is correct
        self.exit_x: int = 2 # index of the exit row
        self.exit_y: int = 5 # index of the exit column

        self.goal_car: Car  = goal_car
        self.cars: set[Car] = set(cars)

        # generate ids

        for (i, car) in enumerate(cars):
            car.id = i

        self.goal_car.id = -1

    def solve(self):
        pass
