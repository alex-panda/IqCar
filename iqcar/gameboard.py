"""A game board."""
from iqcar.car import Car
from iqcar.solver import BoardState


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

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return self.goal_car == other.goal_car and self.cars == other.cars

    def into_state(self) -> BoardState:
        return BoardState.from_gameboard(self) 
