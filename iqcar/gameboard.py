"""A game board."""
from iqcar.car import Car
from iqcar.solver import BoardState


class Gameboard:
    """Gameboard"""
    def __init__(self, goal_car: Car | None, cars: list[Car]) -> None:
        self.width: int  = 6
        self.height: int = 6

        # Make sure to orient the board such that this is correct
        self.exit_x: int = 2 # index of the exit row
        self.exit_y: int = 5 # index of the exit column

        self.goal_car: Car | None = goal_car
        self.cars: list[Car] = cars

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

    def __str__(self) -> str:
        return Gameboard.board_str(self.cars, self.width, self.height)

    def __repr__(self) -> str:
        return self.__str__()

    @staticmethod
    def board_str(cars: list[Car], width: int = 6, height: int = 6) -> str:
        car_map = {}
        for i, car in enumerate(cars):
            for point in car.points():
                car_map[point] = i

        l = len(str(max(len(cars) - 1, 0)))

        formatting = '{' + f':>{l}' '}|'

        full = ""
        delim = lambda: "+" + "-"*((l+1)*(width - 1)+(l)) + "+\n"

        full += delim()
        for r in range(height):
            row = "|"

            for c in range(width):

                if (c, r) in car_map:
                    row += formatting.format(car_map[(c, r)])
                else:
                    row += ' '*l + '|'

            full += row + "\n"
            full += delim()
        return full
