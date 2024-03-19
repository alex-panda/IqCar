"""Solver for IQCar puzzles"""

from collections.abc import Generator
from dataclasses import dataclass
import itertools
from typing import Iterator


bitboard = int


class BoardState:
    """State of some game board"""
    # Length of the side of the board.
    BOARD_SIZE = 6

    # Row containing the goal car and exit.
    EXIT_ROW = 2

    # Length of the goal car.
    GOAL_CAR_LENGTH = 2

    def __init__(self):
        self.h_obstacles = []
        self.v_obstacles = []
        exit_row_start = self.BOARD_SIZE * self.EXIT_ROW
        self.goal_car = (1 << exit_row_start + self.GOAL_CAR_LENGTH) - (1 << exit_row_start)

    @property
    def all_cars(self) -> Iterator[bitboard]:
        return itertools.chain([self.goal_car], self.h_obstacles, self.v_obstacles)

    @property
    def state(self) -> bitboard:
        """A bitfield with car-occupied squares set to 1 and empty squares set to 0"""
        state = 0
        for obs in self.all_cars:
            state |= obs
        return state

    def is_valid(self) -> bool:
        """Check if the board is valid.

        The board is valid if no two cars overlap.

        Returns:
            True if no cars overlap.
        """
        overlaps = 0
        for obs in self.all_cars:
            overlaps ^= obs
        return overlaps == self.state

    def add_car(
            self,
            pos: tuple[int, int],
            length: int,
            *,
            horiz: bool
    ) -> "BoardState":
        """Create a BoardState with a single car.

        The given position is the top left corner of the car. If the car is too
        long to fit on the board given its position, a ValueError is raised.
        """
        x, y = pos
        start_idx = y * self.BOARD_SIZE + x
        if horiz:
            if x + length > self.BOARD_SIZE:
                raise ValueError(f"Car is too long for horizontal position "
                                 f"(position: {pos}, length: {length}, "
                                 "board size: {self.BOARD_SIZE}")
            self.h_obstacles.append((1 << start_idx + length) - (1 << start_idx))
        else:
            if y + length > self.BOARD_SIZE:
                raise ValueError(f"Car is too long for vertical position "
                                 f"(position: {pos}, length: {length}, "
                                 "board size: {cls.BOARD_SIZE}")
            self.v_obstacles.append(sum(1 << start_idx + i * self.BOARD_SIZE
                                        for i in range(length)))
        return self

    def plies(self) -> Generator["BoardState", None, None]:
        """Iterate over all single-move perturbations of the game state"""
        raise NotImplementedError

    def __hash__(self) -> int:
        return int(self.state)

    def __str__(self) -> str:
        line = "+-+-+-+-+-+-+\n"
        buf = str(line)
        for y in range(self.BOARD_SIZE):
            buf += "|"
            for x in range(self.BOARD_SIZE):
                if (2 ** (y * self.BOARD_SIZE + x)) & self.state:
                    buf += "o"
                else:
                    buf += " "
                buf += "|"
            buf += "\n" + line
        return buf


@dataclass
class Move:
    car: int
    horiz: bool
    dist: int


class Solver:
    """IQCar gameboard"""
    def __init__(self):
        self.car_states = []

    def add_car(self, pos: tuple[int, int], length: int, horiz: bool):
        self.car_states.append(BoardState.single_car(pos, length, horiz=horiz))

    def solve(self) -> list[Move]:
        raise NotImplemented
