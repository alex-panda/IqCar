"""Solver for IQCar puzzles"""

from collections.abc import Generator
from collections import deque
import itertools
from typing import Iterator, Optional


bitboard = int


def first_set_bit(x):
    """Get the first set bit of a bitfield."""
    if not hasattr(first_set_bit, '_lut'):
        first_set_bit._lut = {
            1 << i: i
            for i in range(64)
        }

    return first_set_bit._lut[x & -x]


class BoardState:
    """State of some game board"""
    # Length of the side of the board.
    BOARD_SIZE = 6

    # Row containing the goal car and exit.
    EXIT_ROW = 2

    # Mask for checking the exit row
    EXIT_ROW_MASK = ((1 << BOARD_SIZE) - 1) << BOARD_SIZE * EXIT_ROW

    # Length of the goal car.
    GOAL_CAR_LENGTH = 2

    # Solved board
    SOLVED_BOARD = ((1 << GOAL_CAR_LENGTH) - 1) << \
        (BOARD_SIZE * EXIT_ROW + BOARD_SIZE - GOAL_CAR_LENGTH)

    def __init__(self, h_obstacles=None, v_obstacles=None, goal_car=None):
        self.h_obstacles = []
        if h_obstacles:
            self.h_obstacles = list(h_obstacles)
        self.v_obstacles = []
        if v_obstacles:
            self.v_obstacles = list(v_obstacles)
        exit_row_start = self.BOARD_SIZE * self.EXIT_ROW
        if goal_car:
            if not goal_car & self.EXIT_ROW_MASK:
                raise ValueError(f"Goal car must be in row {self.EXIT_ROW}")
            self.goal_car = goal_car
        else:
            self.goal_car = (1 << exit_row_start + self.GOAL_CAR_LENGTH) - \
                (1 << exit_row_start)

    def with_replacement(self, old: bitboard, new: bitboard) -> "BoardState":
        cls = type(self)
        if old == self.goal_car:
            return cls(self.h_obstacles, self.v_obstacles, new)
        return cls(
            [h if h != old else new for h in self.h_obstacles],
            [v if v != old else new for v in self.v_obstacles],
            self.goal_car,
        )

    def all_cars(self) -> Iterator[bitboard]:
        return itertools.chain(
            [self.goal_car],
            self.h_obstacles,
            self.v_obstacles
        )

    @property
    def state(self) -> bitboard:
        """A bitfield with car-occupied squares set to 1 and empty squares set to 0"""
        state = 0
        for obs in self.all_cars():
            state |= obs
        return state

    def is_valid(self) -> bool:
        """Check if the board is valid.

        The board is valid if no two cars overlap.

        Returns:
            True if no cars overlap.
        """
        return sum(self.all_cars()) == self.state

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
        """Iterate over all valid single-move perturbations of the game state"""
        def _plies():
            for car in itertools.chain([self.goal_car], self.h_obstacles):
                start_idx = first_set_bit(car)
                car_length = car.bit_count()
                if start_idx % self.BOARD_SIZE > 0:
                    # car can move left; moving left is a right shift
                    yield self.with_replacement(car, car >> 1)
                if (start_idx + car_length) % self.BOARD_SIZE > 0:
                    # car can move right; moving right is a left shift
                    yield self.with_replacement(car, car << 1)

            for car in self.v_obstacles:
                start_idx = first_set_bit(car)
                car_length = car.bit_count()
                if start_idx // self.BOARD_SIZE > 0:
                    # car can move up; moving up is a right shift by the size of the board
                    yield self.with_replacement(car, car >> self.BOARD_SIZE)
                if start_idx // self.BOARD_SIZE < self.BOARD_SIZE - car_length:
                    # car can move down; moving down is a left shift by the size of the board
                    yield self.with_replacement(car, car << self.BOARD_SIZE)

        for ply in _plies():
            if ply.is_valid():
                yield ply

    def is_solved(self):
        return self.is_valid() and self.goal_car == self.SOLVED_BOARD

    def __hash__(self) -> int:
        return int(self.state)

    def bitboard_to_string(self, state: bitboard) -> str:
        line = "+-+-+-+-+-+-+\n"
        buf = str(line)
        for y in range(self.BOARD_SIZE):
            buf += "|"
            for x in range(self.BOARD_SIZE):
                if (1 << (y * self.BOARD_SIZE + x)) & state:
                    buf += "o"
                else:
                    buf += " "
                buf += "|"
            buf += "\n" + line
        return buf

    def __str__(self):
        return self.bitboard_to_string(self.state)


def solve(board: BoardState) -> list[bitboard]:
    """Solve an IQCar puzzle.

    Simply do breadth-first search over the game state space for a valid
    solution.
    """
    paths: dict[BoardState, Optional[BoardState]] = {board: None}
    moves: deque[BoardState] = deque([board])

    while True:
        print(moves)
        if not moves:
            raise ValueError("Ran out of moves to try")
        curr = moves.pop()
        if curr.is_solved():
            path = []
            p: Optional[BoardState] = curr
            while p is not None:
                path.append(p.state)
                p = paths[p]
            path.reverse()
            return path

        next_moves = list(curr.plies())
        if not next_moves:
            continue

        for move in next_moves:
            if move in paths:
                continue
            paths[move] = curr
            moves.appendleft(move)
