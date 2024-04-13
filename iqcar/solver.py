"""Solver for IQCar puzzles"""

from collections.abc import Generator
from collections import deque
import itertools
from typing import Iterator, Optional

from iqcar.gameboard import Gameboard


bitboard = int


def bitboard_from_gameboard(gb: Gameboard):
    gc = gb.goal_car
    goal_car_xy = gc.x, gb.y
    bs = BoardState(goal_car=goal_car_xy)


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

    def __init__(
            self,
            h_obstacles=None,
            v_obstacles=None,
            goal_car=None,
            board_size=6,
            goal_car_length=2,
            exit_row=2,
    ):
        """Constructor for a BoardState"""
        # Length of the side of the board.
        self.board_size = board_size

        # Row containing the goal car and exit.
        self.exit_row = exit_row

        # Mask for checking the exit row
        self.exit_row_mask = ((1 << self.board_size) - 1) << self.board_size * self.exit_row

        # Length of the goal car.
        self.goal_car_length = 2

        # Solved board
        self.solved_board = ((1 << self.goal_car_length) - 1) << \
        (self.board_size * self.exit_row + self.board_size - self.goal_car_length)
        self.h_obstacles = []
        if h_obstacles:
            self.h_obstacles = list(h_obstacles)
        self.v_obstacles = []
        if v_obstacles:
            self.v_obstacles = list(v_obstacles)
        exit_row_start = self.board_size * self.exit_row
        if goal_car:
            if isinstance(goal_car, tuple):
                x, y = goal_car
                goal_car = ((1 << self.goal_car_length) - 1) << \
                    self.board_size * y + x
            if goal_car & self.exit_row_mask != goal_car:
                raise ValueError(f"Goal car must be entirely in row {self.exit_row}")
            self.goal_car = goal_car
        else:
            self.goal_car = (1 << exit_row_start + self.goal_car_length) - \
                (1 << exit_row_start)

    def with_replacement(self, old: bitboard, new: bitboard) -> "BoardState":
        """Create a new BoardState from this BoardState, replacing some car with another car

        Args:
            old: the bitboard containing the car to replace
            new: the bitboard containing the new car

        Returns:
            the new BoardState with the replacement
        """
        cls = type(self)
        if old == self.goal_car:
            return cls(self.h_obstacles, self.v_obstacles, new)
        return cls(
            [h if h != old else new for h in self.h_obstacles],
            [v if v != old else new for v in self.v_obstacles],
            self.goal_car,
        )

    def all_cars(self) -> Iterator[bitboard]:
        """Get an iterator over all the cars in this board state"""
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

        Args:
            pos: the x and y coordinates of the top left corner of the car
            length: length of the car
            horiz: if True, the car is horizontal; if False, then the car is vertical

        Returns:
            this BoardState with the car addeed
        """
        x, y = pos
        start_idx = y * self.board_size + x
        if horiz:
            if x + length > self.board_size:
                raise ValueError(f"Car is too long for horizontal position "
                                 f"(position: {pos}, length: {length}, "
                                 "board size: {self.board_size}")
            self.h_obstacles.append((1 << start_idx + length) - (1 << start_idx))
        else:
            if y + length > self.board_size:
                raise ValueError(f"Car is too long for vertical position "
                                 f"(position: {pos}, length: {length}, "
                                 "board size: {cls.BOARD_SIZE}")
            self.v_obstacles.append(sum(1 << start_idx + i * self.board_size
                                        for i in range(length)))
        return self

    def plies(self) -> Generator["BoardState", None, None]:
        """Iterate over all valid single-move perturbations of the game state

        Each ply is a new game state which is one move away from this game state.

        Generates:
            BoardStates which are single-move perturbations of this state
        """
        def _plies():
            for car in itertools.chain([self.goal_car], self.h_obstacles):
                start_idx = first_set_bit(car)
                car_length = car.bit_count()
                if start_idx % self.board_size > 0:
                    # car can move left; moving left is a right shift
                    yield self.with_replacement(car, car >> 1)
                if (start_idx + car_length) % self.board_size > 0:
                    # car can move right; moving right is a left shift
                    yield self.with_replacement(car, car << 1)

            for car in self.v_obstacles:
                start_idx = first_set_bit(car)
                car_length = car.bit_count()
                if start_idx // self.board_size > 0:
                    # car can move up; moving up is a right shift by the size of the board
                    yield self.with_replacement(car, car >> self.board_size)
                if start_idx // self.board_size < self.board_size - car_length:
                    # car can move down; moving down is a left shift by the size of the board
                    yield self.with_replacement(car, car << self.board_size)

        for ply in _plies():
            if ply.is_valid():
                yield ply

    def is_solved(self) -> bool:
        """Check if the game board is solved."""
        return self.is_valid() and self.goal_car == self.solved_board

    def __hash__(self) -> int:
        return int(self.state)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and other.state == self.state

    def to_string(self, state: Optional[bitboard] = None) -> str:
        """Represent a bitboard as a string.

        Returns:
            game state laid out with cars represented with "o" characters
        """
        if not state:
            state = self.state
        line = "+-+-+-+-+-+-+\n"
        buf = str(line)
        for y in range(self.board_size):
            buf += "|"
            for x in range(self.board_size):
                if (1 << (y * self.board_size + x)) & state:
                    buf += "o"
                else:
                    buf += " "
                buf += "|"
            buf += "\n" + line
        return buf

    def __str__(self):
        return self.to_string()


def solve(board: BoardState) -> list[bitboard]:
    """Solve an IQCar puzzle.

    Simply do breadth-first search over the game state space for a valid
    solution.
    """
    paths: dict[BoardState, Optional[BoardState]] = {board: None}
    moves: deque[BoardState] = deque([board])

    while True:
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
