"""Solver for IQCar puzzles"""

from collections.abc import Generator
import ctypes


bitboard = ctypes.c_uint64


class BoardState(ctypes.Structure):
    """State of some game board"""
    _fields_ = [
        ("n_cars", ctypes.c_size_t),
        ("horiz", bitboard),
        ("vert", bitboard)
    ]

    # Length of the side of the board.
    BOARD_SIZE = 6

    @property
    def state(self) -> bitboard:
        """A bitfield with car-occupied squares set to 1 and empty squares set to 0"""
        return self.horiz | self.vert

    def is_valid(self) -> bool:
        """Check if the board is valid.

        The board is valid if no two cars overlap.

        Returns:
            True if horizontal and vertical cars do not overlap
        """
        return not bool(self.horiz & self.vert)

    @classmethod
    def single_car(
            cls,
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
        start_idx = y * cls.BOARD_SIZE + x
        if horiz:
            if x + length > cls.BOARD_SIZE:
                raise ValueError(f"Car is too long for horizontal position "
                                 f"(position: {pos}, length: {length}, "
                                 "board size: {cls.BOARD_SIZE}")
            return cls(1, 2 ** (start_idx + length) - (2 ** start_idx), 0)
        if y + length > cls.BOARD_SIZE:
            raise ValueError(f"Car is too long for vertical position "
                             f"(position: {pos}, length: {length}, "
                             "board size: {cls.BOARD_SIZE}")
        raise NotImplementedError

    def plies(self) -> Generator["BoardState", None, None]:
        """Iterate over all single-move perturbations of the game state"""
        raise NotImplementedError

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


class Solver:
    """IQCar gameboard"""
