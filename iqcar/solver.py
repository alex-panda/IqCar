"""Solver for IQCar puzzles"""

from collections.abc import Generator
import ctypes


class BoardState(ctypes.Structure):
    """State of some game board"""
    _fields_ = [
        ("n_cars", ctypes.c_size_t),
        ("horiz", ctypes.c_uint64),
        ("vert", ctypes.c_uint64)
    ]

    # Length of the side of the board.
    BOARD_SIZE = 6

    @property
    def state(self) -> ctypes.c_uint64:
        """A bitfield with car-occupied squares set to 1 and empty squares set to 0"""
        return self.horiz | self.vert

    def is_valid(self) -> bool:
        """Check if the board is valid.

        The board is valid if no two cars overlap.

        Returns:
            True if horizontal and vertical cars do not overlap
        """
        return not bool(self.horiz & self.vert)

    def plies(self) -> Generator["BoardState", None, None]:
        pass


class Solver:
    """IQCar gameboard"""
