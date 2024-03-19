"""Tests for the IQCar solver"""
from iqcar.solver import BoardState


class TestBoardState:
    def test_is_valid(self):
        """Test that valid board states are detected"""
        s = BoardState(2, 3, 8320)
        assert s.is_valid()

    def test_is_not_valid(self):
        """Test that invalid board states are detected"""
        s = BoardState(2, 3, 67)
        assert not s.is_valid()

    def test_creating_single_car(self):
        """Test creating a single car board"""
        s = BoardState.single_car((0, 1), 2, horiz=True)
        print(f"{s.horiz:b}")
        assert s.horiz == 3
