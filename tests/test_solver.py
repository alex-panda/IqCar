"""Tests for the IQCar solver"""
from iqcar.solver import BoardState


class TestBoardState:
    def test_is_valid(self):
        """Test that valid board states are detected"""
        s = BoardState(3, 8320)
        assert s.is_valid()

    def test_is_not_valid(self):
        """Test that invalid board states are detected"""
        s = BoardState(3, 129)
        assert not s.is_valid()
