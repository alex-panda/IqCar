"""Tests for the IQCar solver"""
from iqcar.solver import BoardState


class TestBoardState:
    def test_is_valid(self):
        """Test that non-overlapping board states are detected"""
        s = BoardState()
        s.add_car((0, 0), 2, horiz=True) \
         .add_car((5, 0), 3, horiz=False)
        print(s)
        assert s.is_valid()

    def test_is_not_valid(self):
        """Test that overlapping board states are detected"""
        s = BoardState()
        s.add_car((0, 0), 2, horiz=True) \
         .add_car((1, 0), 3, horiz=False)
        assert not s.is_valid()

    def test_multiple_overlaps(self):
        """Test that multiple overlapping cars are detected"""
        s = BoardState()
        s.add_car((0, 0), 3, horiz=False) \
         .add_car((0, 2), 2, horiz=False)
        assert not s.is_valid()
