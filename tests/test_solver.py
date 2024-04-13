"""Tests for the IQCar solver"""
import itertools

import pytest

from iqcar.solver import BoardState, solve


class TestBoardState:
    def test_is_valid(self):
        """Test that non-overlapping board states are detected"""
        s = BoardState()
        s.add_car((0, 0), 2, horiz=True) \
         .add_car((5, 0), 3, horiz=False)
        assert s.is_valid()

    def test_is_not_valid(self):
        """Test that overlapping board states are detected"""
        s = BoardState()
        s.add_car((0, 0), 2, horiz=True) \
         .add_car((1, 0), 3, horiz=False)
        assert not s.is_valid()

    def test_invalid_goal_car(self):
        """Test detecting an invalid goal car position"""
        board_size = 6
        exit_row = 2
        goal_car_length = 2
        for car in itertools.product(range(board_size), range(board_size)):
            if car[1] == exit_row and car[0] + goal_car_length <= board_size:
                _ = BoardState(
                    goal_car=car,
                    board_size=board_size,
                    exit_row=exit_row,
                    goal_car_length=goal_car_length
                )
            else:
                with pytest.raises(ValueError):
                    print(car)
                    b = BoardState(
                        goal_car=car,
                        board_size=board_size,
                        exit_row=exit_row,
                        goal_car_length=goal_car_length
                    )

    def test_multiple_overlaps(self):
        """Test that multiple overlapping cars are detected"""
        s = BoardState()
        s.add_car((0, 0), 3, horiz=False) \
         .add_car((0, 2), 2, horiz=False)
        assert not s.is_valid()

    def test_more_multiple_overlaps(self):
        """Test detecting multiple overlaps"""
        s = BoardState()
        s.add_car((4, 0), 2, horiz=False)
        s.add_car((4, 1), 2, horiz=True)
        s.add_car((4, 1), 2, horiz=False)
        assert not s.is_valid()

    def test_is_solved(self):
        """Test that solved boards are detected"""
        s = BoardState()
        s.goal_car = ((1 << s.goal_car_length) - 1) << \
            (s.board_size * (s.exit_row + 1) - s.goal_car_length)
        assert s.is_solved()

    def test_enumerate_plies(self):
        """Test that plies can be enumerated"""
        s = BoardState()
        s.add_car((0, 0), 2, horiz=True)
        s.add_car((5, 3), 2, horiz=False)
        plies = list(s.plies())
        assert len(plies) == 4

    def test_ply_one_level(self):
        """Test that plies can be modified one level down"""
        s = BoardState()
        s.add_car((0, 0), 2, horiz=True)
        p = s.with_replacement(s.h_obstacles[0], 3 << 1)
        assert p.state & 3 << 1

    def test_ply_multiple_levels(self):
        """Test that plies can be modified several level down"""
        car_length = 2
        s = BoardState()
        s.add_car((0, 0), car_length, horiz=True)
        p = s.with_replacement(s.h_obstacles[0], 3 << 1)
        for i in range(s.board_size - car_length):
            p_prime = p.with_replacement(p.h_obstacles[0], 3 << i + 1)
            p = p_prime

        assert (p.state ^ 3 << s.board_size - car_length) == p.goal_car

    def test_generating_plies_from_ply(self):
        """Test that plies can generate more plies"""
        s = BoardState()
        p = s.with_replacement(s.goal_car, s.goal_car << 1)
        assert len(list(p.plies())) == 2


class TestSolver:
    def test_solve_trivial_puzzle(self):
        """Test solving the empty puzzle (only 1 car)"""
        s = BoardState()
        soln = [12288, 24576, 49152, 98304, 196608]
        assert solve(s) == soln

    def test_solve_complex_puzzle(self):
        """Test solving a nontrivial puzzle"""
        s = BoardState()
        s.add_car((0, 3), 3, horiz=False)
        s.add_car((3, 0), 3, horiz=False)
        s.add_car((1, 3), 3, horiz=True)
        soln = solve(s)
        assert len(soln) == 12

    def test_unsolvable(self):
        """Test solving an unsolvable puzzle"""
        s = BoardState()
        # Two vertical cars blocking the way
        s.add_car((5, 0), 3, horiz=False)
        s.add_car((5, 3), 3, horiz=False)
        with pytest.raises(ValueError):
            solve(s)
