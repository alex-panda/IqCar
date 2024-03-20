"""Tests for the IQCar solver"""
from iqcar.solver import BoardState, solve


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
        s.goal_car = ((1 << s.GOAL_CAR_LENGTH) - 1) << \
            (s.BOARD_SIZE * (s.EXIT_ROW + 1) - s.GOAL_CAR_LENGTH)
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
        for i in range(s.BOARD_SIZE - car_length):
            p_prime = p.with_replacement(p.h_obstacles[0], 3 << i + 1)
            p = p_prime

        assert (p.state ^ 3 << s.BOARD_SIZE - car_length) == p.goal_car

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
