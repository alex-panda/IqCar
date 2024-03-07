import argparse
from runTests import run_tests
from PIL import Image
import numpy as np

def runHw4():
    parser = argparse.ArgumentParser(
        description='Execute a specific test or all tests.')
    parser.add_argument(
        'function_name', type=str, nargs='?', default='all',
        help='Name of the function to test or "all" to execute all the registered functions')
    args = parser.parse_args()

    # Call test harness
    fun_handles = {
        'foo' : bar
    }
    run_tests(args.function_name, fun_handles)

def bar():
    print("Hello IQCar!")
    return 1

if __name__ == "__main__":
    runHw4()


# Steps

def segmentation():
    """
    Finding where the cars are on the gameboard (specefically the goal car).
    """

def edge_detection():
    """
    Get orentation/placement of the gameboard (as well as its exit).
    """

def parse_into_game_board():
    """
    Turn images into the internal gameboard representation.
    """

def solve():
    """
    Solve the gameboard.
    """

def simple_gameboard_image():
    """
    Generate overlay for image.
    """



