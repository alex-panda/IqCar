import argparse
from runTests import run_tests
from PIL import Image
import numpy as np
from skimage import filters, feature
import matplotlib.pyplot as plt

def IQCar():
    parser = argparse.ArgumentParser(
        description='Execute a specific test or all tests.')
    parser.add_argument(
        'function_name', type=str, nargs='?', default='all',
        help='Name of the function to test or "all" to execute all the registered functions')
    args = parser.parse_args()

    # Call test harness
    fun_handles = {
        'foo' : bar,
        'edge_detection' : edge_detection
    }
    run_tests(args.function_name, fun_handles)

def bar():
    print("Hello IQCar!")
    return 1

# Steps
def edge_detection():
    """
    Get orentation/placement of the gameboard (as well as its exit).
    """
    fig, axs = plt.subplots(2, 2)
    img = Image.open('data/IMG_1.png')
    gray_img = img.convert('L')
    axs[0, 0].imshow(img)
    axs[0, 0].set_title('Color Image')
    axs[0, 1].imshow(gray_img, cmap='gray')
    axs[0, 1].set_title('Grayscale Image')

    img = np.array(img)
    gray_img = np.array(gray_img)

    # Sobel edge detection
    thresh = 0.20
    edge_img = filters.sobel(gray_img) > thresh
    axs[1, 0].imshow(edge_img, cmap='gray')
    axs[1, 0].set_title('Sobel Edge Detection')

    # Canny edge detection
    sig = 2
    l_thresh = 50
    h_thresh = 60
    edge_img = feature.canny(gray_img, sigma=sig, low_threshold=l_thresh, high_threshold=h_thresh)
    axs[1,1].imshow(edge_img, cmap='gray')
    axs[1,1].set_title('Canny Edge Detection')

    plt.savefig('outputs/hello_edges.png')
    plt.show()


def segmentation():
    """
    Finding where the cars are on the gameboard (specefically the goal car).
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



if __name__ == "__main__":
    IQCar()
