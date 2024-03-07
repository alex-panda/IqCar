import argparse
from runTests import run_tests
from PIL import Image
import numpy as np
from skimage import filters, feature
import matplotlib.pyplot as plt
import cv2

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
        'edge_detection' : edge_detection,
        'segmentation' : segmentation
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
    img = Image.open('data/IMG_1.png')
    gray_img = img.convert('L')

    img = np.array(img)
    gray_img = np.array(gray_img)

    # Canny edge detection
    sig = 12
    l_thresh = 5
    h_thresh = 10
    edge_img = feature.canny(gray_img, sigma=sig, low_threshold=l_thresh, high_threshold=h_thresh)

    foo = Image.fromarray(edge_img)
    foo.show()

    # plt.savefig('outputs/hello_edges.png')
    # plt.show()


def segmentation():
    """
    Finding where the cars are on the gameboard (specefically the goal car).
    """
    # Retrieved from ChatGPT https://chat.openai.com/share/4d46a1c8-c082-4d77-87c7-9c4fe9f99e2e
    image = cv2.imread('IMG_1.png')

    # Convert the image to the HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds of the color range
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    # Threshold the image to get only the specified color range
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Apply morphological operations (optional)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Visualize the result
    result = cv2.bitwise_and(image, image, mask=mask)

    # Display the original image and the segmented regions
    cv2.imshow('Original Image', image)
    cv2.imshow('Segmented Regions', result)

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
