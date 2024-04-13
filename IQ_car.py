import argparse
from typing import Tuple
from runTests import run_tests
from PIL import Image
import numpy as np
from skimage import filters, feature
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.data import astronaut
from skimage.color import label2rgb
import skimage as ski
import pygame.camera
from PIL import Image

def IQCar():
    parser = argparse.ArgumentParser(
        description='Execute a specific test or all tests.')
    parser.add_argument(
        'function_name', type=str, nargs='?', default='all',
        help='Name of the function to test or "all" to execute all the registered functions')
    args = parser.parse_args()

    # Call test harness
    fun_handles = {
        'capture_image' : capture_image,
        'edge_detection' : edge_detection,
        'segmentation' : segmentation
    }
    run_tests(args.function_name, fun_handles)

def capture_image(cam:int=0, dims:Tuple[int, int]=(640, 480), show:bool=True) -> Image.Image | None:
    """
    Captures an image from a webcam and returns it as a PIL image.

    `cam`: the index of the camera to capture the image from
    `dims`: the dimensions of the image you want
    `show`: whether to call `Image.show()` to show the image before returning it

    returns the image if captured or `None` if there was no camera `cam` to capture the image
    """

    pygame.camera.init()
    camlist = pygame.camera.list_cameras()

    if len(camlist) >= cam:
        cam = pygame.camera.Camera(camlist[0], dims)

        cam.start()
        image = cam.get_image()

        pil_string_image = pygame.image.tostring(image, "RGBA", False)
        im = Image.frombytes("RGBA", dims, pil_string_image)
    
        if show:
            im.show()

        return im

    return None

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
    # Setting the plot size as 15, 15
    plt.figure(figsize=(10,10))
    
    # Sample Image of scikit-image package
    img = np.asarray(Image.open('data/IMG_1.png'))
    
    # Applying Simple Linear Iterative
    # Clustering on the image
    num_seg = 50
    compact = 50
    segments = slic(img, n_segments=num_seg, compactness=compact)
    plt.subplot(1,2,1)
    
    # Plotting the original image
    plt.imshow(img)
    plt.subplot(1,2,2)
    
    # Converts a label image into
    # an RGB color image for visualizing
    # the labeled regions. 
    img_2 = label2rgb(segments, img, kind = 'avg')
    plt.imshow(img_2)

    plt.show()

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
