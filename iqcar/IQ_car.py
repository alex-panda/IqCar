import argparse
from typing import Tuple
from runTests import run_tests
from PIL import Image
import numpy as np
from skimage import filters, feature
import matplotlib.pyplot as plt
from skimage.segmentation import slic, quickshift
from skimage.color import label2rgb
import skimage as ski
import pygame.camera
from PIL import Image
from sklearn.cluster import MeanShift, estimate_bandwidth
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage.transform import hough_line, hough_line_peaks, probabilistic_hough_line
from skimage.draw import line as draw_line
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from skimage.filters import gaussian, threshold_otsu, threshold_minimum
from matplotlib import cm
from skimage.morphology import dilation, erosion
from scipy import stats

GOAL_CAR_HEX = "FF0000"

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
        'corner_detection' : corner_detection,
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
    return mask of bounding box of gameboard
    """
    for i in range(13, 18):
        img = Image.open(f'data/IMG_{i}.png')
        gray_img = img.convert('L')

        img = np.array(img)
        gray_img = np.array(gray_img)

        thresh_min = threshold_otsu(gray_img)
        binary_img = gray_img > thresh_min

        # Canny edge detection
        sig = 12
        l_thresh = 2
        h_thresh = 10
        edge_img = feature.canny(binary_img, sigma=sig, low_threshold=l_thresh, high_threshold=h_thresh)

        nonzero_y, nonzero_x = np.nonzero(edge_img)
        # top left
        min_y = np.min(nonzero_y, axis=0)
        min_x = np.min(nonzero_x, axis=0)
        # bottom right
        max_y = np.max(nonzero_y, axis=0)
        max_x = np.max(nonzero_x, axis=0)
        mask = bounding_box_mask(min_x, min_y, max_x, max_y, img)

        min_line_length = 100
        min_line_gap = 100
        # lines = probabilistic_hough_line(edge_img, line_length=min_line_length, line_gap=min_line_gap)

            
        # # Generating figure 1
        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        ax = axes.ravel()

        ax[0].imshow(img, cmap=cm.gray)
        ax[0].set_title('Input image')

        ax[1].imshow(edge_img, cmap=cm.gray)
        ax[1].set_title('Canny edges')
        ax[1].plot(
            min_y, min_x, color='cyan', marker='o', linestyle='None', markersize=6
        )
        ax[1].plot(
            max_y, max_x, color='cyan', marker='o', linestyle='None', markersize=6
        )

        ax[2].imshow(binary_img)

        # ax[2].imshow(edge_img * 0)
        # for line in lines:
        #     p0, p1 = line
        #     ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]))
        # ax[2].set_xlim((0, img.shape[1]))
        # ax[2].set_ylim((img.shape[0], 0))
        # ax[2].set_title('Probabilistic Hough')

        for a in ax:
            a.set_axis_off()

        plt.tight_layout()
        plt.savefig(f'outputs/probablistic_hough_{i}.png')
        plt.show()
    return mask

def corner_detection():
    """
    return corners as top left, top right, bottom left, bottom right
    """
    for i in range(13, 18):
        img = Image.open(f'data/IMG_{i}.png')
        gray_img = img.convert('L')
        img = np.array(img)
        gray_img = np.array(gray_img)

        print("foo")
        thresh_min = threshold_otsu(gray_img)
        binary_img = gray_img > thresh_min
        binary_img = clean_binary_image(binary_img)

        # Canny edge detection
        sig = 12
        l_thresh = 2
        h_thresh = 10
        edge_img = feature.canny(gray_img, sigma=sig, low_threshold=l_thresh, high_threshold=h_thresh)

        print(np.unique(edge_img))
        center_x, center_y = center_of_mass(binary_img)
        size_y, size_x = binary_img.shape

        # TODO: snap to nearest white pixel
        # top left
        nonzero_y, nonzero_x = np.nonzero(binary_img[0:center_y, 0:center_x])
        tl_y = np.min(nonzero_y, axis=0)
        tl_x = np.min(nonzero_x, axis=0)
        # top right
        nonzero_y, nonzero_x = np.nonzero(binary_img[0:center_y, center_x:size_x])
        tr_y = np.min(nonzero_y, axis=0)
        tr_x = np.max(nonzero_x, axis=0) + center_x
        # bottom left
        nonzero_y, nonzero_x = np.nonzero(binary_img[center_y:size_y, 0:center_x])
        bl_y = np.max(nonzero_y, axis=0) + center_y
        bl_x = np.min(nonzero_x, axis=0)
        # bottom right
        nonzero_y, nonzero_x = np.nonzero(binary_img[center_y:size_y, center_x:size_x])
        br_y = np.max(nonzero_y, axis=0) + center_y
        br_x = np.max(nonzero_x, axis=0) + center_x

        coords = np.array([[tl_x, tl_y], [tr_x, tr_y], [bl_x, bl_y],[br_x, br_y]])
        print(coords)

        print("end corner detection")

        fig, ax = plt.subplots(1, 2, figsize=(15, 6))
        ax[0].imshow(img, cmap=plt.cm.gray)
        ax[0].plot(
            center_x, center_y, color='cyan', marker='o', linestyle='None', markersize=6
        )
        ax[0].plot(
            coords[0][0],coords[0][1], color='red', marker='o', linestyle='None', markersize=6
        )
        ax[0].plot(
            coords[1][0],coords[1][1], color='green', marker='o', linestyle='None', markersize=6
        )
        ax[0].plot(
            coords[2][0],coords[2][1], color='yellow', marker='o', linestyle='None', markersize=6
        )
        ax[0].plot(
            coords[3][0],coords[3][1], color='pink', marker='o', linestyle='None', markersize=6
        )
        ax[1].imshow(binary_img, cmap=cm.gray)
        plt.show()
        return coords

def clean_binary_image(binary_image, k=25):
    footprint = np.ones((k, k))

    print("cleaning binary_image")
    fig, ax = plt.subplots(1, 2)
    processed_img = erosion(binary_image, footprint=footprint)
    # ax[0].imshow(processed_img, cmap=cm.gray)
    # ax[0].set_title('After Erosion')

    processed_img = dilation(processed_img, footprint=footprint)
    # ax[1].imshow(processed_img, cmap=cm.gray)
    # ax[1].set_title('After Dilation')

    # plt.show()
    return processed_img

# method interpolate_points retrieved from chatgpt
def interpolate_points(point1, point2, n):
    """
    Interpolate `n` points including the given start and end points.
    
    Parameters:
        point1 (tuple): (x1, y1) coordinates of the first point.
        point2 (tuple): (x2, y2) coordinates of the second point.
        n (int): Number of points to interpolate between the given points.
        
    Returns:
        list of tuples: List of interpolated (x, y) points.
    """
    # Unpack points
    x1, y1 = point1
    x2, y2 = point2
    
    # Calculate increments
    dx = (x2 - x1) / (n - 1)
    dy = (y2 - y1) / (n - 1)
    
    # Generate interpolated points
    interpolated_points = [(x1 + i * dx, y1 + i * dy) for i in range(n)]
    
    return interpolated_points

def identify_colors_of_chunks(transformed_segmented_image, transformed_corners):
    """
    Expecting a transformed_segmented_image and transformed_corners (4x2 array)

    Returns modal color of each chunk between the corners as a 6x6 array
    """

    top_points = interpolate_points(transformed_corners[0], transformed_corners[1], 7)
    bottom_points  = interpolate_points(transformed_corners[2], transformed_corners[3], 7)

    colors = np.zeros([6,6,3])

    for i in range(len(top_points) - 1):
        tr_points = interpolate_points(top_points[i], bottom_points[i], 7)
        br_points = interpolate_points(top_points[i+1], bottom_points[i+1], 7)

        for j in range(len(tr_points) - 1):
            chunk = transformed_segmented_image[tr_points[j]:br_points[j+1]]
            flat_chunk = chunk.reshape(-1, 3)
            unique, counts = np.unique(flat_chunk, return_counts=True)
            mode = unique[np.argmax(counts)]
            colors[j][i] = mode

    return colors



def bounding_box_mask(x1, y1, x2, y2, img):
    # Create an empty binary mask
    mask = np.zeros_like(img, dtype=np.int8)

    # Ensure x1 <= x2 and y1 <= y2
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    # Set the region between (x1, y1) and (x2, y2) to 1
    mask[y1:y2+1, x1:x2+1] = 1

    return mask

def center_of_mass(binary_image):
    # Create an array of coordinates for each pixel
    y_coords, x_coords = np.nonzero(binary_image)

    # Compute the center of mass
    center_y = np.mean(y_coords)
    center_x = np.mean(x_coords)

    return (int(center_x), int(center_y))

def segmentation():
    """
    Finding where the cars are on the gameboard (specifically the goal car).
    """
    # Sample Image of scikit-image package
    img = np.asarray(Image.open('data/IMG_14.png'))
    img = img_as_float(img[::2, ::2])
    labels, segmented_img = segment_image(img)
    # mask = get_edge_mask(Image.fromarray(img))

    plt.imshow(segmented_img)
    plt.savefig(f'outputs/segmentation_14.png')
    plt.show()

def segment_image(img : np.array):
    """
    params np.array image
    returns labels, segmented img average color
    """
    # Applying Simple Linear Iterative
    # Clustering on the image
    num_seg = 250
    compact = 10
    print("starting slic")
    segments_slic = slic(img, n_segments=num_seg, compactness=compact)
    print("end slic")
    
    # Converts a label image into
    # an RGB color image for visualizing
    # the labeled regions. 
    img_2 = label2rgb(segments_slic,
                        img,
                        kind = 'avg')
    return segments_slic, img_2


def parse_into_game_board(labels, segmented_img):
    """
    Turn images into the internal gameboard representation.
    """
    # segment

    # edge mask


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
