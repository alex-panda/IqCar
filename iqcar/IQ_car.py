import argparse
import math
import os
from typing import Tuple

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from PIL import Image
import pygame.camera
from skimage import filters, feature
import skimage
from skimage.color import label2rgb
from skimage.draw import line as draw_line
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from skimage.filters import gaussian, threshold_otsu, threshold_minimum
from skimage.morphology import dilation, erosion
from skimage.segmentation import mark_boundaries, slic, quickshift, flood_fill
from skimage.transform import hough_line, hough_line_peaks, probabilistic_hough_line, ProjectiveTransform, warp
from skimage.util import img_as_float
from sklearn.cluster import MeanShift, estimate_bandwidth
from scipy import stats

from iqcar.car import Car
from iqcar.gameboard import Gameboard
from iqcar.runTests import run_tests
from iqcar.solver import BoardState

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
        'segmentation' : segmentation,
        'main' : main,
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

def corner_detection(gray_img: np.ndarray):
    """Find corners of the game board"""
    thresh_min = threshold_otsu(gray_img)
    binary_img = gray_img > thresh_min
    binary_img = clean_binary_image(binary_img)

    # Canny edge detection
    sig = 12
    l_thresh = 2
    h_thresh = 10
    edge_img = feature.canny(gray_img, sigma=sig, low_threshold=l_thresh, high_threshold=h_thresh)

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

    return coords

def clean_binary_image(binary_image, k=25):
    footprint = np.ones((k, k))

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
    interpolated_points = [(int(x1 + i * dx), int(y1 + i * dy)) for i in range(n)]
    
    return interpolated_points

def identify_colors_of_chunks(transformed_segmented_image : np.array):
    """
    Expecting a transformed_segmented_image and transformed_corners (4x2 array)

    Returns modal color of each chunk between the corners as a 6x6 array
    """

    shape = transformed_segmented_image.shape
    print(shape)
    top_points = interpolate_points((0,0), (0, shape[1]-1), 7) # x - col
    bottom_points  = interpolate_points((shape[0]-1,0), (shape[0]-1,shape[1]-1), 7) # y - row

    # print(f"top points: {top_points}")
    # print(f"bottom points: {bottom_points}")

    colors = np.zeros([6,6,3])

    for i in range(len(top_points) - 1):
        tl_points = interpolate_points(top_points[i], bottom_points[i], 7)
        br_points = interpolate_points(top_points[i+1], bottom_points[i+1], 7)

        # print(f"top left points: {tl_points}")
        # print(f"bottom right points: {br_points}")

        for j in range(len(tl_points) - 1):
            # print(f"range: {tl_points[j][0]}:{br_points[j+1][0]}, {tl_points[j][1]}:{br_points[j+1][1]}")
            chunk = transformed_segmented_image[tl_points[j][0]:br_points[j+1][0], tl_points[j][1]:br_points[j+1][1]]

            chunk_show = np.array(chunk*255, dtype=np.uint8)
            #plt.imshow(chunk_show)
            #plt.show()

            flat_chunk = chunk.reshape(-1, 3)
            unique, counts = np.unique(flat_chunk, return_counts=True, axis=0)
            mode = unique[np.argmax(counts)]
            colors[j][i] = mode

    return colors


def draw_chunk_lines(transformed_segmented_image : np.array):
    """
    Expecting a transformed_segmented_image and transformed_corners (4x2 array)

    Returns modal color of each chunk between the corners as a 6x6 array
    """
    plt.imshow(transformed_segmented_image)
    shape = transformed_segmented_image.shape
    print(shape)
    top_points = interpolate_points((0,0), (0, shape[1]-1), 7) # x - col
    bottom_points  = interpolate_points((shape[0]-1,0), (shape[0]-1,shape[1]-1), 7) # y - row

    left_points = interpolate_points((0,0), (shape[0]-1, 0), 7) # x - col
    right_points  = interpolate_points((0, shape[1]-1), (shape[0]-1,shape[1]-1), 7) # y - row

    # print(f"top points: {top_points}")
    # print(f"bottom points: {bottom_points}")

    colors = np.zeros([6,6,3])

    for i in range(len(top_points) - 1):
        tl_points = interpolate_points(top_points[i], bottom_points[i], 7)
        br_points = interpolate_points(top_points[i+1], bottom_points[i+1], 7)

        foo_points = interpolate_points(left_points[i], right_points[i], 7)
        bar_points = interpolate_points(left_points[i+1], right_points[i+1], 7)

        for j in range(len(tl_points)):
            x = [tl_points[j][1], br_points[j][1]]
            y = [tl_points[j][0], br_points[j][0]]
            plt.plot(x, y, color="red", linewidth=3) 

            x = [foo_points[j][1], bar_points[j][1]]
            y = [foo_points[j][0], bar_points[j][0]]
            plt.plot(x, y, color="red", linewidth=3) 

            
            

    plt.savefig("square_with_lines.png")
    plt.show()


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

def segment_image(img : np.ndarray, num_seg = 250, compact = 10):
    """
    params np.array image
    returns labels, segmented img average color
    """
    # Applying Simple Linear Iterative
    # Clustering on the image
    segments_slic = slic(img, n_segments=num_seg, compactness=compact)
    
    # Converts a label image into
    # an RGB color image for visualizing
    # the labeled regions. 
    img_2 = label2rgb(segments_slic, img, kind='avg')

    return segments_slic, img_2

def normalize_board_square(segmented_img: np.ndarray, gray_img: np.ndarray, buffer=3) -> np.ndarray:
    """Transform a game board image into a square.

    Args:
        img: the image from which to extract the gameboard

    Returns:
        an ndarray which contains a warped image of the
        gameboard, transformed to a square
    """
    corners = corner_detection(gray_img)
    bbox = np.array([[np.min(a), np.max(a)] for a in corners.T])
    sidelen = np.min(np.abs(bbox[1] - bbox[0]))
    square = np.array([[0, 0], [sidelen, 0], [0, sidelen], [sidelen, sidelen]])
    hom = computeHomography(corners, square)
    square_img = warp(segmented_img, ProjectiveTransform(matrix=np.linalg.inv(hom)))
    return square_img[buffer:sidelen-buffer, buffer:sidelen-buffer]

def parse_into_game_board():
    """
    Turn images into the internal gameboard representation.
    """
    # segment
    img = Image.open('data/IMG_17.png')
    gray_img = img.convert('L')
    img = np.asarray(img)
    img = img_as_float(img[::2, ::2])
    # segment
    _labels, segmented_img = segment_image(img)
    segmented_img = np.array(segmented_img*255, dtype=np.uint8)
    # plt.imshow(segmented_img)
    # plt.show()
    
    gray_img  = Image.fromarray(segmented_img).convert('L')
    gray_img = np.array(gray_img, dtype=np.float32)
    # print(np.unique(gray_img))
    # plt.imshow(gray_img)
    # plt.show()
    

    # warp image
    square_img = normalize_board_square(segmented_img, gray_img)
    plt.imshow(square_img)
    plt.savefig("square_img.png")
    plt.show()

    draw_chunk_lines(square_img)

    # color chunks
    colors = identify_colors_of_chunks(square_img)
    colors = np.array(colors*255, dtype=np.uint8)

    #plt.imshow(colors)
    #plt.show()

    return board_from_colors(colors)

def board_from_colors(colors: np.ndarray[np.uint8]) -> Gameboard:
    """
    Creates a gameboard from the given colors.

    Assumptions:
     - there is always at least one background square
     - cars are either 2 units long or 3

    Args:
        colors: a 6x6x3 numpy array containing the colors of the game board
    Returns:
        Gameboard: the gameboard
    """
    red = np.array([240, 70, 50], dtype=np.float64)

    def in_bounds(array: np.ndarray, y: int, x: int) -> bool:
        return 0 <= y < array.shape[0] and 0 <= x < array.shape[1]

    def color_dist(color1: np.ndarray | None, color2: np.ndarray | None) -> float:
        if color1 is None: return np.inf
        if color2 is None: return np.inf
        c1r, c1g, c1b = np.array(color1, dtype=np.float64)
        c2r, c2g, c2b = np.array(color2, dtype=np.float64)

        rmean = (c1r + c2r) / 2.0
        r = c1r - c2r
        g = c1g - c2g
        b = c1b - c2b
        return math.sqrt((int((512+rmean)*r*r) >> 8) + int(4*g*g) + (int((767-rmean)*b*b) >> 8))

    e = 30

    background_squares: set[Tuple[int, int]] = set()

    # first, figure out what are the background squares
    # All background squares are assumed to be a shade of white/gray/black
    for y in range(colors.shape[0]):
        for x in range(colors.shape[1]):
            r,g,b = colors[y, x, :]
            r,g,b = int(r), int(g), int(b)
            if (np.abs(r - g) <= e) and (np.abs(g - b) <= e) and (np.abs(b - r) <= e):
                background_squares.add((y, x))
                colors[y, x, :] = 0
    
    cars: list[Car] = []
    taken_squares: set[Tuple[int, int]] = set(background_squares)

    # go through every point of the board
    for y in range(colors.shape[0]):
        for x in range(colors.shape[1]):

            # if the square is already taken, then ignore it
            p   = (y, x)
            if p in taken_squares:
                continue

            possible_cars = []

            # Get possible cars for square (y, x)
            for h, oy, ox in [(False, 1, 0), (True, 0, 1)]:

                pn  = (y+oy, x+ox)
                pnn = (y+oy+oy, x+ox+ox)

                if pn not in taken_squares and in_bounds(colors, *pn):

                    if pnn not in taken_squares and in_bounds(colors, *pnn):
                        possible_cars.append((h, (p, pn), (p, pn, pnn)))
                    else:
                        possible_cars.append((h, (p, pn), None))

            if len(possible_cars) == 0:
                # all cars out of bounds
                continue

            def car_key(c):
                (y , x ) = c[1][0]
                (yn, xn) = c[1][1]
                return color_dist(colors[y, x], colors[yn, xn])

            # sort cars by how far the first color is from the next color in the car
            sorted_cars = sorted(possible_cars, key=car_key)

            (h, car2, car3) = sorted_cars[0]

            # check if car should be 3-long
            car = car2
            if car3 is not None:
                (yn , xn ) = car3[1]
                (ynn, xnn) = car3[2]
                if color_dist(colors[yn, xn], colors[ynn, xnn]) <= 100:
                    car = car3
                
            # update background squares so that later iterations do not use the values already in use
            taken_squares = taken_squares.union(set(car))

            # append the new car
            cars.append(Car(car[0][1], car[0][0], h, len(car)))

    #print("Background:")
    #print(Gameboard.board_str([Car(x, y, True, 1) for (y, x) in background_squares]))

    #print("Result:")
    #print(Gameboard.board_str(cars))

    s = sorted(cars, key=lambda c: color_dist(colors[c.y, c.x], red))
    #if len(s) > 0:
    #    print(s[0] is cars[2])
    return Gameboard(None if len(s) == 0 else s[0], cars)

def test_main() -> float:
    import re
    f = re.compile('[0123456789]+')
    def key(file):
        ints = f.findall(file)
        if len(ints) == 0:
            return np.inf
        return float(ints[0])

    main([
        Image.open(os.path.join('.', 'data', file))
        for file in sorted(os.listdir(os.path.join('.', 'data')), key=key)
    ])
    return 0.0

def main(images: None | list[Image.Image] = None):

    if images is None:
        return test_main()

    images: list[np.ndarray] = [img_as_float(img) for img in images]

    for (i, img) in enumerate(images):
        if i != 15:
            continue

        #Image.fromarray(img).show()

        print(f"Image {i}")

        # segment
        _labels, segmented_img = segment_image(img)
        segmented_img = np.array(segmented_img*255, dtype=np.uint8)
        # plt.imshow(segmented_img)
        # plt.show()

        gray_img  = Image.fromarray(segmented_img).convert('L')
        gray_img = np.array(gray_img, dtype=np.float64)
        # print(np.unique(gray_img))
        # plt.imshow(gray_img)
        # plt.show()

        # warp image
        square_img = normalize_board_square(segmented_img, gray_img)
        # plt.imshow(square_img)
        # plt.show()

        # color chunks
        colors = identify_colors_of_chunks(square_img)
        colors = np.array(colors*255, dtype=np.uint8)

        plt.imshow(colors)
        plt.show()

        board = board_from_colors(colors)
        print(board)
        for b in board.into_state().solve():
            new_gb = b.to_gameboard()
            print(new_gb)

def computeHomography(src_pts_nx2: np.ndarray, dest_pts_nx2: np.ndarray) -> np.ndarray:
    '''
    Compute the homography matrix.
    Arguments:
        src_pts_nx2: the coordinates of the source points (nx2 numpy array).
        dest_pts_nx2: the coordinates of the destination points (nx2 numpy array).
    Returns:
        H_3x3: the homography matrix (3x3 numpy array).
    '''
    A = np.zeros((src_pts_nx2.shape[0] * 2, 9))
    for pt, ((x_s, y_s), (x_d, y_d)) in enumerate(zip(src_pts_nx2, dest_pts_nx2)):
        for i in range(2):
            r = pt * 2 + i
            A[r, :6] = [
                x_s * ((r + 1) % 2),
                y_s * ((r + 1) % 2),
                (r + 1) % 2,
                x_s * (r % 2),
                y_s * (r % 2),
                r % 2
            ]
            if r % 2 == 0:
                A[r, 6:] = [-x_d * x_s, -x_d * y_s, -x_d]
            else:
                A[r, 6:] = [-y_d * x_s, -y_d * y_s, -y_d]

    # Get the homography matrix by decomposing to U * S * VT and getting the last row of VT
    _, _, Vh = np.linalg.svd(-A)
    h = Vh[-1, :]
    H = h.reshape((3, 3))
    H /= H[-1, -1]
    return H


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


