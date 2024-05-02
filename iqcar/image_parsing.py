import math
from typing import Tuple
from PIL import Image
import numpy as np
from skimage.segmentation import slic
from skimage.color import label2rgb
from PIL import Image
from skimage.filters import threshold_otsu
from skimage.morphology import dilation, erosion
from skimage.transform import ProjectiveTransform, warp
from skimage.util import img_as_float 
import matplotlib.pyplot as plt
from iqcar.car import Car
from iqcar.gameboard import Gameboard

DEBUG = False
SAVE_IMGS = False
SHOW_IMGS = False


# ============== HELPERS ==============

def clean_binary_image(binary_image, k=25):
    """
    Cleans a binary image using erosion and then dilation. default footpring is 25x25 square.

    parameters binary_image, k=25
    returns processed_image
    """
    footprint = np.ones((k, k))

    processed_img = erosion(binary_image, footprint=footprint)
    processed_img = dilation(processed_img, footprint=footprint)

    return processed_img

def center_of_mass(binary_image):
    """
    Return the center of mass of a binary image. Expecting a cleaned binary image of gameboard.

    parameters binary_image
    returns (center_x, centery_y)
    """
    # Create an array of coordinates for each pixel
    y_coords, x_coords = np.nonzero(binary_image)

    # Compute the center of mass
    center_y = np.mean(y_coords)
    center_x = np.mean(x_coords)

    return (int(center_x), int(center_y))

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

# ============== DOERS ==============

def corner_detection(img : Image):
    """
    finds the corners of the gameboard assuming the gameboard is oriented roughly straight on and not rotated.
    
    parameter img
    return corners as top left, top right, bottom left, bottom right
    """
    gray_img = img.convert('L')
    img = np.array(img)
    gray_img = np.array(gray_img)

    thresh_min = threshold_otsu(gray_img)
    binary_img = gray_img > thresh_min
    binary_img = clean_binary_image(binary_img)

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

    if DEBUG:
        plt.imshow(img, cmap=plt.cm.gray)
        plt.plot(
            center_x, center_y, color='cyan', marker='o', linestyle='None', markersize=6
        )
        plt.plot(
            coords[0][0],coords[0][1], color='red', marker='o', linestyle='None', markersize=6
        )
        plt.plot(
            coords[1][0],coords[1][1], color='green', marker='o', linestyle='None', markersize=6
        )
        plt.plot(
            coords[2][0],coords[2][1], color='yellow', marker='o', linestyle='None', markersize=6
        )
        plt.plot(
            coords[3][0],coords[3][1], color='pink', marker='o', linestyle='None', markersize=6
        )
        plt.set_title("corner_detection_debug")
        plt.show()
        if SAVE_IMGS:
            plt.savefig("outputs/corner_detection_debug.png")

    return coords



def segment_image(img : np.array):
    """
    params np.array image
    returns labels, segmented img as ndarray
    """
    # Applying Simple Linear Iterative
    # Clustering on the image
    num_seg = 250
    compact = 10
    segments_slic = slic(img, n_segments=num_seg, compactness=compact)
    
    # Converts a label image into
    # an RGB color image for visualizing
    # the labeled regions. 
    img_2 = label2rgb(segments_slic,
                        img,
                        kind = 'avg')
    return segments_slic, img_2

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


def identify_colors_of_chunks(transformed_segmented_image : np.array):
    """
    Expecting a transformed_segmented_image and transformed_corners (4x2 array)

    Returns modal color of each chunk between the corners as a 6x6 array
    """

    shape = transformed_segmented_image.shape
    top_points = interpolate_points(0, shape[1], 7) # x - col
    bottom_points  = interpolate_points(0, shape[0], 7) # y - row

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

    def is_eq(one: np.ndarray | None, two: np.ndarray | None, e: float) -> bool:
        if one is None: return False
        if two is None: return False
        if np.abs(two - one) > e: return True
        return False

    def in_bounds(array: np.ndarray, y: int, x: int) -> bool:
        return 0 <= y < array.shape[0] and 0 <= x < array.shape[1]

    def get(array: np.ndarray, y: int, x: int) -> np.ndarray | None:
        if in_bounds(array, y, x):
                return array[y, x, :]
        return None

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
    return Gameboard(None if len(s) == 0 else s[0], cars)

def parse_into_game_board(img : Image):
    """
    Turn images into the internal gameboard representation.
    """
    # segment
    gray_img = img.convert('L')
    img = np.asarray(img)
    handle_image(img, "input_image")
    img = img_as_float(img[::2, ::2])
    # segment
    labels, segmented_img = segment_image(img)
    segmented_img = np.array(segmented_img*255, dtype=np.uint8)
    handle_image(segmented_img, "segmented_image")
    
    gray_img  = Image.fromarray(segmented_img).convert('L')
    gray_img = np.array(gray_img, dtype=np.float32)
    handle_image(gray_img, "gray_segmented_image")


    # warp image
    square_img = normalize_board_square(segmented_img, gray_img)
    handle_image(square_img, "warped_image")

    # color chunks
    colors = identify_colors_of_chunks(square_img)
    colors = np.array(colors*255, dtype=np.uint8)
    handle_image(colors, "6x6_pixel_image")

    # board
    board = board_from_colors(colors)
    return board

def handle_image(img, title):
    plt.imshow(img)
    plt.set_title(title)

    if SAVE_IMGS:
        plt.savefig(f'outputs/{title}.png')
    if SHOW_IMGS:
        plt.show()