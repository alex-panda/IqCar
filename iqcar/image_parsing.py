from PIL import Image
import numpy as np
from skimage.segmentation import slic
from skimage.color import label2rgb
from PIL import Image
from skimage.filters import threshold_otsu
from skimage.morphology import dilation, erosion

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

    return coords



def segment_image(img : np.array):
    """
    params np.array image
    returns labels, segmented img average color
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
