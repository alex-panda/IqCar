# IqCar

By Casey Ford, Alex Peseckis, Jack Stanek

![Presentation Link](./CS%20766%20Computer%20Vision%20Project.pdf)

![GitHub Repo](https://github.com/alex-panda/IqCar)

## Introduction

### What is IQ Car?

IQ Car is a knockoff of the more well-known board game Rush Hour. The goal of the game is to move a goal car off of the game board while only moving it forward or backward. The challenge of the game comes from the fact that there are typically other cars in the way of the goal car who also can only be moved forward and backward and who can block the goal car from exiting the game board. This causes the game to be like a puzzle where only a specific sequence of moves using a specific sequence of cars will allow the goal car to be moved off of the gameboard.

### Example and Problem Statement

![An image of IQ car](./imgs/iq_car_example.png)

In the above example game board, the goal is to move the red car to the right and off the game board through the slot in the game board's wall. The issue is that multiple cars lie in its path and must be moved out of the way before the it can reach the slot. While this may be a simple example of the game, much more difficult setups can be imagined in which the average person may have great difficulty solving the board. Even worse, they may become stuck and want the solution for future reference but not know anyone good enough at the game to solve it for them. Our project empowers users to overcome such situations by allowing them to upload a photo of their game board and have it solved for them by the computer.

## Methodology

Our project uses an image analysis pipeline to construct an internal representation of a board state from some image,
which it then feeds into a simple Rush Hour game solver. The first step of the image analysis pipeline is
segmentation. After reading an image from the camera, we segment it into color bins. We know that our game board is
white, and we have constrained the possible images to be against a black background, so we can identify the board as a
blob, as well as the cars of various colors on the board. We used SLIC to perform a k-means clustering of the colors in
the image in order to segment the image. After segmenting the image, we use a corner detection algorithm to identify
the four corners of the board. Since we knew that the shape of the board would be a quadrangle in the image space, we
identify the outer corners of the board by dividing it into four quadrants along its axes, and then finding the corners
at the extremal points in each quadrant.

After finding the corners, we then compute a projective transformation to a square from the quadrangle shape of the
board in image space. This allows us to normalize the board shape across arbitrary camera orientations in order to
identify the piece positions consistently. We identify the piece positions by dividing the transformed square board into
an _n_ by _n_ grid, and determining the modal color in each bin. We convert each bin's color to the HSV color space, in
which colors with sufficiently close modal color hues are assumed to be part of the same car. We make attempts to filter
out erroneous grid square assignments based on assumptions about the car shapes (i.e. no L-shaped cars, cars are either
2 or 3 squares long) in order to generate a valid board state.

After determining the board state, we pass this representation into a Rush Hour solver. Unfortunately, there is no known
algorithm which can solve general _n_ by _n_ Rush Hour boards (it is in fact PSPACE-complete), but since our real-life
board is bounded at size 6 by 6, a simple brute-force solution with backtracking turns out to be practical since the
search space is sufficiently small. This solver simply recursively enumerates all single move perturbations of the board
state, given a single input, and returns a sequence of moves which would solve the game if such a sequence exists, or
returns an error if the board is unsolvable.

## Results


## Demos


