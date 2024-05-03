# IqCar

From top directory run:

python iqcar/IQ_car.py main

and you should see a sequence of images as well as the board being solved from the terminal output!

Take a look into the /outputs folder if you want to take a look at what more of our sample images look like, but our demo is hardcoded to run on IMG_18.png which is fully functional. 

If you'd like to see the code run on more examples, you can go to IQ_car.py and change the constant IMG_18 to false, which will then run our system on a larger subset of available images. The system will output wether or not the system failed (crashed) while attempting to parse the image, or if it didn't crash. The image may crash if the corners that were detected resulted in a square larger than the canvas, or if the 6x6 pixel array provided to the gameboard extracter does not have a clear distinction between "background" pixels and "car pixels". It will not crash if the 6x6 pixel array is simply wrong, and will attempt to solve the board no matter what. Thankfully, from the 6x6 pixel array it should be pretty obvious which case is which. There are many images which do not produce good boards but pass all the way through, and only 1-2 that are correctly read. 