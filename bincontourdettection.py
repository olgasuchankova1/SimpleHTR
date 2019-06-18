# import the necessary packages

from skimage import exposure
import numpy as np
import argparse
import imutils
import cv2
from PIL import Image

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-q", "--query", default = 'enhanced.jpg',
                help="Path to the query image")
args = vars(ap.parse_args())

# load the query image, compute the ratio of the old height
# to the new height, clone it, and resize it
image = cv2.imread(args["query"])
orig = image.copy()

# convert the image to grayscale, blur it, and find edges
# in the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 11, 17, 17)
edged = cv2.Canny(gray, 30, 200)
cv2.imshow("Game Boy Screen", edged)


# find contours in the edged image, keep only the largest
# ones, and initialize our screen contour
cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]

sizer = Image.open('thresh.jpg')
width, height = sizer.size
canvas = np.ones((height,width,3), np.uint8)*255

cv2.drawContours(canvas, cnts, -1, (0, 0, 0), 2)
cv2.imwrite("canvas.jpg", canvas)
cv2.waitKey(0)