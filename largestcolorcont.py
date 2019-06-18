import numpy as np
import cv2

# load the image
image = cv2.imread("bucket.jpg", 1)

# red color boundaries (R,B and G)
lower = [0, 0, 0]
upper = [150, 150, 150]

# create NumPy arrays from the boundaries
lower = np.array(lower, dtype="uint8")
upper = np.array(upper, dtype="uint8")

# find the colors within the specified boundaries and apply
# the mask
mask = cv2.inRange(image, lower, upper)
output = cv2.bitwise_and(image, image, mask=mask)


ret,thresh = cv2.threshold(mask, 40, 255, 0)
contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

if len(contours) != 0:
    # draw in blue the contours that were founded
    cv2.drawContours(output, contours, -1, 255, 3)

    #find the biggest area
    c = max(contours, key = cv2.contourArea)
    cv2.imshow("res", output)
    (x,y), radius = cv2.minEnclosingCircle(c)
    center = (int(x), int(y))
    radius = int(radius)
    # draw the book contour (in green)
   # cv2.circle(output,center,radius,(0,255,0),2)

# show the images
#cv2.imshow("Result", np.hstack([image, output]))

cv2.waitKey(0)