import cv2
import numpy as np
import scipy.misc
from PIL import Image, ImageEnhance
from matplotlib import pyplot as plt

image = cv2.imread("crop0.jpg", cv2.IMREAD_GRAYSCALE)

image = cv2.bilateralFilter(image,9,75,75)
th2 = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,251,2)

cv2.imwrite("thresh.jpg", th2)

plt.imshow(th2, cmap='gray'), plt.axis("off")
plt.show()
#scipy.misc.imsave('enhanced1.png',image_enhanced)
im = cv2.imread("crop0.jpg", 0)
enhanced_im = cv2.equalizeHist(im)
enhanced_im = cv2.bilateralFilter(enhanced_im,9,75,200)
cv2.imwrite("enhanced.jpg", enhanced_im)

