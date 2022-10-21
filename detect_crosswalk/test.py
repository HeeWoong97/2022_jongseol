from mistune import InlineParser
import numpy as np
import cv2 as cv2
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

def fixColor(img):
    return (cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

img = cv2.imread("test.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
plt.imshow(fixColor(blurred))
canny = cv2.Canny(blurred, 30, 300)
plt.imshow(fixColor(canny))

(cnts, _) = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
coins = img.copy()
cv2.drawContours(coins, cnts, -1, (255, 0, 0), 2)
plt.imshow(fixColor(coins))